import uuid
import datetime
import threading
from typing import List
from pathlib import Path
from copy import deepcopy

from tmunan.common.event import Event
from tmunan.listen.asr import ASR
from tmunan.imagine.txt2img import Txt2Img
from tmunan.api.pydantic_models import ImageSequence, ImageSequenceScript, ImageInstructions, SequencePrompt


class ImageScript:

    def __init__(self, txt2img: Txt2Img, asr: ASR, cache_dir):

        # env
        self.cache_dir = cache_dir
        self.seq_dir = None

        # generators
        self.asr = asr
        self.txt2img = txt2img
        self.txt2img.on_image_ready += self.process_ready_image

        # internal
        self.stop_requested = False
        self.sync_event = threading.Event()

        # events
        self.on_image_ready = Event()

    def stop(self):
        self.stop_requested = True
    #
    # def start_sequence(self, seq: ImageSequence, config: ImageInstructions, seq_id, parent_dir=None):
    #
    #     # init stop flag
    #     self.stop_requested = False
    #
    #     # start generating sequence
    #     self.run_image_sequence(seq, config, seq_id, parent_dir)
    #
    # def start_script(self, script: ImageSequenceScript, config: ImageInstructions, script_id):
    #
    #     # inits flags
    #     self.stop_requested = False
    #
    #     # run script
    #     self.run_script(script, config, script_id)

    def run_image_sequence(self, seq: ImageSequence, config: ImageInstructions, seq_id, parent_dir=None):

        # prepare dir
        self.seq_dir = Path(f'{parent_dir if parent_dir else self.cache_dir}/seq_{seq_id}/')
        Path.mkdir(Path(self.seq_dir), exist_ok=True, parents=True)

        # verify seed
        if not config.seed:
            config.seed = self.txt2img.get_random_seed()

        # iterate as many times as requested
        for i in range(0, seq.num_images):

            # check if we should stop
            if self.stop_requested:
                break

            # generate dynamic text prompt
            effective_prompts = seq.prompts
            recognized_phrase_list = self.asr.consume_text()
            if recognized_phrase_list:
                for phrase in recognized_phrase_list:
                    effective_prompts.append(SequencePrompt(text=phrase, start_weight=1.2, end_weight=1.2))

            # gen prompt for current sequence progress
            prompt = self.gen_seq_prompt(effective_prompts, (i / seq.num_images * 100))
            print(f'Generating image {i} with prompt: {prompt}')

            # gen image
            self.sync_event.clear()
            self.txt2img.txt2img(
                prompt=prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height, width=config.width,
                seed=config.seed,
                randomize_seed=False
            )

            # wait until image is ready
            self.sync_event.wait()

    def process_ready_image(self, image):

        # release sync event
        self.sync_event.set()

        # generate timestamp
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        # save image to disk
        image_path = self.seq_dir / f'{now_ts}.png'
        image.save(str(image_path))

        # notify image ready
        self.on_image_ready.notify({
            'image_path': str(image_path)
        })

    def run_script(self, script: ImageSequenceScript, config: ImageInstructions, script_id):

        # prepare dir
        script_dir = f'{self.cache_dir}/script_{script_id}/'
        Path.mkdir(Path(script_dir), exist_ok=True, parents=True)

        # run until stopped
        continuity_prompts = None
        while not self.stop_requested:

            # iterate as many times as requested
            for i, seq in enumerate(script.sequences):

                # make a copy of the original sequence
                effective_seq = deepcopy(seq)

                # check if we should stop
                if self.stop_requested:
                    break

                # gen seq id
                seq_id = str(uuid.uuid4())[:8]

                # if we have continuity prompts from previous loop
                if i == 0 and continuity_prompts:
                    effective_seq.prompts.extend(continuity_prompts)
                    continuity_prompts = None

                # if this is NOT the first sequence
                if i > 0:

                    # iterate prompts from previous sequence
                    reversed_prompts = self.reverse_prompt_weights(script.sequences[i - 1].prompts)
                    effective_seq.prompts.extend(reversed_prompts)

                # display sequence
                self.run_image_sequence(effective_seq, config, seq_id=seq_id, parent_dir=script_dir)

            # stop, unless loop requested
            if script.loop:

                # generate new seed
                config.seed = self.txt2img.get_random_seed()

                # take last sequence prompts to loop back into first sequence smoothly
                continuity_prompts = self.reverse_prompt_weights(script.sequences[-1].prompts)

            else:
                break

    @classmethod
    def reverse_prompt_weights(cls, prompts: List[SequencePrompt]):

        # new list
        reversed_prompt_list = []

        for p in prompts:

            # reverse weight trajectories
            reversed_prompt = deepcopy(p)
            reversed_prompt.start_weight = reversed_prompt.end_weight
            reversed_prompt.end_weight = 0.0

            # add reversed prompt to new sequence
            reversed_prompt_list.append(reversed_prompt)

        return reversed_prompt_list

    @classmethod
    def gen_seq_prompt(cls, prompts, prog: float):

        def calc_weight(p):
            weight_step = (p.end_weight - p.start_weight) / 100  # How much is one percent?
            return p.start_weight + (weight_step * prog)  # How progressed is this sequence?

        def format_prompt(text, weight):
            return text if weight == 1.0 else f'({text}){round(weight, 3)}'

        # build master prompt string
        return ', '.join(format_prompt(p.text, calc_weight(p)) for p in prompts)
