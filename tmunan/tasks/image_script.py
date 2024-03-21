import uuid
import time
import datetime
import threading
from typing import List
from pathlib import Path
from copy import deepcopy

from tmunan.common.event import Event
from tmunan.common.log import get_logger
from tmunan.imagine.sd_lcm.lcm_bg_task import TaskType
from tmunan.imagine.image_generator import ImageGenerator
from tmunan.api.pydantic_models import ImageSequence, ImageSequenceScript, ImageInstructions, SequencePrompt


class ImageScript:

    def __init__(self, image_gen: ImageGenerator, cache_dir):

        # env
        self.cache_dir = cache_dir
        self.seq_dir = None
        self.logger = get_logger(self.__class__.__name__)

        # imaging
        self.image_gen = image_gen
        self.image_gen.on_image_ready += self.process_ready_image
        self.last_image_path = None

        # text
        self.external_text_prompt = None

        # internal
        self.stop_requested = False
        self.sync_event = threading.Event()

        # events
        self.on_image_ready = Event()

    def stop(self):
        self.stop_requested = True

    def set_text_prompt(self, text_prompt):
        print(f'Setting external_text_prompt to: {text_prompt}')
        self.external_text_prompt = text_prompt

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

    def run_image_sequence(self, seq: ImageSequence, img_config: ImageInstructions, seq_id, parent_dir=None):

        self.logger.info(f'Starting image sequence {seq=}')
        self.logger.info(f'Image configuration {img_config=}')

        # prepare dir
        self.seq_dir = Path(f'{parent_dir if parent_dir else self.cache_dir}/seq_{seq_id}/')
        Path.mkdir(Path(self.seq_dir), exist_ok=True, parents=True)
        self.logger.info(f'Saving images in {self.seq_dir}')

        # verify seed
        if not img_config.seed:
            img_config.seed = self.image_gen.get_random_seed()

        # iterate as many times as requested
        for i in range(0, seq.num_images):

            # check if we should stop
            if self.stop_requested:
                break

            effective_prompts = deepcopy(seq.prompts)
            if self.external_text_prompt:
                effective_prompts.insert(0, self.external_text_prompt)

            # gen prompt for current sequence progress
            prompt = self.gen_seq_prompt(effective_prompts, (i / seq.num_images * 100))
            print(f'Generating image {i} with prompt: {prompt}')

            # gen image
            start_time = time.time()
            self.sync_event.clear()

            # check transition type
            if seq.transition == TaskType.Image2Image and self.last_image_path is not None:
                self.image_gen.img2img(
                    image_url=self.last_image_path,
                    prompt=prompt,
                    num_inference_steps=img_config.num_inference_steps,
                    guidance_scale=img_config.guidance_scale,
                    strength=img_config.strength,
                    height=img_config.height, width=img_config.width
                )

            else:
                self.image_gen.txt2img(
                    prompt=prompt,
                    num_inference_steps=img_config.num_inference_steps,
                    guidance_scale=img_config.guidance_scale,
                    height=img_config.height, width=img_config.width,
                    seed=img_config.seed,
                    randomize_seed=False
                )

            # wait until image is ready
            self.sync_event.wait()

            # check elapsed time
            elapsed_time = time.time() - start_time
            sleep_time = (img_config.key_frame_duration * img_config.key_frame_repeat) - elapsed_time + 1
            if sleep_time > 0:
                print(f'Sleeping for: {sleep_time}')
                time.sleep(sleep_time)

    def process_ready_image(self, image):

        # release sync event
        self.sync_event.set()

        # generate timestamp
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        # save image to disk
        image_path = self.seq_dir / f'{now_ts}.png'
        image.save(str(image_path))

        # save last image path
        self.last_image_path = str(image_path)

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

                # check if we should stop
                if self.stop_requested:
                    break

                # gen seq id
                seq_id = str(uuid.uuid4())[:8]

                # make a copy of the original sequence
                effective_seq = deepcopy(seq)

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
                config.seed = self.image_gen.get_random_seed()

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
