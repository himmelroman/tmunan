import uuid
import time
import datetime
import threading
import numpy as np
from typing import List
from pathlib import Path
from copy import deepcopy

from diffusers.utils import load_image
from pydantic import BaseModel

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
        self.last_image_url = None
        self.last_image_path = None
        self.image_config = None
        self.script_config = None

        # text
        self.external_text_prompt = None

        # internal
        self.stop_requested = False
        self.feed_thread = None
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

    def run_image_sequence(self, seq: ImageSequence, global_img_config: ImageInstructions, seq_id, parent_dir=None):

        self.logger.info(f'Starting image sequence {seq=}')
        self.logger.info(f'Image configuration {global_img_config=}')

        # prepare dir
        self.seq_dir = Path(f'{parent_dir if parent_dir else self.cache_dir}/seq_{seq_id}/')
        Path.mkdir(Path(self.seq_dir), exist_ok=True, parents=True)
        self.logger.info(f'Saving images in {self.seq_dir}')

        # save image config
        self.image_config = global_img_config
        self.feed_thread = None

        # verify seed
        seed = self.get_model_attribute(seq.img_config, global_img_config, 'seed')
        if not seed:
            seed = self.image_gen.get_random_seed()

        # precalculate prompt weights
        for p in seq.prompts:
            # calculate unless already specified
            if p.weight_list is None:
                p.weight_list = np.linspace(p.start_weight, p.end_weight, seq.num_images, endpoint=True)

        # start with base image if provided
        if seq.base_image_url:
            self.last_image_url = None
        # if seq.transition == TaskType.Image2Image and seq.base_image_url:
        #     image = load_image(str(seq.base_image_url))
        #     image.resize((global_img_config.width, global_img_config.height))
        #     self.process_ready_image(seq.base_image_url, image)

        # iterate as many times as requested
        for i in range(0, seq.num_images):

            # check if we should stop
            if self.stop_requested:
                break

            # mark the time before start of image generation
            img_gen_start_time = time.time()
            self.sync_event.clear()

            # check sequence task type
            if seq.transition == TaskType.Text2Image:

                effective_prompts = deepcopy(seq.prompts)
                if self.external_text_prompt:
                    effective_prompts.insert(0, self.external_text_prompt)

                # gen prompt for current sequence progress
                prompt = self.gen_seq_prompt(effective_prompts, i)

                # gen image
                self.logger.info(f'Generating image {i} with prompt: {prompt}')
                self.image_gen.txt2img(
                    prompt=prompt,
                    guidance_scale=self.get_model_attribute(seq.img_config, global_img_config, 'guidance_scale'),
                    num_inference_steps=self.get_model_attribute(seq.img_config, global_img_config, 'num_inference_steps'),
                    height=global_img_config.height, width=global_img_config.width,
                    seed=seed, randomize_seed=seed is None
                )

            # check sequence task type
            elif seq.transition == TaskType.Image2Image:

                # pick base image
                base_image_url = self.last_image_url or seq.base_image_url

                # gen prompt for current sequence progress
                prompt = self.gen_seq_prompt(seq.prompts, i)

                # determine seed
                seq_seed = seq.img_config.model_dump().get('seed')

                self.logger.info(f'Generating image from: {base_image_url} with prompt: {prompt}, '
                                 f'strength: {self.get_model_attribute(seq.img_config, global_img_config, "strength")}')
                self.image_gen.img2img(
                    prompt=prompt,
                    image_url=base_image_url,
                    strength=self.get_model_attribute(seq.img_config, global_img_config, 'strength'),
                    guidance_scale=self.get_model_attribute(seq.img_config, global_img_config, 'guidance_scale'),
                    num_inference_steps=self.get_model_attribute(seq.img_config, global_img_config, 'num_inference_steps'),
                    height=global_img_config.height, width=global_img_config.width,
                    seed=seq_seed, randomize_seed=seq_seed is None
                )

            # wait until image is ready
            self.sync_event.wait()

            # to keep RT - we need to sleep some time
            if self.script_config.keep_rtf:
                sleep_time = self.calc_rtf_sleep_time(global_img_config, img_gen_start_time)
                if sleep_time > 0:
                    self.logger.info(f'Sleeping for: {sleep_time}')
                    time.sleep(sleep_time)

    @staticmethod
    def calc_rtf_sleep_time(img_config: ImageInstructions, img_gen_start_time) -> int:

        # check elapsed time since image generation request
        # Image Gen:  |------|
        # Image DL:           |-|
        # Video Gen:             |---|
        # Video DL:                   |-|
        # Video Play:                    |-----|-----|-----|      (kf_repeat)
        #             S      E
        # Req. Sleep:        |--------------------|?

        image_gen_time = time.time() - img_gen_start_time
        video_play_time = img_config.key_frame_period * img_config.key_frame_repeat
        sleep_time = video_play_time  # - image_gen_time

        return sleep_time

    def process_ready_image(self, image_url, image):

        # generate timestamp
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        # save image to disk
        image_path = self.seq_dir / f'{now_ts}.png'
        image.save(str(image_path))

        # save last image
        self.last_image_path = image_path
        self.last_image_url = image_url
        self.logger.info(f'Updated last image: {self.last_image_url}')

        # output image
        self.feed_image()

        # release sync event
        self.sync_event.set()

        # start thread
        # if not self.feed_thread:
        #     self.feed_thread = threading.Thread(target=self.feed_thread_worker)
        #     self.feed_thread.start()

    def feed_thread_worker(self):

        # loop while script is running
        while not self.stop_requested:

            # push image & repeat as specified
            self.feed_image()

            # sleep
            if not self.script_config.keep_rtf:
                time.sleep(self.image_config.key_frame_period)

    def feed_image(self):

        # push image & repeat as specified
        for _ in range(self.image_config.key_frame_repeat):

            # fire image
            self.on_image_ready.notify({
                'image_path': str(self.last_image_path)
            })

            # sleep
            if self.script_config.keep_rtf:
                time.sleep(self.image_config.key_frame_period)

    def run_script(self, script: ImageSequenceScript, config: ImageInstructions, script_id):

        # prepare dir
        script_dir = f'{self.cache_dir}/script_{script_id}/'
        Path.mkdir(Path(script_dir), exist_ok=True, parents=True)

        # save config
        self.script_config = script

        # count loops
        loop_number = 1

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
                if i > 0 and seq.transition != TaskType.Image2Image:

                    # iterate prompts from previous sequence
                    reversed_prompts = self.reverse_prompt_weights(script.sequences[i - 1].prompts)
                    effective_seq.prompts.extend(reversed_prompts)

                # display sequence
                self.run_image_sequence(effective_seq, config, seq_id=seq_id, parent_dir=script_dir)

            # stop, unless loop requested
            print(f'Loop: {script.loop=}, {script.loop_count=}, {loop_number=}')
            if script.loop and loop_number < script.loop_count:

                # increment loop number
                loop_number += 1

                # generate new seed
                config.seed = self.image_gen.get_random_seed()

                # take last sequence prompts to loop back into first sequence smoothly
                continuity_prompts = self.reverse_prompt_weights(script.sequences[-1].prompts)

            else:
                break

        # stop feed thread
        self.stop()

        # wait and kill thread
        time.sleep(self.image_config.key_frame_period + 1)
        self.feed_thread = None

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

    # @classmethod
    # def calc_weight(cls, p, prog):
    #     weight_step = (p.end_weight - p.start_weight) / 100  # How much is one percent?
    #     return p.start_weight + (weight_step * prog)  # How progressed is this sequence?

    @classmethod
    def gen_seq_prompt(cls, prompts, iteration):

        def format_prompt(text, weight):
            return text if weight == 1.0 else f'({text}){round(weight, 3)}'

        # prepare prompts
        unique_prompts = set()
        effective_prompts = list()
        for p in prompts:

            # default weight
            weight = p.start_weight

            # check if there's a weight list
            if p.weight_list is not None:
                weight = p.weight_list[iteration]

            # drop prompts with low weight
            if p.text not in unique_prompts and weight > 0.02:
                unique_prompts.add(p.text)
                effective_prompts.append(format_prompt(p.text, weight))

        # build master prompt string
        return ', '.join(effective_prompts)

    @classmethod
    def get_model_attribute(cls, model: BaseModel, model_base: BaseModel, attribute):

        if model:
            attr_value = model.model_dump().get(attribute)
            if attr_value:
                return attr_value

        return model_base.model_dump().get(attribute)
