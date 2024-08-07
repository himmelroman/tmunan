import os
import time
import random

import torch
import numpy as np
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, AutoencoderTiny

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from tmunan.utils.log import get_logger
from tmunan.utils.image import load_image


class StreamLCM:
    """
    This is cool
    But also look at :obj:`~LCM`
    """

    model_map = {
        'lcm1.5': {
            'model': "SimianLuo/LCM_Dreamshaper_v7",
            'lcm_lora': "latent-consistency/lcm-lora-sdv1-5"
        },
        'sd-turbo': {
            'model': "stabilityai/sd-turbo"
        },
        'ssd-1b': {
            'model': "segmind/SSD-1B",
            'lcm_lora': "latent-consistency/lcm-lora-ssd-1b"
        },
        'sdxl': {
            'model': "stabilityai/stable-diffusion-xl-base-1.0",
            'lcm_lora': "latent-consistency/lcm-lora-sdxl"
        }
    }

    # constructor
    def __init__(self, model_id=None, cache_dir=None):

        # model sizes
        self.model_id = model_id

        # pipelines
        self.stream = None
        self.stream_cache = {
            'prompt': None,
            'guidance_scale': None,
            'strength': None
        }
        self.img2img_pipe = None

        # prompt
        self.img2img_compel = None

        # comp device
        self.device = self.get_device()

        # env
        self.logger = get_logger(self.__class__.__name__)
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")

    @classmethod
    def get_device(cls):

        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load(self):

        self.logger.info(f"Loading models onto device: {self.device}")

        # text to image
        self.logger.info(f"Loading img2img model: {self.model_map[self.model_id]['model']}")
        self.img2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_map[self.model_id]['model'],
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # StreamDiffusion
        self.stream = StreamDiffusion(
            self.img2img_pipe,
            t_index_list=[32, 45],
            torch_dtype=torch.float16,
            width=904,
            height=512,
            # cfg_type='self'
        )
        # self.stream.enable_similar_image_filter(threshold=0.99, max_skip_frame=3)

        # check for LCM lora
        if self.model_map[self.model_id].get('lcm_lora'):

            # load and fuse sd_lcm lora
            self.logger.info(f"Loading LCM Lora: {self.model_map[self.model_id]['lcm_lora']}")
            self.stream.load_lcm_lora(self.model_map[self.model_id]['lcm_lora'])
            self.stream.fuse_lora()

        # Use Tiny VAE for further acceleration
        self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=self.img2img_pipe.device,
            dtype=self.img2img_pipe.dtype
        )

        # accelerate with tensor-rt
        if self.device == 'cuda':

            self.logger.info(f"Accelerating with TensorRT! {self.cache_dir=}")
            from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
            self.stream = accelerate_with_tensorrt(
                stream=self.stream,
                engine_dir=f'{self.cache_dir}/tensorrt',
                max_batch_size=2,
                engine_build_options={
                    'opt_image_height': 512,
                    'opt_image_width': 904,

                    # 'build_dynamic_shape': True
                }
            )

        self.logger.info("Loading models finished.")

    def update_stream_preparation(self, prompt, guidance_scale, strength, seed):

        # determine if "prepare" is needed
        if (self.stream_cache['prompt'] != prompt or
                self.stream_cache['guidance_scale'] != guidance_scale or
                self.stream_cache['strength'] != strength or
                self.stream_cache['seed'] != seed):

            # prepare
            self.stream.prepare(
                prompt=prompt,
                guidance_scale=float(guidance_scale),
                strength=float(strength),
                seed=int(seed)
            )
            self.logger.info(
                f"Stream configuration: "
                f"{prompt=}, "
                f"{guidance_scale=}, "
                f"{strength=}, "
                f"{seed=}"
            )

            # update cache
            self.stream_cache['prompt'] = prompt
            self.stream_cache['guidance_scale'] = guidance_scale
            self.stream_cache['strength'] = strength
            self.stream_cache['seed'] = seed

    def img2img(self,
                prompt: str,
                image: str | Image,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 4,
                guidance_scale: float = 1.0,
                strength: float = 0.6,
                control_net_scale: float = 1.0,
                ip_adapter_weight: float = 0.6,
                seed: int = 0,
                randomize_seed: bool = False,
                ):

        if not self.img2img_pipe:
            raise Exception('Image to Image pipe not initialized!')

        # load image
        if type(image) is str:
            base_image = load_image(image)
        else:
            base_image = image

        # seed
        if seed == 0 or randomize_seed:
            seed = self.get_random_seed()

        # update preparation
        # if none of the arguments changed since last time, this will do nothing (fast)
        self.update_stream_preparation(
            prompt=prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed
        )

        # convert and resize
        # base_image = base_image.convert("RGB").resize((width, height))

        # pre-process image
        self.logger.info(f"Preprocessing Image: {height=}, {width=}")
        t_start_pre = time.perf_counter()
        input_latent = self.stream.image_processor.preprocess(base_image, height, width).to(
            device=self.device,
            dtype=self.img2img_pipe.dtype
        )

        # generate image
        t_start_stream = time.perf_counter()
        output_latent = self.stream(input_latent)

        # post-process image
        t_start_post = time.perf_counter()
        result_images = self.post_process_image(output_latent)

        # log times
        self.logger.info(
            f"Total: {time.perf_counter() - t_start_pre}, "
            f"Pre: {t_start_stream - t_start_pre}, "
            f"Stream: {t_start_post - t_start_stream}, "
            f"Post: {time.perf_counter() - t_start_post}"
        )
        return result_images

    def pre_process_image(self, image):
        image_latent = self.stream.preprocess_image(image)
        return image_latent

    def post_process_image(self, image_latent):
        images = postprocess_image(image_latent, output_type="pil")
        return images

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)
