import os
import time
import random

import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from huggingface_hub import hf_hub_download

from tmunan.common.log import get_logger
from tmunan.common.utils import load_image


class ControlLCM:
    """
    This is cool
    But also look at :obj:`~LCM`
    """

    model_map = {
        'xs': {
            'model': "IDKiro/sdxs-512-dreamshaper",
            'control_net': "IDKiro/sdxs-512-dreamshaper-sketch"
        }
    }

    # constructor
    def __init__(self, model_id=None, cache_dir=None):

        # model sizes
        self.model_id = model_id

        # pipelines
        self.control_net_model = None
        self.control_net_pipe = None

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

        # load control net model
        self.logger.info(f"Loading ControlNet model: {self.model_map[self.model_id]['control_net']}")
        self.control_net_model = ControlNetModel.from_pretrained(
            self.model_map[self.model_id]['control_net'],
            torch_dtype=torch.float16
        ).to(self.device)

        # load model
        self.logger.info(f"Loading model: {self.model_map[self.model_id]['model']}")
        self.control_net_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_map[self.model_id]['model'],
            controlnet=self.control_net_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # check for LCM lora
        if self.model_map[self.model_id].get('lcm_lora'):

            # load and fuse sd_lcm lora
            self.logger.info(f"Loading LCM Lora: {self.model_map[self.model_id]['lcm_lora']}")
            # self.stream.load_lcm_lora(self.model_map[self.model_size]['lcm_lora'])
            self.control_net_pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))

        # # accelerate with tensor-rt
        # if self.device == 'cuda':
        #
        #     self.logger.info(f"Accelerating with TensorRT! {self.cache_dir=}")
        #     from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
        #     self.stream = accelerate_with_tensorrt(
        #         stream=self.stream,
        #         engine_dir=f'{self.cache_dir}/tensorrt',
        #         max_batch_size=2,
        #         engine_build_options={
        #             'opt_image_height': 512,
        #             'opt_image_width': 904,
        #
        #             # 'build_dynamic_shape': True
        #         }
        #     )

        self.logger.info("Loading models finished.")

    def img2img(self,
                prompt: str,
                image: str,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 4,
                guidance_scale: float = 1.0,
                strength: float = 0.6,
                control_net_scale: float = 1.0,
                ip_adapter_weight: float = 0.6,
                seed: int = 0,
                randomize_seed: bool = False
                ):

        if not self.control_net_pipe:
            raise Exception('Image to Image pipe not initialized!')

        # seed
        if seed == 0 or randomize_seed:
            seed = self.get_random_seed()

        # load image
        if type(image) is str:
            base_image = load_image(image)
            self.logger.info(f"Loaded image from: {image}")
        else:
            base_image = image
            self.logger.info(f"Image instance provided.")

        # convert and resize
        # base_image = base_image.convert("RGB").resize((width, height))

        # generate image
        t_start_stream = time.perf_counter()
        result_image = self.control_net_pipe(
            prompt=prompt,
            image=base_image,
            width=width, height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=1,
            num_images_per_prompt=1,
            output_type="pil",
            controlnet_conditioning_scale=control_net_scale,
            seed=seed
        ).images

        # log times
        self.logger.info(
            f"Total: {time.perf_counter() - t_start_stream}"
        )
        return result_image

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)
