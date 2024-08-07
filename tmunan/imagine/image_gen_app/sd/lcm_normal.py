import os
import time
import random

import torch
import numpy as np
from diffusers import AutoPipelineForImage2Image, TCDScheduler
from huggingface_hub import hf_hub_download

from tmunan.common.log import get_logger
from tmunan.common.utils import load_image


class NormalLCM:
    """
    This is cool
    """

    model_map = {
        'lightning': {
            'model': "runwayml/stable-diffusion-v1-5",
            'lora': {
                "repo_id": "ByteDance/Hyper-SD",
                "filename": "Hyper-SD15-1step-lora.safetensors"
            }
        },
        'hyper-sd': {
            'model': "runwayml/stable-diffusion-v1-5",
            'lora': {
                "repo_id": "ByteDance/Hyper-SD",
                "filename": "Hyper-SD15-1step-lora.safetensors"
            }
        }
    }

    # constructor
    def __init__(self, model_id=None, cache_dir=None):

        # model sizes
        self.model_id = model_id

        # pipelines
        self.img2img_pipe = None

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

        # load model
        self.logger.info(f"Loading model: {self.model_map[self.model_id]['model']}")
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_map[self.model_id]['model'],
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # check for lora
        if self.model_map[self.model_id].get('lora'):

            # load and fuse sd_lcm lora
            self.logger.info(f"Loading Lora: {self.model_map[self.model_id]['lora']}")
            self.img2img_pipe.load_lora_weights(hf_hub_download(
                repo_id=self.model_map[self.model_id]['lora']["repo_id"],
                filename=self.model_map[self.model_id]['lora']["filename"]
            ))
            self.img2img_pipe.fuse_lora()

        # update scheduler
        self.img2img_pipe.scheduler = TCDScheduler.from_config(self.img2img_pipe.scheduler.config)

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

        if not self.img2img_pipe:
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

        self.logger.info(f"Generating img2img: {prompt=}, "
                         f"{num_inference_steps=}, {guidance_scale=}, "
                         f"{strength=}, {ip_adapter_weight=}, "
                         f"{seed=}")

        # generate image
        t_start_stream = time.perf_counter()
        result_image = self.img2img_pipe(
            prompt=prompt,
            image=base_image,
            num_inference_steps=1,
            num_images_per_prompt=1,
            width=width, height=height,
            guidance_scale=guidance_scale,
            strength=strength,
            # eta=0.5,
            output_type="pil",
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
