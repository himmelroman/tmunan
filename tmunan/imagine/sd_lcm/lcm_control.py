import os
import time
import random

import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from streamdiffusion import StreamDiffusion

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
        },
        'hyper-sd': {
            'model': "runwayml/stable-diffusion-v1-5",
            'control_net': "lllyasviel/control_v11f1e_sd15_tile",
            'lora': {
                "repo_id": "ByteDance/Hyper-SD",
                "filename": "Hyper-SD15-1step-lora.safetensors"
            },
            'scheduler': TCDScheduler
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

        # check for lora
        if self.model_map[self.model_id].get('lora'):

            # load and fuse sd_lcm lora
            self.logger.info(f"Loading Lora: {self.model_map[self.model_id]['lora']}")
            self.control_net_pipe.load_lora_weights(hf_hub_download(
                repo_id=self.model_map[self.model_id]['lora']["repo_id"],
                filename=self.model_map[self.model_id]['lora']["filename"]
            ))
            self.control_net_pipe.fuse_lora()

        # update scheduler
        if self.model_map[self.model_id].get('scheduler'):
            scheduler_class = self.model_map[self.model_id].get('scheduler')
            self.control_net_pipe.scheduler = scheduler_class.from_config(self.control_net_pipe.scheduler.config)

        # accelerate
        self.control_net_pipe.enable_xformers_memory_efficient_attention()

        # compile with pytorch
        # self.control_net_pipe.unet = torch.compile(
        #     self.control_net_pipe.unet, mode="reduce-overhead", fullgraph=True
        # )
        # self.control_net_pipe.vae = torch.compile(
        #     self.control_net_pipe.vae, mode="reduce-overhead", fullgraph=True
        # )

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
            controlnet_conditioning_scale=control_net_scale,
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
