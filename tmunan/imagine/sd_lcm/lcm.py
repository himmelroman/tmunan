import random
import time
from pathlib import Path

import torch
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
# from latentblending import BlendingEngine

from tmunan.common.log import get_logger
from tmunan.common.utils import load_image


class LCM:

    model_map = {
        'sdxs': {
            'model': "IDKiro/sdxs-512-dreamshaper",
        },
        'small': {
            'model': "SimianLuo/LCM_Dreamshaper_v7",
            # 'model': "lykon/dreamshaper-8-lcm",
            'lcm_lora': "latent-consistency/lcm-lora-sdv1-5",
            'ip_adapter': "ip-adapter_sd15.bin",
            'subfolder': "models"
        },
        'medium': {
            'model': "segmind/SSD-1B",
            'lcm_lora': "latent-consistency/lcm-lora-ssd-1b",
            'ip_adapter': "ip-adapter_sdxl.bin",
            'subfolder': "models"
        },
        'large': {
            'model': "stabilityai/stable-diffusion-xl-base-1.0",
            'lcm_lora': "latent-consistency/lcm-lora-sdxl",
            'ip_adapter': "ip-adapter_sdxl.bin",
            'subfolder': "sdxl_models"
        }
    }

    # constructor
    def __init__(self, model_size=None, ip_adapter_folder=None):

        # model sizes
        self.model_size = model_size

        # ip adapter
        self.ip_adapter_folder = ip_adapter_folder
        self.ip_adapter_images = None

        # pipelines
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.blend_engine = None

        # prompt
        self.txt2img_compel = None
        self.img2img_compel = None

        # comp device
        self.device = self.get_device()

        # env
        self.logger = get_logger(self.__class__.__name__)

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
        self.logger.info(f"Loading txt2img model: {self.model_map[self.model_size]['model']}")
        self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_map[self.model_size]['model'],
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # check for LCM lora
        if self.model_map[self.model_size].get('lcm_lora'):

            # update scheduler
            self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)

            # load and fuse sd_lcm lora
            self.logger.info(f"Loading LCM Lora: {self.model_map[self.model_size]['lcm_lora']}")
            self.txt2img_pipe.load_lora_weights(self.model_map[self.model_size]['lcm_lora'],
                                                weight_name='pytorch_lora_weights.safetensors')
            self.txt2img_pipe.fuse_lora()

        # load img2img model
        self.logger.info(f"Loading img2img model: {self.model_map[self.model_size]['model']}")
        self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.txt2img_pipe).to(self.device)
        # self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
        #     self.model_map[self.img2img_size]['model'],
        #     torch_dtype=torch.float16).to(self.device)

        # check if ip-adapter source folder was provided
        if self.ip_adapter_folder and Path(self.ip_adapter_folder).exists():

            # load style images
            self.ip_adapter_images = self.load_ip_adapter_images()
            self.logger.info(f"Found {len(self.ip_adapter_images)} style images...")

            # load adapter
            if self.ip_adapter_images:
                self.logger.info(f"Loading IPAdapter model: {self.model_map[self.model_size]['ip_adapter']}")
                self.img2img_pipe.load_ip_adapter("h94/IP-Adapter",
                                                  subfolder=self.model_map[self.model_size]['subfolder'],
                                                  weight_name=[self.model_map[self.model_size]['ip_adapter']] * len(self.ip_adapter_images))
            else:
                self.logger.info(f"Not loading IPAdapter!")

        # init prompt generator
        if self.model_size == 'large':
            self.txt2img_compel = Compel(
                tokenizer=[self.txt2img_pipe.tokenizer, self.txt2img_pipe.tokenizer_2],
                text_encoder=[self.txt2img_pipe.text_encoder, self.txt2img_pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            self.img2img_compel = Compel(
                tokenizer=[self.img2img_pipe.tokenizer, self.img2img_pipe.tokenizer_2],
                text_encoder=[self.img2img_pipe.text_encoder, self.img2img_pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )

        self.logger.info("Loading models finished.")

    def load_ip_adapter_images(self):

        if Path(self.ip_adapter_folder).exists():

            # iterate style folder images
            style_images = list()
            for child_path in Path(self.ip_adapter_folder).iterdir():
                if child_path.is_file():
                    img = load_image(str(child_path))
                    style_images.append(img)

            return style_images

    def txt2img(self,
                prompt: str,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 4,
                guidance_scale: float = 0.0,
                seed: int = 0,
                randomize_seed: bool = False
                ):

        if not self.txt2img_pipe:
            raise Exception('Text to Image pipe not initialized!')

        # seed
        if randomize_seed:
            seed = self.get_random_seed()
        torch.manual_seed(seed)

        # gen prompt
        prompt_dict = self.gen_prompt(prompt, seed, self.txt2img_compel)

        # run image generation
        self.logger.info(f"Generating txt2img: {prompt=}, {seed=}")
        start_time = time.time()
        result = self.txt2img_pipe(**prompt_dict,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale,
                                   height=height, width=width,
                                   num_images_per_prompt=1
                                   ).images
        elapsed_time = time.time() - start_time
        self.logger.info(f"Done generating txt2img: {elapsed_time=}")

        return result

    def txt2img_latents(self,
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        latents,
                        height: int = 512,
                        width: int = 512,
                        num_inference_steps: int = 4,
                        guidance_scale: float = 0.0,
                        seed: int = 0,
                        randomize_seed: bool = False,
                        ):

        if not self.txt2img_pipe:
            raise Exception('Text to Image pipe not initialized!')

        # seed
        if randomize_seed:
            seed = self.get_random_seed()
        torch.manual_seed(seed)

        # run image generation
        start_time = time.time()
        result = self.txt2img_pipe(prompt_embeds=prompt_embeds,
                                   negative_prompt_embeds=negative_prompt_embeds,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                   latents=latents,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale,
                                   height=height, width=width,
                                   ).images
        elapsed_time = time.time() - start_time
        self.logger.info(f"Done generating txt2img: {elapsed_time=}")

        return result

    def img2img(self,
                prompt: str,
                image_url: str,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 4,
                guidance_scale: float = 1.0,
                strength: float = 0.6,
                ip_adapter_weight: float = 0.6,
                seed: int = 0,
                randomize_seed: bool = False,
                ):

        if not self.img2img_pipe:
            raise Exception('Text to Image pipe not initialized!')

        # load image
        self.logger.info(f"Loading image from: {image_url}")
        base_image = load_image(image_url)
        self.logger.info(f"Image loaded! {base_image}")

        # seed
        if randomize_seed:
            seed = self.get_random_seed()
        torch.manual_seed(seed)

        # gen prompt
        prompt_dict = self.gen_prompt(prompt, seed, self.img2img_compel)

        # prepare ip-adapter params
        ip_adapter_params = dict()
        if self.ip_adapter_images:
            ip_adapter_params['ip_adapter_image'] = self.ip_adapter_images
            self.img2img_pipe.set_ip_adapter_scale([ip_adapter_weight / len(self.ip_adapter_images)] * len(self.ip_adapter_images))

        # pass prompt and image to pipeline
        self.logger.info(f"Generating img2img: {image_url=}\n{prompt=}\n"
                         f"{num_inference_steps=}, {guidance_scale=}, "
                         f"{strength=}, {ip_adapter_weight=}, "
                         f"{seed=}")
        start_time = time.time()
        result = self.img2img_pipe(**prompt_dict,
                                   **ip_adapter_params,
                                   image=base_image,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale,
                                   strength=strength,
                                   height=width, width=height,
                                   num_images_per_prompt=1
                                   ).images
        elapsed_time = time.time() - start_time
        self.logger.info(f"Done generating img2img: {elapsed_time=}")

        return result

    def gen_prompt(self, prompt, seed, compel):

        # check if we're using compel (needed for SDXL model)
        if compel:

            # create compel prompt components
            conditioning, pooled = compel(prompt)
            seed_generator = torch.Generator().manual_seed(seed)

            return {
                'prompt_embeds': conditioning,
                'pooled_prompt_embeds': pooled,
                'generator': seed_generator
            }

        else:

            return {
                'prompt': prompt,
                'seed': seed
            }

    def get_prompt_embeds(self, prompt_text):

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.txt2img_pipe.encode_prompt(
            prompt=prompt_text,
            prompt_2=prompt_text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
            negative_prompt_2="",
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)
