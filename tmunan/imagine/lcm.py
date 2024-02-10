import random
import numpy as np

import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from compel import Compel, ReturnedEmbeddingsType


class LCM:

    model_map = {
        'small': {
            'model': "SimianLuo/LCM_Dreamshaper_v7",
            'adapter': "latent-consistency/lcm-lora-sdv1-5"
        },
        'medium': {
            'model': "segmind/SSD-1B",
            'adapter': "latent-consistency/lcm-lora-ssd-1b"
        },
        'large': {
            'model': "stabilityai/stable-diffusion-xl-base-1.0",
            # 'model': "stabilityai/sdxl-turbo",
            'adapter': "latent-consistency/lcm-lora-sdxl"
        }
    }

    def __init__(self, txt2img_size=None, img2img_size=None):

        # model sizes
        self.txt2img_size = txt2img_size
        self.img2img_size = img2img_size

        # pipelines
        self.txt2img_pipe = None
        self.img2img_pipe = None

        # prompt
        self.compel = None

        # comp device
        self.device = self.get_device()

    @classmethod
    def get_device(cls):

        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load(self):

        # text to image
        if self.txt2img_size:
            self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_map[self.txt2img_size]['model'],
                local_files_only=True,
                torch_dtype=torch.float16).to(self.device)
            self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)

            # load and fuse lcm lora
            self.txt2img_pipe.load_lora_weights(self.model_map[self.txt2img_size]['adapter'],
                                                weight_name='pytorch_lora_weights.safetensors')
            self.txt2img_pipe.fuse_lora()

            # init prompt generator
            if self.txt2img_size == 'large':
                self.compel = Compel(
                    tokenizer=[self.txt2img_pipe.tokenizer, self.txt2img_pipe.tokenizer_2],
                    text_encoder=[self.txt2img_pipe.text_encoder, self.txt2img_pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )

        # image to image
        if self.img2img_size:
            self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_map[self.img2img_size]['model'],
                torch_dtype=torch.float16).to(self.device)
            self.img2img_pipe.scheduler = LCMScheduler.from_config(self.img2img_pipe.scheduler.config)

            # load LCM-LoRA
            self.img2img_pipe.load_lora_weights(self.model_map[self.img2img_size]['adapter'])
            self.img2img_pipe.fuse_lora()

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
        prompt_dict = self.gen_prompt(prompt, seed)

        # run image generation
        result = self.txt2img_pipe(**prompt_dict,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale,
                                   height=height, width=width
                                   ).images
        return result

    def img2img(self,
                image_url: str,
                prompt: str,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 4,
                guidance_scale: float = 1.0,
                strength: float = 0.6
                ):

        if not self.img2img_pipe:
            raise Exception('Text to Image pipe not initialized!')

        # load image
        prompt_image = load_image(image_url)

        # pass prompt and image to pipeline
        generator = torch.manual_seed(0)
        result = self.img2img_pipe(prompt=prompt,
                                   image=prompt_image,
                                   num_inference_steps=num_inference_steps,
                                   height=width, width=height,
                                   guidance_scale=guidance_scale,
                                   strength=strength,
                                   generator=generator
                                   ).images
        return result

    def gen_prompt(self, prompt, seed):

        # check if we're using compel (needed for SDXL model)
        if self.compel:

            # create compel prompt components
            conditioning, pooled = self.compel(prompt)
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

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)
