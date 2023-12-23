import torch
import random
import numpy as np
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image


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

    def load(self, torch_device: str):

        # text to image
        if self.txt2img_size:
            self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_map[self.txt2img_size]['model'],
                torch_dtype=torch.float16).to(torch_device)
            self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)

            # load and fuse lcm lora
            self.txt2img_pipe.load_lora_weights(self.model_map[self.txt2img_size]['adapter'])
            self.txt2img_pipe.fuse_lora()

        # image to image
        if self.img2img_size:
            self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_map[self.img2img_size]['model'],
                torch_dtype=torch.float16).to(torch_device)
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

        result = self.txt2img_pipe(prompt=prompt,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale,
                                   height=height, width=width,
                                   num_images_per_prompt=1,
                                   # output_type="pil"
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
        init_image = load_image(image_url)

        # pass prompt and image to pipeline
        generator = torch.manual_seed(0)
        result = self.img2img_pipe(prompt=prompt,
                                   image=init_image,
                                   num_inference_steps=num_inference_steps,
                                   height=width, width=height,
                                   guidance_scale=guidance_scale,
                                   strength=strength,
                                   generator=generator
                                   ).images
        return result

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)


if __name__ == '__main__':

    lcm = LCM()
    lcm.load(torch_device="mps", model_size='medium')

    images = lcm.txt2img(prompt='photo of beautiful old lady with golden hair, detailed face')
    grid = make_image_grid(images, rows=1, cols=len(images))
    grid.show()
    # grid.save()
    #new_image = lcm.img2img(image, 'Self-portrait oil painting, a beautiful cyborg with golden hair. Big scar on cheek and torn hair, 8k')
    #make_image_grid([image, new_image], rows=1, cols=2).show()

    # image.save("cache/img_base.png")
    # new_image.save("cache/img_new.png")

    # make_image_grid([image, new_image], rows=1, cols=2)
