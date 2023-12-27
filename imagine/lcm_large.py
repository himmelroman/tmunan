from typing import List

import torch
import random
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
from diffusers import LCMScheduler, UNet2DConditionModel, DiffusionPipeline
from diffusers.utils import make_image_grid, load_image


class LCMLarge:

    model_map = {
        'sdxl': {
            'model': "stabilityai/stable-diffusion-xl-base-1.0",
            'unet': "latent-consistency/lcm-sdxl"
        }
    }

    def __init__(self, model_id):

        # model key
        self.model_id = model_id

        # members
        self.unet = None
        self.pipe = None
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

        # load txt2img models
        self.unet = UNet2DConditionModel.from_pretrained(self.model_map[self.model_id]['unet'],
                                                         torch_dtype=torch.float16,
                                                         variant="fp16")
        self.pipe = DiffusionPipeline.from_pretrained(self.model_map[self.model_id]['model'],
                                                      unet=self.unet,
                                                      torch_dtype=torch.float16,
                                                      variant="fp16")

        # optimize
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        # init prompt generator
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        # LCM scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

    def txt2img(self,
                prompt_list: List[str],
                weight_list: List[float],
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 5,
                guidance_scale: float = 1.5,
                seed: int = 0,
                randomize_seed: bool = False
                ):

        if not self.pipe:
            raise Exception('Text to Image pipe not initialized!')

        # seed
        if randomize_seed:
            seed = self.get_random_seed()
        torch.manual_seed(seed)

        # generate prompt components
        prompt_dict = self.gen_prompt(prompt_list, weight_list, seed)

        # generate image
        result = self.pipe(**prompt_dict,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale,
                           height=height, width=width,
                           num_images_per_prompt=1).images
        return result

    def gen_prompt(self, prompts, weights, seed):

        # format
        prompts_string = ', '.join(f'"{p}"' for p in prompts)
        weights_string = ', '.join(str(w) for w in weights)

        # create prompt
        conditioning, pooled = self.compel(f'({prompts_string}).blend({weights_string})')
        generator = [torch.Generator().manual_seed(seed)]

        return dict(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator)

    @classmethod
    def get_random_seed(cls):
        return random.randint(0, np.iinfo(np.int32).max)


if __name__ == '__main__':

    lcm = LCMLarge(model_id='sdxl')
    lcm.load()

    images = lcm.txt2img(prompt_list='photo of beautiful old lady with golden hair, detailed face')
    grid = make_image_grid(images, rows=1, cols=len(images))
    grid.show()

    # grid.save()
    #new_image = lcm.img2img(image, 'Self-portrait oil painting, a beautiful cyborg with golden hair. Big scar on cheek and torn hair, 8k')
    #make_image_grid([image, new_image], rows=1, cols=2).show()

    # image.save("cache/img_base.png")
    # new_image.save("cache/img_new.png")

    # make_image_grid([image, new_image], rows=1, cols=2)
