import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter

from diffusers.utils import make_image_grid, load_image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, AutoencoderTiny

from tmunan.imagine.sd_lcm.lcm import LCM

device = "mps"
weight_type = torch.float16

# controlnet = ControlNetModel.from_pretrained(
#     "IDKiro/sdxs-512-dreamshaper-sketch", torch_dtype=weight_type
# ).to(device)
# vae_tiny = AutoencoderTiny.from_pretrained(
#     "IDKiro/sdxs-512-dreamshaper", subfolder="vae"
# )
# vae_tiny.to(device, dtype=weight_type)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "IDKiro/sdxs-512-dreamshaper", controlnet=controlnet, torch_dtype=weight_type
# )
# pipe.vae = vae_tiny
# pipe.to(device)

lcm = LCM(model_size='small')
lcm.load()

source_image = load_image('/tmp/reimagine/source_0.png')
# gs_image = source_image.convert("L")

for strength in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

    small_image = source_image.resize((512, 512))

    res = lcm.img2img(
            prompt='oil painting',
            image=small_image,
            width=512,
            height=512,
            guidance_scale=0.5,
            num_inference_steps=4,
            strength=strength
        )

    res[0].save(f'/tmp/reimagine/strength_{strength}.png')
