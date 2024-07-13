import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    TCDScheduler
)
from diffusers.utils import load_image, make_image_grid
from huggingface_hub import hf_hub_download

controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/control_v11f1e_sd15_tile',
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to('mps')
# pipe.enable_xformers_memory_efficient_attention()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))

original = load_image(
    # 'https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png'
    '/Users/himmelroman/Desktop/bili.png'
)

original = original.resize((512, 512))
low_res = original.resize((64, 64))

prompt = f"winged lion"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.manual_seed(2)

output_images = list()
for cnet_scale in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for eta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original,
            control_image=low_res,
            width=512,
            height=512,
            num_inference_steps=1,
            guidance_scale=1.0,
            controlnet_conditioning_scale=cnet_scale,
            eta=eta,
            strength=1.0,
            generator=generator
        ).images[0]
        output_images.append(image)

grid = make_image_grid(output_images, 2, len(output_images))
grid.show()
