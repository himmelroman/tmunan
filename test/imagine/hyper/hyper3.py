import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    AutoPipelineForImage2Image,
    TCDScheduler
)
from diffusers.utils import load_image, make_image_grid
from huggingface_hub import hf_hub_download

# controlnet = ControlNetModel.from_pretrained(
#     'lllyasviel/control_v11f1e_sd15_tile',
#     torch_dtype=torch.float16
# )

pipe = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to('mps')
# pipe.enable_xformers_memory_efficient_attention()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))
pipe.fuse_lora()

original = load_image(
    # 'https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png'
    '/Users/himmelroman/Desktop/bili.png'
)
# original = original.resize((512, 512))
# low_res = original.resize((64, 64))

prompt = f"winged lion"
# negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.manual_seed(5)
eta = 1.0
output_images = list()
for guidance_scale in [0.25, 0.5, 0.8, 1.0]:
    image = pipe(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        image=original,
        width=512,
        height=512,
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        eta=eta,
        strength=1.0,
        generator=generator,
    ).images[0]
    output_images.append(image)

grid = make_image_grid(output_images, 1, len(output_images))
grid.show()
