import numpy as np
from PIL import Image

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, make_image_grid


if __name__ == '__main__':

    device = "mps"
    weight_type = torch.float16

    controlnet = ControlNetModel.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper-sketch", torch_dtype=weight_type
    ).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper", controlnet=controlnet, torch_dtype=weight_type
    )
    pipe.to(device)

    # image = load_image('/Users/himmelroman/Desktop/Bialik/me_canny.png')
    image = load_image('/Users/himmelroman/Desktop/Bialik/me.png')
    # image.show()
    # control_image = image.convert("RGB")
    # control_image = Image.fromarray(255 - np.array(control_image))

    output_images = list()
    for control_scale in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]:
        output_images.append(pipe(
            prompt='frida kahlo, self portrait',
            image=image,
            # width=512, height=512,
            width=768, height=768,
            # width=910, height=512,
            # width=1024, height=576,
            guidance_scale=0.8,
            num_inference_steps=1,
            num_images_per_prompt=1,
            output_type="pil",
            controlnet_conditioning_scale=control_scale
        ).images[0])

    # image.save("output.png")
    grid = make_image_grid(output_images, 1, len(output_images))
    grid.show()
