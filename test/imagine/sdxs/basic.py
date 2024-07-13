import time

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoPipelineForText2Image, AutoPipelineForImage2Image

from tmunan.common.utils import load_image


def gen_image(base_image):
    start_time = time.time()
    image = pipe(
        prompt="a picture of an old woman's face standing in the rain",
        image=base_image,
        num_inference_steps=1,
        guidance_scale=0,
        strength=0.8,
        height=512, width=512,
        generator=torch.Generator(device="mps").manual_seed(234235)
    ).images[0]

    elapsed_time = time.time() - start_time
    print(f"Done generating txt2img: {elapsed_time=}")

    return image


if __name__ == '__main__':
    repo = "IDKiro/sdxs-512-dreamshaper"
    # repo = "IDKiro/sdxs-512-0.9"
    weight_type = torch.float16     # or float32

    # Load model.
    # pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=weight_type)
    pipe = AutoPipelineForImage2Image.from_pretrained(repo, torch_dtype=weight_type)
    pipe.to("mps")

    base_image = load_image('/Users/himmelroman/Desktop/Bialik/me.png')

    # Ensure using 1 inference step and CFG set to 0.
    for _ in range(5):
        image = gen_image(base_image)

    # image.save("output.png")
    image.show()
