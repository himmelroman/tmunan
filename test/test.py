import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image

from concurrent.futures import ThreadPoolExecutor


def gen_image(seed):

    # pass prompt and image to pipeline
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=4,
        guidance_scale=1,
        strength=0.6,
        generator=generator
    ).images[0]
    return image


if __name__ == '__main__':

    torch.mps.empty_cache()

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16
    ).to("mps")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    # prepare image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
    init_image = load_image(url)
    prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

    seeds = [1, 2, 3]
    with ThreadPoolExecutor() as executor:
        futures = executor.map(gen_image, seeds)
        results = [f.result() for f in futures]

    res = make_image_grid(results, rows=1, cols=len(results))
    res.show()
