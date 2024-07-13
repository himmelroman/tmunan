import time

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("IDKiro/sdxs-512-dreamshaper", torch_dtype=torch.float16)
pipeline.to("mps")

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
upscaler.to("mps")

prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
generator = torch.Generator(device="mps").manual_seed(234235)

# we stay in latent space! Let's make sure that Stable Diffusion returns the image
# in latent space
low_res_latents = pipeline(prompt,
                           generator=generator,
                           output_type="latent",
                           num_inference_steps=1,
                           guidance_scale=0,
                           height=512, width=512).images


for i in range(1):
    t_start = time.perf_counter()
    upscaled_image = upscaler(
        prompt=prompt,
        image=low_res_latents,
        num_inference_steps=10,
        guidance_scale=0,
        generator=generator,
    ).images[0]
    print(f"Total: {time.perf_counter() - t_start}")

# Let's save the upscaled image under "upscaled_astronaut.png"
upscaled_image.save("astronaut_1024.png")

# as a comparison: Let's also save the low-res image
with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

image.save("astronaut_512.png")

# import time
#
# import requests
# from PIL import Image
# from io import BytesIO
# from diffusers import StableDiffusionUpscalePipeline
# import torch
#
# # load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("mps")
#
# # let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))
#
# prompt = "a white cat"
#
# for i in range(10):
#     t_start = time.perf_counter()
#     upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=1).images[0]
#     print(f"Total: {time.perf_counter() - t_start}")
#     upscaled_image.save("upsampled_cat.png")
