import torch

from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

base_model = "IDKiro/sdxs-512-0.9"
taesd_model = "madebyollin/taesd"


pipe = AutoPipelineForImage2Image.from_pretrained(
    base_model,
    safety_checker=None,
)

tiny_vae = True
if tiny_vae:
    pipe.vae = AutoencoderTiny.from_pretrained(
        taesd_model, torch_dtype=torch.float16, use_safetensors=True
    ).to("mps")

pipe.to(device="mps", dtype=torch.float16)

generator = torch.manual_seed(1234)
prompt_embeds = None
prompt = "winged lion"

image = load_image('/Users/himmelroman/Desktop/bili.png')
image = image.resize((910, 512))

results = pipe(
    image=image,
    prompt=prompt,
    prompt_embeds=prompt_embeds,
    generator=generator,
    strength=1.0,
    num_inference_steps=1,
    guidance_scale=0.1,
    width=910,
    height=512,
    output_type="pil",
)
results.images[0].show()
