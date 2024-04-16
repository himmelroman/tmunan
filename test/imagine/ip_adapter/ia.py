import torch
from diffusers.utils import load_image
from diffusers import LCMScheduler, AutoPipelineForImage2Image

torch.mps.empty_cache()

# model_id = "sd-dreambooth-library/herge-style"
model_id = "SimianLuo/LCM_Dreamshaper_v7"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipeline = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16).to('mps')

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         # weight_name=["ip-adapter_sd15.bin", "ip-adapter_sd15.bin", "ip-adapter_sd15.bin"])
                         weight_name="ip-adapter_sd15.bin")
pipeline.load_lora_weights(lcm_lora_id)
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
# pipeline.enable_model_cpu_offload()

# pipeline.set_ip_adapter_scale([0.8, 0.8, 0.8])
pipeline.set_ip_adapter_scale(0.5)

prompt = "man"
generator = torch.Generator(device="mps").manual_seed(123)

ip_adapter_image1 = load_image("https://79d4-84-110-164-242.ngrok-free.app/rubin1.jpg")
ip_adapter_image2 = load_image("https://79d4-84-110-164-242.ngrok-free.app/rubin2.jpg")
ip_adapter_image3 = load_image("https://79d4-84-110-164-242.ngrok-free.app/rubin3.jpg")
input_image = load_image("https://79d4-84-110-164-242.ngrok-free.app/me.jpg")
image = pipeline(
    prompt=prompt,
    image=input_image,
    strength=0.2,
    # ip_adapter_image=[ip_adapter_image1, ip_adapter_image2, ip_adapter_image3],
    ip_adapter_image=ip_adapter_image1,
    num_inference_steps=4,
    guidance_scale=1,
    width=768, height=768
).images[0]

image.save('/tmp/out.png')
# image.show()
