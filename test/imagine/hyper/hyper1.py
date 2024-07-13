import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, TCDScheduler
from huggingface_hub import hf_hub_download

controlnet_checkpoint = "lllyasviel/control_v11p_sd15_canny"

# Load original image
image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/input.png")
image = np.array(image)

# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16).to("mps")

# Load Hyper-SD15-1step lora
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))
pipe.fuse_lora()

# Use TCD scheduler to achieve better image quality
# pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# Lower eta results in more detail for multi-steps inference
eta = 1.0
image = pipe("a blue paradise bird in the jungle", num_inference_steps=1, image=control_image, guidance_scale=0, eta=eta).images[0]
image.show('image_out.png')
