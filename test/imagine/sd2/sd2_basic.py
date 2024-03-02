import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

start_time = time.time()
image = pipe(prompt=prompt,
             num_inference_steps=20,
             guidance_scale=0.9,
             height=768, width=768
             ).images[0]
elapsed_time = time.time() - start_time
print(f'Elapsed time: {elapsed_time}')

image.save("astronaut_rides_horse.png")
