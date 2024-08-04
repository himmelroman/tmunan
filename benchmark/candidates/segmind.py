import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

model_id = "segmind/Segmind-Vega"
adapter_id = "segmind/Segmind-VegaRT"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("mps")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# disable guidance_scale by passing 0
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images[0]
# image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]
image.show()
