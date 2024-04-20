from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-8-lcm', torch_dtype=torch.float16) #, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")

prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=4, guidance_scale=2, generator=generator).images[0]
# image.save("./image.png")
image.show()
