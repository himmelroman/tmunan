import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16).to("mps")
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=1,
    max_sequence_length=256,
    generator=torch.Generator("mps").manual_seed(0)
).images[0]
image.show()
