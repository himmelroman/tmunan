import torch
from diffusers import DDPMParallelScheduler
from diffusers import StableDiffusionParadigmsPipeline

scheduler = DDPMParallelScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", timestep_spacing="trailing")
pipe = StableDiffusionParadigmsPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", scheduler=scheduler, torch_dtype=torch.float16
)
pipe = pipe.to("mps")
ngpu, batch_per_device = torch.cuda.device_count(), 5
pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])
prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt, parallel=ngpu * batch_per_device, num_inference_steps=1000).images[0]
image.save("image.png")
