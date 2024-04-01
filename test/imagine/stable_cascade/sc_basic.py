# import torch
# from diffusers import StableCascadeCombinedPipeline
#
# pipe = StableCascadeCombinedPipeline.from_pretrained("stabilityai/stable-cascade",
#                                                      variant="bf16",
#                                                      torch_dtype=torch.bfloat16)
#
# prompt = "an image of a shiba inu, donning a spacesuit and helmet"
# pipe(
#     prompt=prompt,
#     negative_prompt="",
#     num_inference_steps=10,
#     prior_num_inference_steps=20,
#     prior_guidance_scale=3.0,
#     width=1024,
#     height=1024,
# ).images[0].save("cascade-combined.png")

# import torch
# from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
#
# prompt = "an image of a shiba inu, donning a spacesuit and helmet"
# negative_prompt = ""
#
# prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
# decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)
#
# prior.enable_model_cpu_offload()
# prior_output = prior(
#     prompt=prompt,
#     height=1024,
#     width=1024,
#     negative_prompt=negative_prompt,
#     guidance_scale=4.0,
#     num_images_per_prompt=1,
#     num_inference_steps=20
# )
#
# decoder.enable_model_cpu_offload()
# decoder_output = decoder(
#     image_embeddings=prior_output.image_embeddings.to(torch.float16),
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     guidance_scale=0.0,
#     output_type="pil",
#     num_inference_steps=10
# ).images[0]
# decoder_output.save("cascade.png")

import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
)
from diffusers.models import StableCascadeUNet

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite")
decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder_lite")

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet)

# prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

# decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
