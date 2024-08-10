import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, TCDScheduler, LCMScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def create_sd15_pipe(device):

    base_model_id = "runwayml/stable-diffusion-v1-5"
    repo_name = "ByteDance/Hyper-SD"
    ckpt_name = "Hyper-SD15-1step-lora.safetensors"

    # Load model.
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()

    # Use TCD scheduler to achieve better image quality
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load to device
    pipe.to(device)

    return pipe


def create_sdxl_pipe_1step(device):

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    ckpt_name = "Hyper-SDXL-1step-lora.safetensors"

    # Load model.
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()

    # Use TCD scheduler to achieve better image quality
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    return pipe


def create_sdxl_pipe_2step(device):

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"

    # Take 2-steps lora as an example
    ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"

    # Load model.
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()

    # Ensure ddim scheduler timestep spacing set as trailing !!!
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    return pipe


# def create_sdxl_unet_pipe(device):
#
#     base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
#     repo_name = "ByteDance/Hyper-SD"
#     ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
#
#     # Load model.
#     unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
#     unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device=device))
#     pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
#
#     # Use LCM scheduler instead of ddim scheduler to support specific timestep number inputs
#     pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#
#     # Set start timesteps to 800 in the one-step inference to get better results
#     prompt = "a photo of a cat"
#     image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800]).images[0]
#     image.show()
