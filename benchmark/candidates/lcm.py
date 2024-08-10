import torch
from diffusers import LCMScheduler, DiffusionPipeline


def create_sd15_pipe(device):

    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # load to device
    pipe.to(device)

    return pipe


def create_sdxl_pipe(device):

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    return pipe
