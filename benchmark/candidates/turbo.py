import torch
from diffusers import AutoPipelineForText2Image


def create_sdxl_turbo_pipe(device):

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

    # load to device
    pipe.to(device)

    return pipe


def create_sd_turbo_pipe(device):

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")

    # load to device
    pipe.to(device)

    return pipe
