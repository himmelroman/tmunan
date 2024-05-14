import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoPipelineForText2Image


def create_sdxs_pipe(device):

    repo = "IDKiro/sdxs-512-dreamshaper"
    weight_type = torch.float16  # or float32

    # Load model.
    pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)
    pipe.to(device)

    return pipe
