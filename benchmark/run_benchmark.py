import numpy as np
from time import perf_counter

import torch

import benchmark.candidates.hyper_sd as hyper_sd
import benchmark.candidates.turbo as turbo
import benchmark.candidates.sdxs as sdxs
import benchmark.candidates.latent_consistency as latent_consistency


def run_inference(pipe, **pipe_args):

    perf_list = []
    image_list = []
    for _ in range(5):

        t_start = perf_counter()
        images = pipe(**pipe_args).images
        t_elapsed = perf_counter() - t_start

        image_list.append(images[0])
        perf_list.append(t_elapsed)

    return image_list, perf_list


def benchmark_pipe(pipe, **pipe_args):

    # run benchmark
    images, perf = run_inference(pipe, **pipe_args)

    # clear memory
    del pipe
    pipe = None
    torch.mps.empty_cache()

    return images, perf


def benchmark_sdxs(device, prompt, height, width):

    # create pipe
    pipe = sdxs.create_sdxs_pipe(device)

    # run benchmark
    images, perf = benchmark_pipe(pipe,
                                  prompt=prompt,
                                  num_inference_steps=1,
                                  guidance_scale=0.5,
                                  height=height,
                                  width=width)

    return perf


def benchmark_hyper_sd_sd15(device, prompt, height, width):

    # create pipe
    pipe = hyper_sd.create_sd15_pipe(device)

    # special args
    eta = 1.0

    # run benchmark
    images, perf = benchmark_pipe(pipe,
                                  prompt=prompt,
                                  num_inference_steps=1,
                                  guidance_scale=0.5,
                                  height=height,
                                  width=width,
                                  eta=eta)

    return perf


def benchmark_hyper_sd_sdxl_1step(device, prompt, height, width):

    # create pipe
    pipe = hyper_sd.create_sdxl_pipe_1step(device)

    # run benchmark
    images, perf = benchmark_pipe(pipe,
                                  prompt=prompt,
                                  num_inference_steps=1,
                                  guidance_scale=0.0,
                                  height=height,
                                  width=width)

    return perf


def benchmark_hyper_sd_sdxl_2step(device, prompt, height, width):

    # create pipe
    pipe = hyper_sd.create_sdxl_pipe_2step(device)

    # run benchmark
    images, perf = benchmark_pipe(pipe,
                                  prompt=prompt,
                                  num_inference_steps=2,
                                  guidance_scale=0.0,
                                  height=height,
                                  width=width)

    return perf


def benchmark_hyper_sd_sdxl_unet(device, prompt, height, width):

    # create pipe
    pipe = hyper_sd.create_sdxl_unet_pipe(device)

    # special args
    timesteps = [800]

    # run benchmark
    images, perf = benchmark_pipe(pipe,
                                  prompt=prompt,
                                  num_inference_steps=1,
                                  guidance_scale=0.5,
                                  height=height,
                                  width=width,
                                  timesteps=timesteps)

    return perf


def benchmark_latent_consistency_sd15(device, prompt, height, width):

    # create pipe
    pipe = latent_consistency.create_sd15_pipe(device)

    # run benchmark
    images, perf = run_inference(pipe,
                                 prompt=prompt,
                                 num_inference_steps=4,
                                 guidance_scale=0.5,
                                 height=height,
                                 width=width)

    return perf


def benchmark_latent_consistency_sdxl(device, prompt, height, width):

    # create pipe
    pipe = latent_consistency.create_sdxl_pipe(device)

    # run benchmark
    images, perf = run_inference(pipe,
                                 prompt=prompt,
                                 num_inference_steps=4,
                                 guidance_scale=0.5,
                                 height=height,
                                 width=width)

    return perf


def benchmark_sd_turbo(device, prompt, height, width):

    # create pipe
    pipe = turbo.create_sd_turbo_pipe(device)

    # run benchmark
    images, perf = run_inference(pipe,
                                 prompt=prompt,
                                 num_inference_steps=1,
                                 guidance_scale=0.0,
                                 height=height,
                                 width=width)

    return perf


def benchmark_sdxl_turbo(device, prompt, height, width):

    # create pipe
    pipe = turbo.create_sdxl_turbo_pipe(device)

    # run benchmark
    images, perf = run_inference(pipe,
                                 prompt=prompt,
                                 num_inference_steps=1,
                                 guidance_scale=0.5,
                                 height=height,
                                 width=width)

    return perf


if __name__ == '__main__':

    # determine device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    # params
    prompt = 'cat'
    perf_results = dict()

    # SD 1.5 models
    height = 512
    width = 512
    perf_results['sdxs'] = benchmark_sdxs(device, prompt, height, width)
    perf_results['sd_turbo'] = benchmark_sdxl_turbo(device, prompt, height, width)
    perf_results['latent_consistency_sd15'] = benchmark_latent_consistency_sd15(device, prompt, height, width)
    perf_results['hyper_sd_sd15'] = benchmark_hyper_sd_sd15(device, prompt, height, width)

    # SDXL models
    height = 512
    width = 512
    perf_results['sdxl_turbo'] = benchmark_sdxl_turbo(device, prompt, height, width)
    perf_results['latent_consistency_sdxl'] = benchmark_latent_consistency_sdxl(device, prompt, height, width)
    perf_results['hyper_sd_sdxl_1step'] = benchmark_hyper_sd_sdxl_1step(device, prompt, height, width)
    perf_results['hyper_sd_sdxl_2step'] = benchmark_hyper_sd_sdxl_2step(device, prompt, height, width)
    # DOESN'T WORK: perf_results['hyper_sd_sdxl_unet'] = benchmark_hyper_sd_sdxl_unet(device, prompt, height, width)

    for p in perf_results:
        print(f'Mean for {p}: {np.mean(perf_results[p][1:])}')
