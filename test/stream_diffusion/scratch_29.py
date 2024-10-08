from time import perf_counter

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def run_case(pipe, t_index_list, prompt, init_image):

    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        # t_index_list=[32, 45],
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
    )

    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # Enable acceleration
    # pipe.enable_xformers_memory_efficient_attention()

    for strength in [1.0]:

        # Prepare the stream
        t_start = perf_counter()
        stream.prepare(
            prompt=prompt,
            guidance_scale=1.0,
            strength=strength,
            seed=int(12345)
        )
        print(f'Prepare time: {perf_counter() - t_start}')

        print('Warming up')
        for _ in range(2):
            t_start = perf_counter()
            stream(init_image)
            print(f'Warmup time: {perf_counter() - t_start}')

        t_start = perf_counter()
        x_output = stream(init_image)
        print(f'Diffusion time: {perf_counter() - t_start}')

        t_start = perf_counter()
        image = postprocess_image(x_output, output_type="pil")[0]
        print(f'Postprocess time: {perf_counter() - t_start}')

        t_index_list_str = '_'.join([str(i) for i in t_index_list])
        image.save(f'/Users/himmelroman/Desktop/inter/{t_index_list_str}.png')


def run_test():

    # You can load any models using diffuser's StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(
    pipe = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(
    # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
        # device=torch.device("mps"),
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("mps")

    # If the loaded model is not LCM, merge LCM
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))
    pipe.fuse_lora()

    # prepare input
    prompt = "old photograph, b&w, black and white, 1890, window, lush greenery"
    init_image = load_image("/Users/himmelroman/Desktop/House/373408370_6478098395560526_1428679915222833350_n.jpg").resize((512, 904))

    run_case(pipe, [32, 45], prompt, init_image)
    for ti in range(30, 48, 2):
        run_case(pipe, [ti], prompt, init_image)


def streamdiffusion_test():

    # You can load any models using diffuser's StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    pipe = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(
        device=torch.device("mps"),
        dtype=torch.float16,
    )

    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=[32, 45],
        torch_dtype=torch.float16,
    )

    # If the loaded model is not LCM, merge LCM
    stream.load_lcm_lora("latent-consistency/lcm-lora-sdv1-5")
    stream.fuse_lora()

    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # Enable acceleration
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = "black dog, flying on fire wings"

    # Prepare the stream
    stream.prepare(prompt, strength=2.0)

    # Prepare image
    init_image = load_image("/Users/himmelroman/projects/speechualizer/StreamDiffusion/assets/img2img_example.png").resize((512, 512))

    # Warmup >= len(t_index_list) x frame_buffer_size
    # print('Warming up')
    # for _ in range(2):
    #     stream(init_image)

    # Run the stream infinitely
    while True:
        print('Processing image')
        t_start = perf_counter()
        x_output = stream(init_image)
        t_stream = perf_counter() - t_start
        image = postprocess_image(x_output, output_type="pil")[0]
        t_post = perf_counter() - t_start - t_stream
        print(f'Image ready: {t_stream=}, {t_post=}')

        # input_response = input("Press Enter to continue or type 'stop' to exit: ")
        # if input_response == "stop":
        #     break


if __name__ == '__main__':

    run_test()
