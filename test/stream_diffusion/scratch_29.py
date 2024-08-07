from time import perf_counter

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def streamdiffusion_strength_test():

    # You can load any models using diffuser's StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    # pipe = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
        # device=torch.device("mps"),
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("mps")

    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=[32, 45],
        torch_dtype=torch.float16,
    )

    # If the loaded model is not LCM, merge LCM
    # stream.load_lcm_lora("latent-consistency/lcm-lora-sdv1-5")
    # stream.fuse_lora()

    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    # Enable acceleration
    # pipe.enable_xformers_memory_efficient_attention()

    # prepare input
    prompt = "black dog, flying on fire wings"
    # init_image = load_image("/Users/himmelroman/projects/speechualizer/StreamDiffusion/assets/img2img_example.png").resize((512, 512))
    init_image = load_image("/Users/himmelroman/Desktop/House/373408370_6478098395560526_1428679915222833350_n.jpg").resize((512, 512))

    for strength in [2.4]:

        # Prepare the stream
        t_start = perf_counter()
        stream.prepare(
            prompt=prompt,
            guidance_scale=float(1.2),
            strength=float(1.6),
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

        image.save(f'/tmp/strength_{strength}.png')


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

    streamdiffusion_strength_test()
