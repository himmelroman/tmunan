import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from diffusers.utils import make_image_grid

WORK_DIR = f'/tmp/lcm_large/'


def gen_image(prompt_dict):

    image = pipe(**prompt_dict,
                 num_inference_steps=5,
                 guidance_scale=1.4,
                 height=768, width=768,
                 num_images_per_prompt=1).images[0]
    return image


def gen_prompt(prompts, weights, seed):

    # format
    prompts_string = ', '.join(f'"{p}"' for p in prompts)
    weights_string = ', '.join(str(w) for w in weights)

    # create prompt
    conditioning, pooled = compel(f'({prompts_string}).blend({weights_string})')
    generator = [torch.Generator().manual_seed(seed)]

    return dict(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator)


if __name__ == "__main__":

    print('loading models...')
    unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("mps")

    seed = 6789

    compel = Compel(
      tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
      text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
      returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
      requires_pooled=[False, True]
    )

    print('generating img...')
    images = []

    # prompts = ["medieval oil painting, city on a hill, river and moat", "jungle", "bear hidding"]
    # prompt_dict = gen_prompt(prompts, [1.0, 0.7, 0.7], seed)
    # image = gen_image(prompt_dict)
    # image.show()

    prompts = ["medieval oil painting, city on a hill, river and moat", "jungle"]
    for i in range(0, 20, 2):

        prompt_dict = gen_prompt(prompts, [1.0, i / 10], seed)
        image = gen_image(prompt_dict)
        image.save(f'{WORK_DIR}/img_{i}.png')
        images.append(image)

    grid = make_image_grid(images, rows=2, cols=5)
    grid.show()
