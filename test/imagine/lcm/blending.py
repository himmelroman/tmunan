import torch
from tmunan.imagine.sd_lcm.lcm import LCM

torch.set_grad_enabled(False)


@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Sllatents1erp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0:
            First tensor for interpolation
        p1:
            Second tensor for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """

    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.float()
    p1 = p1.float()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


# def get_prompt_embeds(prompt):
#
#     (
#         prompt_embeds,
#         negative_prompt_embeds,
#         pooled_prompt_embeds,
#         negative_pooled_prompt_embeds,
#     ) = pipe.encode_prompt(
#         prompt=prompt,
#         prompt_2=prompt,
#         device="mps",
#         num_images_per_prompt=1,
#         do_classifier_free_guidance=True,
#         negative_prompt="",
#         negative_prompt_2="",
#         prompt_embeds=None,
#         negative_prompt_embeds=None,
#         pooled_prompt_embeds=None,
#         negative_pooled_prompt_embeds=None,
#         lora_scale=0,
#         clip_skip=False,
#     )
#     return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def blend_prompts(embeds1, embeds2, fract):
    """
    Blends two sets of prompt embeddings based on a specified fraction.
    """
    prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
    prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

    blended_prompt_embeds = interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
    blended_negative_prompt_embeds = interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
    blended_pooled_prompt_embeds = interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
    blended_negative_pooled_prompt_embeds = interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

    return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds


def blend_sequence_prompts(lcm: LCM, prompts, n_steps):
    """
    Generates a sequence of blended prompt embeddings for a list of text prompts.
    """
    blended_prompts = []
    for i in range(len(prompts) - 1):
        prompt_embeds1 = lcm.get_prompt_embeds(prompts[i])
        prompt_embeds2 = lcm.get_prompt_embeds(prompts[i + 1])
        for step in range(n_steps):
            fract = step / float(n_steps - 1)
            blended = blend_prompts(prompt_embeds1, prompt_embeds2, fract)
            blended_prompts.append(blended)
    return blended_prompts


if __name__ == '__main__':

    # Image generation pipeline
    lcm = LCM(model_size='large-turbo')
    lcm.load()

    # Example usage
    prompts = ["a man walking through the forest", "a man walking through the desert"] #, "a man walking through the village", "a man walking through the war in the village","a man walking through the war in the village with explosions","a man walking through the destructed village, dead bodies, gore" , "a man walking through the desert", "a man walking through the forest"]
    n_steps = 10
    blended_prompts = blend_sequence_prompts(lcm, prompts, n_steps)

    # random latents?
    latents1 = torch.randn((1, 4, 64, 64)).half().to('mps')  # .cuda()
    latents2 = torch.randn((1, 4, 64, 64)).half().to('mps')  # .cuda()

    # Iterate over blended prompts
    for i in range(len(blended_prompts) - 1):
        fract = float(i) / (len(blended_prompts) - 1)
        blended = blend_prompts(blended_prompts[i], blended_prompts[i+1], fract)

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended

        # Calculate fract and blend latents
        latents = interpolate_spherical(latents1, latents2, fract)

        # Generate the image using your pipeline
        image = lcm.txt2img_latents(guidance_scale=0.0,
                                    num_inference_steps=4,
                                    height=768, width=768,
                                    latents=latents,
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)[0]

        # save image to disk
        print(f'Saving image: {i}')
        image.save(f'/tmp/{i}.png')

    #
    # lcm = LCM(model_size='large')
    # lcm.load()
    # res = lcm.txt2img(
    #     prompt='bunny running around screaming at everybody',
    #     num_inference_steps=5,
    #     guidance_scale=0.5,
    #     height=768, width=768,
    #     seed=123,
    #     randomize_seed=False
    # )
    # res[0].save('test_image.png')
