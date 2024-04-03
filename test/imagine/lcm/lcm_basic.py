from tmunan.imagine.sd_lcm.lcm import LCM
from diffusers.utils import make_image_grid

if __name__ == '__main__':

    lcm = LCM(txt2img_size='small')
    lcm.load()

    res = lcm.txt2img(
        prompt='forest',
        # negative_prompt='roots',
        num_inference_steps=4,
        guidance_scale=0.5,
        height=512, width=512,
        seed=123,
        randomize_seed=False
    )

    res2 = lcm.txt2img(
        prompt='forest',
        num_inference_steps=4,
        guidance_scale=0.5,
        height=512, width=512,
        seed=123,
        randomize_seed=False
    )

    make_image_grid([res[0], res2[0]], rows=1, cols=2).show()
