from tmunan.imagine.sd_lcm.lcm import LCM
from diffusers.utils import make_image_grid

if __name__ == '__main__':

    lcm = LCM(model_size='sdxs')  #, ip_adapter_folder='/Users/himmelroman/Desktop/Bialik/rubin_style')
    lcm.load()

    res = lcm.txt2img(
        prompt='forest',
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512, width=512,
        seed=0,
        randomize_seed=True
    )

    res2 = lcm.img2img(
        prompt='forest',
        image='/Users/himmelroman/Desktop/Bialik/me.png',
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512, width=512,
        seed=0,
        randomize_seed=True
    )

    make_image_grid([res[0], res2[0]], rows=1, cols=2).show()
