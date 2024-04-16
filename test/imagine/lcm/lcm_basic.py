from tmunan.imagine.sd_lcm.lcm import LCM
from diffusers.utils import make_image_grid

if __name__ == '__main__':

    lcm = LCM(model_size='small', ip_adapter_folder='/Users/himmelroman/Desktop/Bialik/rubin_style')
    lcm.load()

    res = lcm.img2img(
        prompt='forest',
        image_url='/Users/himmelroman/Desktop/Bialik/rubin_style/me.jpg',
        ip_adapter_weight=0.7,
        num_inference_steps=4,
        guidance_scale=0.5,
        height=512, width=512,
        seed=123,
        randomize_seed=False
    )

    # res2 = lcm.txt2img(
    #     prompt='forest',
    #     num_inference_steps=4,
    #     guidance_scale=0.5,
    #     height=512, width=512,
    #     seed=123,
    #     randomize_seed=False
    # )

    make_image_grid([res[0], res2[0]], rows=1, cols=2).show()
