from diffusers.utils import load_image
from tmunan.imagine.sd_lcm.lcm import LCM

if __name__ == '__main__':

    # load model
    lcm = LCM(img2img_size='large')
    lcm.load()

    res = lcm.img2img(
        prompt='bunny running around screaming at everybody',
        image_url='https://d6a9-62-56-134-6.ngrok-free.app/eye.png',
        num_inference_steps=5,
        guidance_scale=0.5,
        height=768, width=768
    )
    res[0].show()
