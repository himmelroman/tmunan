import time

from tmunan.imagine.image_generator import ImageGeneratorRemote
from tmunan.imagine.sd_lcm.lcm import LCM
from diffusers.utils import load_image, make_image_grid

if __name__ == '__main__':

    images = list()

    # lcm = LCM(model_size='medium')
    # lcm.load()
    rig = ImageGeneratorRemote('http://localhost', 8080)
    rig.on_image_ready += lambda img_id, img: images.append(img)

    # rig.txt2img(
    #     prompt='Realistic photo of city skyline with skyscrapers',
    #     num_inference_steps=5,
    #     guidance_scale=0.5,
    #     height=768, width=768,
    #     seed=4574578
    # )

    for i in range(10):

        weight_step = (1.0 - 0.7) / 100  # How much is one percent?
        strength = 0.7 + (weight_step * (i / 6 * 100))  # How progressed is this sequence?

        print(f'Generating: {strength=}')
        rig.img2img(
            image_url='http://localhost:9000/original_image.png',
            prompt='Buildings made of melting ice cream',
            num_inference_steps=5,
            strength=strength,
            guidance_scale=0.3,
            height=768, width=768,
        )
