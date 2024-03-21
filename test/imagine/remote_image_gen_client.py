import time

from diffusers.utils import make_image_grid

from tmunan.imagine.image_generator import RemoteImageGenerator


if __name__ == '__main__':

    images = list()

    rig = RemoteImageGenerator('http://localhost', 8080)
    rig.on_image_ready += lambda img: images.append(img)

    print('Requesting images')
    rig.txt2img(
        prompt='two winged racoons fighting in the sky over a chaotic battlefield involving many types of armoured animals.',
        num_inference_steps=4,
        guidance_scale=0.4,
        height=768,
        width=768,
        seed=45632
    )
    rig.txt2img(
        prompt='two winged foxes fighting in the sky over a chaotic battlefield involving many types of armoured animals.',
        num_inference_steps=4,
        guidance_scale=0.4,
        height=768,
        width=768,
        seed=45632
    )
    rig.txt2img(
        prompt='two winged badgers fighting in the sky over a chaotic battlefield involving many types of armoured animals.',
        num_inference_steps=4,
        guidance_scale=0.4,
        height=768,
        width=768,
        seed=45632
    )

    print('Waiting for generation')
    while len(images) != 3:
        time.sleep(0.1)

    print('Showing grid')
    image_grid = make_image_grid(images, rows=1, cols=3)
    image_grid.show()

    print('Exiting')