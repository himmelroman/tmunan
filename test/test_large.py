from tmunan.imagine.lcm import LCM
from tmunan.imagine.lcm_large import LCMLarge

WORK_DIR = f'/tmp/lcm_large/'


if __name__ == "__main__":

    print('loading models...')
    lcm = LCM(txt2img_size='large')
    lcm.load()

    txt2img_images = lcm.txt2img(
        prompt='old man in the rain, focus on face, (laughing)1.4',
        # weight_list=[1],
        height=768,
        width=768,
        num_inference_steps=4,
        guidance_scale=0.5,
        seed=11322831497147
    )
    # txt2img_images[0].save('/tmp/source_image.png')
    txt2img_images[0].show()

    # print('generating img...')
    # img2img_images = lcm.img2img(
    #     image_url='/tmp/source_image.png',
    #     prompt='grandmother in bed, wolf standing next to her, (photo realistic)1.8',
    #     height=768,
    #     width=768,
    #     num_inference_steps=5,
    #     guidance_scale=1.2,
    #     strength=0.7
    # )
    # img2img_images[0].show()

    # grid = make_image_grid([txt2img_images[0], img2img_images[0]], rows=1, cols=2)
    # grid.show()
