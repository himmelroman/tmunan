from imagine.lcm import LCM
from diffusers.utils import make_image_grid, load_image

# build image path
WORK_DIR = f'/tmp/lcm_sequence_test/'


def gen_image(prompt):

    # generate image
    images = lcm.txt2img(
        prompt=prompt,
        height=768,
        width=768,
        num_inference_steps=20,
        guidance_scale=0.6,
        randomize_seed=True
    )
    # save image to file
    image_path = f'{WORK_DIR}/init_image.png'
    images[0].save(image_path)

    return image_path


def gen_strength_sequence(init_image_path, start=15, end=80):

    for strength in range(start, end, 10):

        # generate image
        images = lcm.img2img(
            image_url=init_image_path,
            prompt=prompt2,
            height=1024,
            width=1024,
            num_inference_steps=6,
            guidance_scale=1,
            strength=strength / 100
        )

        # save image to file
        image_id = f'str_{strength}'
        file_path = f'{WORK_DIR}/{image_id}.png'
        images[0].save(file_path)


def gen_morph_sequence(init_image_path, frames=10, strength=25):

    for frame_id in range(frames):

        # generate image
        images = lcm.img2img(
            image_url=init_image_path,
            prompt=prompt2,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=1,
            strength=strength / 100
        )

        # save image to file
        image_id = f'frame_{frame_id}'
        file_path = f'{WORK_DIR}/{image_id}.png'
        images[0].save(file_path)

        # base of new image
        init_image_path = file_path


def gen_scale_sequence(prompt1, prompt2, frames=10):

    seed = lcm.get_random_seed()

    prompt1_weight = 1.0
    prompt2_weight = 0.0

    for frame_id in range(frames):

        # generate image
        images = lcm.txt2img(
            prompt=f'("{prompt1}"){prompt1_weight} AND ("{prompt2}"){prompt2_weight}',
            height=768,
            width=768,
            num_inference_steps=5,
            guidance_scale=0.3,
            seed=seed
        )

        # save image to file
        file_path = f'{WORK_DIR}/sc_{frame_id}.png'
        images[0].save(file_path)

        # update weights
        prompt1_weight -= 1 / frames
        prompt2_weight += 1 / frames


if __name__ == "__main__":

    # LCM
    lcm = LCM(txt2img_size='large')
    lcm.load(torch_device='mps')

    prompt = 'medieval oil painting, clean, convent library, big wooden table, bearded priests sitting and drinking beer from bottles, sharp 4k'
    prompt2 = 'medieval oil painting, clean, convent library, big table, bearded priests lying around drunk, beer bottles scattered everywhere, sharp 4k'

    # init_image_path = gen_image(prompt)
    #init_image_path = '/private/tmp/lcm_sequence_test/blowers big.png'

    gen_strength_sequence(init_image_path, start=15, end=100)
    gen_morph_sequence(init_image_path, frames=8, strength=25)
    # gen_scale_sequence(prompt, prompt2, frames=10)
