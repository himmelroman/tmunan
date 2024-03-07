import time

from tmunan.imagine.image_generator import ImageGenerator


def handle_image_ready(image):
    print('Image Ready!')
    image.save('test_image.png')
    t2i.stop()


def request_image():

    print('Worker Ready! Requesting image generation...')
    t2i.request_image(
        prompt='bunny running around screaming at everybody',
        num_inference_steps=5,
        guidance_scale=0.5,
        height=768, width=768,
        seed=123,
        randomize_seed=False
    )


def mark_shutdown():
    global running
    print('Marking shutdown')
    running = False


if __name__ == '__main__':

    running = True

    t2i = ImageGenerator(model_id='large')
    t2i.on_image_ready += handle_image_ready
    t2i.on_startup += request_image
    t2i.on_shutdown += mark_shutdown

    time.sleep(1)
    print('Starting Txt2Img')
    t2i.start()

    while running:
        time.sleep(1)
