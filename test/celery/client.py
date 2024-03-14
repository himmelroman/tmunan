from PIL import Image
from tmunan.celery.app import txt2img, img2img

result = txt2img.apply_async(kwargs=dict(
    prompt='crocodile swalloed a huge guitar and feels bad about it',
    height= 512,
    width= 512,
    num_inference_steps=4,
    guidance_scale=0.5,
    seed=12345,
    randomize_seed=False),
    serializer='pickle')

image = result.get(timeout=5)
#print(pro)
#pho = Image.fromarray(np_pho)
image.show()
