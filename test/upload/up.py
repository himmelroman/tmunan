import requests
import PIL.Image

from io import BytesIO

from tmunan.common.utils import load_image
from tmunan.imagine.image_generator import ImageGeneratorRemote

#
# # load file
# files = [('file', open('/Users/himmelroman/Desktop/Bialik/snippet/h2.jpeg', 'rb'))]
#
# # base url
# url = 'http://54.74.136.111:8080/api/imagine/img2img_upload'
#
# # post data
# resp = requests.post(
#     url=url,
#     params={
#         'prompt': 'Reuven Rubin painting, woman face in hoodie',
#         'guidance_scale': 0.6,
#         'strength': 0.5,
#         'ip_adapter_weight': 0.9,
#         'num_inference_steps': 8
#     },
#     files=files,
#     stream=True
# )
#
# if resp.status_code == 200:
#     img = PIL.Image.open(BytesIO(resp.content))
#     img.show()
# else:
#     print(resp.json())
#     resp.raise_for_status()
#

image = load_image('/Users/himmelroman/Desktop/Bialik/birds.png')

igr = ImageGeneratorRemote('http://3.255.31.250', '8080')

small_image = image.resize((512, 512))

res = igr.img2img_upload(
    image=small_image,
    prompt='panda',
    width=512,
    height=512,
    guidance_scale=0.5,
    num_inference_steps=4,
    strength=0.4,
    randomize_seed=True
)
res.show()
