import requests
import PIL.Image

from io import BytesIO

# load file
files = [('file', open('/Users/himmelroman/Desktop/Bialik/snippet/h2.jpeg', 'rb'))]

# base url
url = 'http://54.74.136.111:8080/api/imagine/img2img_upload'

# post data
resp = requests.post(
    url=url,
    params={
        'prompt': 'Reuven Rubin painting, woman face in hoodie',
        'guidance_scale': 0.6,
        'strength': 0.5,
        'ip_adapter_weight': 0.9,
        'num_inference_steps': 8
    },
    files=files,
    stream=True
)

if resp.status_code == 200:
    img = PIL.Image.open(BytesIO(resp.content))
    img.show()
else:
    print(resp.json())
    resp.raise_for_status()
