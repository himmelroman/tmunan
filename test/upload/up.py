import requests
import PIL.Image

from io import BytesIO

# load file
files = [('file', open('/Users/himmelroman/Desktop/Bialik/me.png', 'rb'))]

# base url
url = 'http://54.74.136.111:8080/api/imagine/img2img_upload'

# post data
resp = requests.post(
    url=url,
    params={
        'prompt': 'Reuven Rubin painting',
        'guidance_scale': 0.8,
        'strength': 0.4,
        'ip_adapter_weight': 0.1,
        'num_inference_steps': 7,
        'seed': 7777777
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
