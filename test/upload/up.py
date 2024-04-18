import requests
import PIL.Image

from io import BytesIO

# load file
files = [('file', open('/Users/himmelroman/Desktop/Bialik/snippet/out.png', 'rb'))]

# base url
url = 'http://52.208.62.108:8080/api/imagine/img2img_upload'

# post data
resp = requests.post(
    url=url,
    params={
        'prompt': 'custom art, futuristic shit',
        'num_inference_steps': 7,
        'guidance_scale': 0.8,
        'strength': 0.6,
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
