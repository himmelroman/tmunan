import io
import requests

from PIL import Image

for i in range(3):

    files = {'image': open('/Users/himmelroman/Desktop/Bialik/birds.png', 'rb')}
    data = {
        'prompt': 'flying lion',
        'num_inference_steps': 1,
        'guidance_scale': 1.0,
        'strength': 1.0,
        'height': 512,
        'width': 512,
        'seed': 12345
    }

    response = requests.post('http://localhost:8080/api/img2img', files=files, params=data)
    response.raise_for_status()

    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img.show()
    else:
        print(f"Error: {response.status_code}")
