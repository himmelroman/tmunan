from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from starlette.middleware import Middleware
# from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from imagine.lcm import LCM
from diffusers.utils import make_image_grid, load_image

# LCM
lcm = LCM(txt2img_size='medium', img2img_size='small')
lcm.load(torch_device='mps')

# middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*']
    )
]

# FastAPI app
app = FastAPI(middleware=middleware)
# app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# env
CACHE_DIR = '/Users/himmelroman/projects/speechualizer/tmunan/cache'


class Prompt_Txt2Img(BaseModel):
    prompt: str
    height: int | None = 512
    width: int | None = 512
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.0
    seed: int | None = None


class Prompt_Img2Img(BaseModel):
    prompt: str
    image_id: str
    height: int | None = 512
    width: int | None = 512
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.0
    strength: float | None = 0.45
    grid: bool | None = False


@app.post("/txt2img",)
def txt2img(prompt: Prompt_Txt2Img, request: Request):

    # generate image
    images = lcm.txt2img(
        prompt=prompt.prompt,
        height=prompt.height,
        width=prompt.width,
        num_inference_steps=prompt.num_inference_steps,
        guidance_scale=prompt.guidance_scale,
        seed=prompt.seed,
        randomize_seed=prompt.seed is None
    )

    # save image to file
    image_id = f't2i_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{CACHE_DIR}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return {'image_id': image_id, 'image_url': f'{request.base_url}images/{image_id}'}


@app.post("/img2img",)
def img2img(prompt: Prompt_Img2Img, request: Request):

    # build image path
    image_url = f'{CACHE_DIR}/{prompt.image_id}.png'

    # generate image
    images = lcm.img2img(
        image_url=image_url,
        prompt=prompt.prompt,
        height=prompt.height,
        width=prompt.width,
        num_inference_steps=prompt.num_inference_steps,
        guidance_scale=prompt.guidance_scale,
        strength=prompt.strength
    )

    # save image to file
    image_id = f'i2i_{prompt.image_id}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{CACHE_DIR}/{image_id}.png'
    images[0].save(file_path)

    # generate grid of original image + new image?
    if prompt.grid:

        # save grid to file
        image_id = f'i2i_grid_{image_id}'
        file_path = f'{CACHE_DIR}/{image_id}.png'
        init_image = load_image(image_url)
        make_image_grid([init_image, images[0]], rows=1, cols=2).save(file_path)

    # return file
    return {'image_id': image_id, 'image_url': f'{request.base_url}images/{image_id}'}


@app.get("/images/{image_id}",)
def img2img(image_id: str):

    # return file
    file_path = f'{CACHE_DIR}/{image_id}.png'
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
