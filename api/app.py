from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware import Middleware
# from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from imagine.lcm import LCM
from diffusers.utils import make_image_grid, load_image

# LCM
lcm = LCM()
lcm.load(torch_device="mps", model_size='medium')

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


@app.get("/txt2img",)
def txt2img(prompt: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 4,
            guidance_scale: float = 0.0,
            seed: int = None):

    # generate image
    images = lcm.txt2img(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=seed is None
    )

    # save image to file
    file_path = f'{CACHE_DIR}/{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.png'
    images[0].save(file_path)

    # return file
    return FileResponse(file_path)


@app.get("/img2img",)
def img2img(prompt: str,
            image_id: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 4,
            guidance_scale: float = 2.0,
            strength: float = 0.5,
            grid: bool = False
            ):

    # build image path
    image_url = f'{CACHE_DIR}/{image_id}.png'

    # generate image
    images = lcm.img2img(
        image_url=image_url,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )

    # save image to file
    file_path = f'{CACHE_DIR}/{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.png'
    images[0].save(file_path)

    if grid:

        # save grid to file
        file_path = f'{CACHE_DIR}/grid_{image_id}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.png'
        init_image = load_image(image_url)
        make_image_grid([init_image, images[0]], rows=1, cols=2).save(file_path)

    # return file
    return FileResponse(file_path)


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
