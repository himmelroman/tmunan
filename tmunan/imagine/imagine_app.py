import os
from datetime import datetime

from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from pydantic_settings import BaseSettings

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.api.pydantic_models import ImageInstructions, Prompt, BaseImage


class AppSettings(BaseSettings):

    # fields
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", 'theatre', 'imagine')

    def __init__(self, **values):
        super().__init__(**values)

        # ensure cache dir exists
        Path.mkdir(Path(self.cache_dir), parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # determine model size
    model_size = 'large' if torch.cuda.is_available() else 'small'

    # LCM
    app.lcm = LCM(model_size=model_size, ip_adapter_folder='/home/ubuntu/.cache/theatre/imagine/rubin_style')
    app.lcm.load()

    # FastAPI app lifespan
    yield

    # Clean up and release resources
    pass


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
app = FastAPI(middleware=middleware, lifespan=lifespan)

# App components
app.context = AppSettings()


@app.get("/api/imagine/{image_id}",)
def get_image_by_id(image_id: str):

    # return file
    file_path = f'{app.context.cache_dir}/{image_id}.png'
    return FileResponse(file_path)


@app.post("/api/imagine/txt2img",)
def txt2img(prompt: Prompt, img_config: ImageInstructions, req: Request):

    # generate image
    print(f'Generating image with prompt: {prompt}')
    images = app.lcm.txt2img(
        prompt=prompt.text,
        num_inference_steps=img_config.num_inference_steps,
        guidance_scale=img_config.guidance_scale,
        height=img_config.height, width=img_config.width,
        seed=img_config.seed,
        randomize_seed=img_config.seed is None
    )

    # save image to file
    image_id = f'txt2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{app.context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return {
        'image_id': image_id,
        'image_url': f'{req.base_url}api/imagine/{image_id}'
    }


@app.post("/api/imagine/img2img",)
def img2img(prompt: Prompt, base_image: BaseImage, img_config: ImageInstructions, request: Request):

    # generate image
    images = app.lcm.img2img(
        image_url=base_image.image_url,
        prompt=prompt.text,
        num_inference_steps=img_config.num_inference_steps,
        guidance_scale=img_config.guidance_scale,
        height=img_config.height, width=img_config.width,
        strength=img_config.strength,
        ip_adapter_weight=img_config.ip_adapter_weight,
        seed=img_config.seed,
        randomize_seed=img_config.seed is None
    )

    # save image to file
    image_id = f'img2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{app.context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return {
        'image_id': image_id,
        'image_url': f'{request.base_url}api/imagine/{image_id}'
    }


@app.post("/api/imagine/img2img_upload")
def img2img_upload(file: UploadFile = File(...)):

    # save uploaded file
    file_path = f'{app.context.cache_dir}/upload_img2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.png'
    save_file(file, file_path)

    # generate image
    images = app.lcm.img2img(
        image_url=file_path,
        prompt="high quality",
        num_inference_steps=6,
        guidance_scale=1.0,
        height=768, width=768,
        strength=0.5,
        ip_adapter_weight=0.7,
        seed=0,
        randomize_seed=True
    )

    # save image to file
    image_id = f'img2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{app.context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return FileResponse(file_path)


def save_file(file: UploadFile, target_path):
    try:
        with open(target_path, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)

    except Exception:
        return {"message": "There was an error uploading the file"}

    finally:
        file.file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

    # HF_HUB_OFFLINE=1
