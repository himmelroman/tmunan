import os
from datetime import datetime

from pathlib import Path
from typing import Annotated
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi import UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic_settings import BaseSettings
from starlette.background import BackgroundTask

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.api.pydantic_models import ImageInstructions, Prompt, BaseImage
from tmunan.imagine.sd_lcm.lcm_stream import StreamLCM


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
    cuda_model_size = os.environ.get('CUDA_MODEL_SIZE') or 'large'
    other_model_size = os.environ.get('OTHER_MODEL_SIZE') or 'small'
    model_size = cuda_model_size if torch.cuda.is_available() else other_model_size

    # LCM
    # app.lcm = LCM(model_size=model_size, ip_adapter_folder='/home/ubuntu/.cache/theatre/imagine/rubin_style')
    app.lcm = StreamLCM(model_size='small')
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
app.in_progress = False

# add gzip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain('/home/ubuntu/tmunan/cert.pem', keyfile='/home/ubuntu/tmunan/key.pem')

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
        image=base_image.image_url,
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
def img2img_upload(
        file: UploadFile,
        prompt: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
        strength: Annotated[float | None, Query(ge=0, le=1.0)] = None,
        guidance_scale: Annotated[float | None, Query(ge=0, le=1.0)] = None,
        ip_adapter_weight: Annotated[float | None, Query(ge=0, le=1.0)] = None,
        num_inference_steps: Annotated[int | None, Query(gt=2, le=50)] = None,
        seed: Annotated[int | None, Query(ge=0)] = None
):
    if app.in_progress:
        raise HTTPException(status_code=500, detail="Image generation in progress")

    # flag in progress
    app.in_progress = True

    try:
        # save uploaded file
        input_file_path = f'{app.context.cache_dir}/upload_img2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.png'
        save_file(file, input_file_path)

        # generate image
        images = app.lcm.img2img(
            image=input_file_path,
            prompt=prompt or "painting, art",
            num_inference_steps=num_inference_steps or 4,
            guidance_scale=guidance_scale or 1.0,
            strength=strength or 0.4,
            ip_adapter_weight=ip_adapter_weight or 0.8,
            height=1080, width=1920,
            seed=seed or 0,
            randomize_seed=seed is None
        )

        # save image to file
        image_id = f'img2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
        output_file_path = f'{app.context.cache_dir}/{image_id}.jpg'
        images[0].save(output_file_path)

        # respond
        return FileResponse(
            output_file_path,
            background=BackgroundTask(clean_files, [input_file_path, output_file_path])
        )

    except Exception as ex:
        raise HTTPException(status_code=500, detail="Image generation failed")


def clean_files(file_path_list):
    for file_path in file_path_list:
        Path(file_path).unlink(missing_ok=True)

    # clear in progress
    app.in_progress = False


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
