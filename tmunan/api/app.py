import os
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, WebSocket, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from starlette import status
from starlette.middleware import Middleware
from starlette.websockets import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from tmunan.api.context import WebSocketConnectionManager, context
from tmunan.api.pydantic_models import Instructions, ImageSequence
from tmunan.api.sequence import generate_image_sequence

from tmunan.imagine.lcm import LCM, load_image, make_image_grid


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # Init context
    context.ws_manager = WebSocketConnectionManager()

    # Load LCM
    context.lcm = LCM(txt2img_size='small')
    context.lcm.load()

    # FastAPI lifespan
    yield

    # Clean up the ML models and release the resources
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
app.mount("/ui", StaticFiles(directory=Path(os.getcwd()).with_name('ui')), name="ui")


@app.get("/images/{image_id}",)
def get_image_by_id(image_id: str):

    # return file
    file_path = f'{context.cache_dir}/{image_id}.png'
    return FileResponse(file_path)


@app.post("/txt2img",)
def txt2img(prompt: str, config: Instructions, req: Request):

    # generate image
    print(f'Generating image with prompt: {prompt}')
    images = context.lcm.txt2img(
        prompt=prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height, width=config.width,
        seed=config.seed,
        randomize_seed=config.seed is None
    )

    # save image to file
    image_id = f'txt2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return {'image_id': image_id, 'image_url': f'{req.base_url}images/{image_id}'}


@app.post("/img2img",)
def img2img(prompt: str, image_id: str, config: Instructions, request: Request, grid: bool = False):

    # build image path
    image_url = f'{context.cache_dir}/{image_id}.png'

    # generate image
    images = context.lcm.img2img(
        image_url=image_url,
        prompt=prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height, width=config.width,
        strength=config.strength
    )

    # save image to file
    image_id = f'i2i_{image_id}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # generate grid of original image + new image?
    if grid:

        # save grid to file
        image_id = f'i2i_grid_{image_id}'
        file_path = f'{context.cache_dir}/{image_id}.png'
        init_image = load_image(image_url)
        make_image_grid([init_image, images[0]], rows=1, cols=2).save(file_path)

    # return file
    return {'image_id': image_id, 'image_url': f'{request.base_url}images/{image_id}'}


@app.post("/sequence",)
def sequence(seq: ImageSequence, background_tasks: BackgroundTasks, status_code=status.HTTP_202_ACCEPTED):

    # start a sequence generation task
    background_tasks.add_task(generate_image_sequence, seq)

    # return file
    return {'sequence_id': 1}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    # wait for connection
    await context.ws_manager.connect(websocket)

    try:
        # dummy handler for incoming messages
        while True:
            await websocket.receive_json()

    except WebSocketDisconnect:
        context.ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
