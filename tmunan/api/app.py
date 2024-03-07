import os
import uuid

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, WebSocket, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings

from starlette import status
from starlette.middleware import Middleware
from starlette.websockets import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from tmunan.theatre.workers import AppWorkers
from tmunan.api.websocket import WebSocketConnectionManager
from tmunan.api.pydantic_models import ImageInstructions, ImageSequenceScript
from tmunan.theatre.performance_factory import create_performance, PerformanceType


class AppSettings(BaseSettings):

    # fields
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", 'theatre')

    def __init__(self, **values):
        super().__init__(**values)

        # ensure cache dir exists
        Path.mkdir(Path(self.cache_dir), parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # pre-start global workers
    fastapi_app.workers.init_imagine()
    # fastapi_app.workers.init_listen()
    pass

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
app.ws_manager = WebSocketConnectionManager()
app.context = AppSettings()
app.workers = AppWorkers()

# Static mounts
app.mount("/ui", StaticFiles(directory=Path(os.path.realpath(__file__)).parent.with_name('ui'), html=True), name="ui")
app.mount("/cache", StaticFiles(directory=app.context.cache_dir), name="cache")


@app.get("/api/images/{seq_id}/{image_id}",)
def get_image_by_id(seq_id: str, image_id: str):

    # return file
    file_path = f'{app.context.cache_dir}/{seq_id}/{image_id}.png'
    return FileResponse(file_path)


@app.get("/api/images/{script_id}/{seq_id}/{image_id}",)
def get_image_by_id(script_id: str, seq_id: str, image_id: str):

    # return file
    file_path = f'{app.context.cache_dir}/{script_id}/{seq_id}/{image_id}.png'
    return FileResponse(file_path)


# @app.post("/api/txt2img",)
# def txt2img(prompt: str, config: ImageInstructions, req: Request):
#
#     # generate image
#     print(f'Generating image with prompt: {prompt}')
#     blend = context.sd_lcm.txt2img(
#         prompt=prompt,
#         num_inference_steps=config.num_inference_steps,
#         guidance_scale=config.guidance_scale,
#         height=config.height, width=config.width,
#         seed=config.seed,
#         randomize_seed=config.seed is None
#     )
#
#     # save image to file
#     image_id = f'txt2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
#     file_path = f'{context.cache_dir}/{image_id}.png'
#     blend[0].save(file_path)
#
#     # return file
#     return {'image_id': image_id, 'image_url': f'{req.base_url}blend/{image_id}'}
#
#
# @app.post("/api/img2img",)
# def img2img(prompt: str, image_id: str, config: ImageInstructions, request: Request, grid: bool = False):
#
#     # build image path
#     image_url = f'{context.cache_dir}/{image_id}.png'
#
#     # generate image
#     blend = context.sd_lcm.img2img(
#         image_url=image_url,
#         prompt=prompt,
#         num_inference_steps=config.num_inference_steps,
#         guidance_scale=config.guidance_scale,
#         height=config.height, width=config.width,
#         strength=config.strength
#     )
#
#     # save image to file
#     image_id = f'i2i_{image_id}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
#     file_path = f'{context.cache_dir}/{image_id}.png'
#     blend[0].save(file_path)
#
#     # generate grid of original image + new image?
#     if grid:
#
#         # save grid to file
#         image_id = f'i2i_grid_{image_id}'
#         file_path = f'{context.cache_dir}/{image_id}.png'
#         init_image = load_image(image_url)
#         make_image_grid([init_image, blend[0]], rows=1, cols=2).save(file_path)
#
#     # return file
#     return {'image_id': image_id, 'image_url': f'{request.base_url}blend/{image_id}'}


@app.post("/api/script",)
def script(script: ImageSequenceScript, img_config: ImageInstructions,
           request: Request, background_tasks: BackgroundTasks, status_code=status.HTTP_202_ACCEPTED):

    # gen id
    script_id = str(uuid.uuid4())[:8]
    script_dir = Path(app.context.cache_dir) / f'script_{script_id}'

    # init
    app.workers.init_display(script_dir, img_config.height, img_config.width, img_config.images_per_second, fps=12)

    # start slideshow generation task
    slideshow = create_performance(PerformanceType.Slideshow, app)
    background_tasks.add_task(slideshow.run, script, img_config, script_id)

    # return file
    return {
        'script_id': script_id,
        'hls_uri': request.base_url.replace(path=f'ui/index.html?display_id=script_{script_id}')
    }


# @app.post("/api/stop",)
# def stop():
#     sequencer.stop()


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):

    # wait for connection
    await app.ws_manager.connect(websocket)

    try:
        while True:

            # get WS message
            data = await websocket.receive()
            if 'bytes' in data:

                pass

                # push to ASR
                # print(f'Pushing audio into Listen: {len(data["bytes"])}')
                # app.workers.listen.push_audio(data['bytes'])

    except (WebSocketDisconnect, RuntimeError) as ex:
        print(f'WS disconnected... {ex}')
        app.ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

    # HF_HUB_OFFLINE=1
