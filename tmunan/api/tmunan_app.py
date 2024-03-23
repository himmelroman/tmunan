import os
import uuid

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, WebSocket, Request, status
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings

from starlette.middleware import Middleware
from starlette.websockets import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from tmunan.theatre.workers import AppWorkers
from tmunan.api.websocket import WebSocketConnectionManager
from tmunan.api.pydantic_models import ImageInstructions, ImageSequenceScript, TextInstructions, ReadTextPrompt
from tmunan.theatre.performance_factory import create_performance, PerformanceType


class AppSettings(BaseSettings):

    # fields
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", 'theatre')

    def __init__(self, **values):
        super().__init__(**values)

        # ensure cache dir exists
        Path.mkdir(Path(self.cache_dir), parents=True, exist_ok=True)
        Path.mkdir(Path(self.cache_dir) / 'hls', parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # determine model size
    # model_size = 'large' if torch.cuda.is_available() else 'medium'

    # determine imagine api address
    api_address = os.environ['API_ADDRESS']
    api_port = os.environ['API_PORT']

    # pre-start global workers
    fastapi_app.workers.init_imagine(api_base_address=api_address, api_port=api_port)
    fastapi_app.workers.init_read()
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
app.mount("/hls", StaticFiles(directory=app.context.cache_dir + '/hls'), name="hls")


@app.post("/api/read/prompt",)
def post_text(input_text: ReadTextPrompt, status_code=status.HTTP_200_OK):

    # consume posted text
    app.workers.read.push_text(input_text.text)
    return {'success': True}


@app.post("/api/script",)
def script(script: ImageSequenceScript, img_config: ImageInstructions, text_config: TextInstructions,
           request: Request, background_tasks: BackgroundTasks, status_code=status.HTTP_202_ACCEPTED):

    # gen id
    script_id = str(uuid.uuid4())[:8]
    script_dir = Path(app.context.cache_dir) / f'script_{script_id}'

    # init
    app.workers.init_display(output_dir=Path(app.context.cache_dir),
                             image_height=img_config.height, image_width=img_config.width,
                             kf_period=img_config.key_frame_period, kf_repeat=img_config.key_frame_repeat,
                             fps=img_config.output_fps)

    # set text instructions
    app.workers.read.set_instructions(text_config)

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

#
# @app.websocket("/api/ws")
# async def websocket_endpoint(websocket: WebSocket):
#
#     # wait for connection
#     await app.ws_manager.connect(websocket)
#
#     try:
#         while True:
#
#             # get WS message
#             data = await websocket.receive()
#             if 'bytes' in data:
#
#                 pass
#
#                 # push to ASR
#                 # print(f'Pushing audio into Listen: {len(data["bytes"])}')
#                 # app.workers.listen.push_audio(data['bytes'])
#
#     except (WebSocketDisconnect, RuntimeError) as ex:
#         print(f'WS disconnected... {ex}')
#         app.ws_manager.disconnect(websocket)


if __name__ == "__main__":

    # setup local env config
    # os.environ['API_ADDRESS'] = 'http://localhost'
    os.environ['API_ADDRESS'] = 'http://3.255.31.250'
    os.environ['API_PORT'] = '8080'

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)

    # HF_HUB_OFFLINE=1
