import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime

from diffusers.utils import load_image
from fastapi import FastAPI, BackgroundTasks, WebSocket, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from starlette import status
from starlette.middleware import Middleware
from starlette.websockets import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from tmunan.api.context import Context, WebSocketConnectionManager
from tmunan.api.pydantic_models import Txt2ImgInstructions, ImageSequence

from tmunan.imagine.lcm_large import LCMLarge


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # Init context
    context.ws_manager = WebSocketConnectionManager()

    # Load LCM
    context.lcm = LCMLarge(model_id='sdxl')
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
context = Context()

app.mount("/ui", StaticFiles(directory="ui"), name="ui")


@app.get("/images/{image_id}",)
def get_image_by_id(image_id: str):

    # return file
    file_path = f'{context.cache_dir}/{image_id}.png'
    return FileResponse(file_path)


@app.post("/txt2img",)
def txt2img(prompt: Txt2ImgInstructions, req: Request):

    # generate image
    images = context.lcm.txt2img(
        prompt_list=prompt.prompt_list,
        weight_list=prompt.prompt_weights,
        num_inference_steps=prompt.num_inference_steps,
        guidance_scale=prompt.guidance_scale,
        height=prompt.height, width=prompt.width,
        seed=prompt.seed,
        randomize_seed=prompt.seed is None
    )

    # save image to file
    image_id = f'txt2img_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
    file_path = f'{context.cache_dir}/{image_id}.png'
    images[0].save(file_path)

    # return file
    return {'image_id': image_id, 'image_url': f'{req.base_url}images/{image_id}'}


@app.post("/sequences",)
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


def generate_image_sequence(seq: ImageSequence):

    # set up event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # iterate as many times as requested
    weight_step = 1 / (seq.num_images - 1)
    for i in range(0, seq.num_images):

        # gen prompt weights list
        weight_list = [1.0, round(i * weight_step, 3)]

        # gen image
        # images = context.lcm.txt2img(
        #     prompt_list=seq.txt2img.prompt_list,
        #     weight_list=weight_list,
        #     num_inference_steps=seq.txt2img.num_inference_steps,
        #     guidance_scale=seq.txt2img.guidance_scale,
        #     height=seq.txt2img.height, width=seq.txt2img.width,
        #     seed=seq.txt2img.seed,
        #     randomize_seed=seq.txt2img.seed is None
        # )
        time.sleep(3)
        images = [
            load_image(f'{context.cache_dir}/img_seq_{i}.png')
        ]

        # save image to disk
        image_id = f'img_seq_{i}'
        images[0].save(f'{context.cache_dir}/{image_id}.png')

        # notify image ready
        if context.ws_manager.active_connections:
            loop.run_until_complete(
                context.ws_manager.active_connections[0].send_json({
                    'event': 'IMAGE_READY',
                    'sequence_id': 1,
                    'image_id': image_id
                })
            )

    # notify sequence ended
    if context.ws_manager.active_connections:

        loop.run_until_complete(
            context.ws_manager.active_connections[0].send_json({
                'event': 'SEQUENCE_FINISHED',
                'sequence_id': 1
            })
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
