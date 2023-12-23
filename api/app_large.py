import asyncio
import json
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, BackgroundTasks, WebSocket
from pydantic import BaseModel
from fastapi.responses import FileResponse
from starlette import status
from starlette.middleware import Middleware
# from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from diffusers.utils import make_image_grid, load_image
from starlette.websockets import WebSocketDisconnect

from imagine.lcm_large import LCMLarge

# LCM
lcm = LCMLarge(model_id='sdxl')
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
active_ws = None
# app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# env
CACHE_DIR = '/Users/himmelroman/projects/speechualizer/tmunan/cache'


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


class ImageSequence(BaseModel):
    prompt_list: List[str]
    prompt_weights: List[str]
    num_images: int = 8
    height: int | None = 512
    width: int | None = 512
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.0
    seed: int | None = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
            # await manager.send_personal_message(f"You wrote: {data}", websocket)
            # await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/sequences",)
def sequence(seq: ImageSequence, background_tasks: BackgroundTasks, status_code=status.HTTP_202_ACCEPTED):

    # start a sequence generation task
    background_tasks.add_task(generate_image_sequence, seq)

    # return file
    return {'sequence_id': 1}


@app.get("/images/{image_id}",)
def get_image_by_id(image_id: str):

    # return file
    file_path = f'{CACHE_DIR}/{image_id}.png'
    return FileResponse(file_path)


def generate_image_sequence(seq: ImageSequence):

    # set up event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # iterate as many times as requested
    for i in range(0, seq.num_images, 2):

        prompt_dict = lcm.gen_prompt(seq.prompt_list, [1.0, i / 10], seq.seed)
        image = lcm.txt2img(prompt_list=seq.prompt_list,
                            weight_list=[1.0, i / 10],
                            num_inference_steps=seq.num_inference_steps,
                            guidance_scale=seq.guidance_scale,
                            height=seq.height, width=seq.width,
                            seed=seq.seed)[0]

        # save image to disk
        image_id = f'img_seq_{i}'
        image.save(f'{CACHE_DIR}/{image_id}.png')

        # notify image ready
        if manager.active_connections:
            loop.run_until_complete(
                manager.active_connections[0].send_json({
                    'event': 'IMAGE_READY',
                    'sequence_id': 1,
                    'image_id': image_id
                })
            )

    # notify sequence ended
    if manager.active_connections:

        loop.run_until_complete(
            manager.active_connections[0].send_json({
                'event': 'SEQUENCE_FINISHED',
                'sequence_id': 1
            })
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
