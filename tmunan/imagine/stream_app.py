import asyncio
import copy
import io
import os
import sys
import uuid
import logging
import mimetypes
from pathlib import Path

import markdown2
from PIL import Image
from types import SimpleNamespace

from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from tmunan.imagine.connection_manager import ConnectionManager, ServerFullException
from tmunan.imagine.sd_lcm.lcm_stream import StreamLCM

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

page_content = """<h1 class="text-3xl font-bold">Tmunan - StreamApp</h1>
<h3 class="text-xl font-bold">Tmunan - Image-to-Image</h3>
<p class="text-sm">
    This is a demo showcases 
    <a
        href="https://github.com/himmelroman/StreamDiffusion"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">StreamDiffusion (himmelroman)
    </a>
    Image to Image pipeline with a MJPEG stream server.
</p>
"""


class ServerInfo(BaseModel):
    name: str = "StreamDiffusion img2img"
    input_mode: str = "image"
    page_content: str = page_content


class StreamInputParams(BaseModel):
    prompt: str = Field(
        'Default prompt text - change in the code!',
        title="Prompt",
        field="textarea",
        id="prompt",
    )
    strength: float = Field(
        1.0, min=1.0, max=2.5, title="Strength", disabled=True, hide=True, id="strength"
    )
    guidance_scale: float = Field(
        1.0, min=1.0, max=2.5, title="Guidance Scale", disabled=True, hide=True, id="guidance_scale"
    )
    seed: int = Field(
        0, min=0, max=sys.maxsize, title="Seed", disabled=True, hide=True, id="seed"
    )
    # negative_prompt: str = Field(
    #     default_negative_prompt,
    #     title="Negative Prompt",
    #     field="textarea",
    #     id="negative_prompt",
    # )

    width: int = Field(
        512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
    )
    height: int = Field(
        512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
    )


class App:
    def __init__(self):
        self.app = FastAPI()
        self.stream_lcm = StreamLCM(model_size='turbo')
        self.conn_manager = ConnectionManager()
        self.init_app()

        self.default_params = {
            'prompt': 'Lions in the sky',
            'strength': 1.0
        }
        self.param_cache = StreamInputParams(**self.default_params)
        # self.param_cache = SimpleNamespace(**self.param_cache.model_dump())

    def init_app(self):

        self.stream_lcm.load()

        self.app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):

            try:
                await self.conn_manager.connect(
                    user_id, websocket, 0   # self.args.max_queue_size
                )
                await handle_websocket_data(user_id)

            except ServerFullException as e:
                logging.error(f"Server Full: {e}")

            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            # last_time = time.time()
            try:
                while True:

                    # if (
                    #     self.args.timeout > 0
                    #     and time.time() - last_time > self.args.timeout
                    # ):
                    #     await self.conn_manager.send_json(
                    #         user_id,
                    #         {
                    #             "status": "timeout",
                    #             "message": "Your session has ended",
                    #         },
                    #     )
                    #     await self.conn_manager.disconnect(user_id)
                    #     return

                    message = await self.conn_manager.receive(user_id)
                    if message is None:
                        await asyncio.sleep(0.01)
                        continue

                    elif message['type'] == 'json':
                        self.param_cache = StreamInputParams(**message['data'])
                        # self.param_cache = SimpleNamespace(**self.param_cache.model_dump())

                    elif message['type'] == 'bytes':

                        if self.param_cache is None:
                            logging.warning('Image arrived, but params not initialized')
                            continue

                        if message['data'] is None or len(message['data']) == 0:
                            logging.warning('Got empty data blob')
                            continue

                        params = copy.deepcopy(self.param_cache)
                        params = params.model_dump()
                        params['image'] = self.bytes_to_pil(message['data'])
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.exception(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:

                async def generate():
                    while True:

                        # await self.conn_manager.send_json(
                        #     user_id, {"status": "send_frame"}
                        # )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            await asyncio.sleep(0.01)
                            continue

                        print('Starting img2img')
                        params = SimpleNamespace(**params)
                        image = self.stream_lcm.img2img(
                            prompt=params.prompt,
                            image=params.image,
                            guidance_scale=params.guidance_scale,
                            strength=params.strength,
                            seed=params.seed,
                            height=params.height,
                            width=params.width
                        )[0]
                        if image is None:
                            continue
                        frame = self.pil_to_frame(image)
                        yield frame
                        #if self.args.debug:
                        #print(f"Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )

            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = ServerInfo.model_json_schema()
            info = ServerInfo()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = StreamInputParams.model_json_schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "page_content": page_content if info.page_content else "",
                }
            )

        if not os.path.exists("public"):
            os.makedirs("public")

        # serve FE
        fe_path = os.environ.get(
            'STATIC_SERVE',
            '/Users/himmelroman/projects/speechualizer/StreamDiffusion/demo/realtime-img2img/frontend/public')
        print(f"FE Path: {fe_path}")
        if Path(fe_path).exists():
            self.app.mount("/", StaticFiles(directory=fe_path, html=True), name="public")
            print(f"Mounted static: {fe_path}")

    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        return image

    @staticmethod
    def pil_to_frame(image: Image.Image) -> bytes:
        frame_data = io.BytesIO()
        image.save(frame_data, format="WEBP", quality=100, method=6)
        frame_data = frame_data.getvalue()
        return (
                b"--frame\r\n"
                + b"Content-Type: image/webp\r\n"
                + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
                + frame_data
                + b"\r\n"
        )


app = App().app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "stream_app:app",
        host="0.0.0.0",
        port=8080,
        # reload=config.reload,
        # ssl_certfile=config.ssl_certfile,
        # ssl_keyfile=config.ssl_keyfile,
    )
