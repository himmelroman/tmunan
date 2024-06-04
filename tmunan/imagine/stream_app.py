import io
import os
import time
import uuid
import logging
import mimetypes
from pathlib import Path

import markdown2
from types import SimpleNamespace

from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse

from pydantic import BaseModel, Field

import torch
from PIL import Image
from tmunan.imagine.connection_manager import ConnectionManager, ServerFullException
from tmunan.imagine.sd_lcm.lcm_stream import StreamLCM

# from config import config, Args
# from util import pil_to_frame, bytes_to_pil
# from connection_manager import ConnectionManager, ServerFullException
# from img2img import Pipeline

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
        1.5, min=0.0, max=2.5, title="Strength", disabled=True, hide=True, id="strength"
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
        self.stream_lcm = StreamLCM(model_size='small')
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()

    def init_app(self):

        self.stream_lcm.load()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
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
            last_time = time.time()
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

                    data = await self.conn_manager.receive_json(user_id)
                    if data["status"] == "next_frame":
                        info = ServerInfo()
                        params = await self.conn_manager.receive_json(user_id)
                        params = StreamInputParams(**params)
                        params = SimpleNamespace(**params.model_dump())
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            params.image = self.bytes_to_pil(image_data)
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
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
                        last_time = time.time()
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        image = self.stream_lcm.img2img(
                            prompt=params.prompt,
                            image=params.image,
                            strength=params.strength
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
        image.save(frame_data, format="JPEG")
        frame_data = frame_data.getvalue()
        return (
                b"--frame\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
                + frame_data
                + b"\r\n"
        )


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch_dtype = torch.float16
# pipeline = Pipeline(config, device, torch_dtype)
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
