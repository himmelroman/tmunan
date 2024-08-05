import os
import uuid
import mimetypes
from contextlib import asynccontextmanager
from http.client import HTTPException
from multiprocessing import freeze_support

from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from tmunan.imagine.stream_manager import StreamManager, ServerFullException
from tmunan.imagine.image_generator.image_generator import ImageGeneratorWorker

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):

    # determine imagine api address
    # api_address = os.environ['API_ADDRESS']
    # api_port = os.environ['API_PORT']

    # start image generator worker
    fastapi_app.image_generator.start()

    # FastAPI app lifespan
    yield

    # Clean up and release resources
    pass


class App:
    def __init__(self):

        # create fastapi app
        self.app = FastAPI(lifespan=lifespan)

        # load image generator
        mode = os.environ.get('TMUNAN_IMAGE_MODE', 'stream')
        if mode == 'stream':
            self.image_generator = ImageGeneratorWorker(model_id='sd-turbo', diff_type=mode)
        elif mode == 'control':
            self.image_generator = ImageGeneratorWorker(model_id='hyper-sd', diff_type=mode)

        # subscribe to events
        self.image_generator.on_image_ready += self.handle_image_ready
        # self.image_generator.on_startup = Event()
        # self.image_generator.on_shutdown = Event()

        # stream management
        self.stream_manager = StreamManager(max_streams=1)
        self.stream_manager.on_input_ready += lambda req: self.image_generator.img2img(**req)

        # init app
        self.app.stream_manager = self.stream_manager
        self.app.image_generator = self.image_generator
        self.init_app()

    def handle_image_ready(self, image):
        self.stream_manager.stream.distribute_output(image)

    def init_app(self):

        self.app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws")
        async def websocket(name: str, websocket: WebSocket):

            # accept incoming ws connection
            await websocket.accept()

            connection_id = uuid.uuid4()
            try:
                await self.stream_manager.connect(name=name, connection_id=connection_id, websocket=websocket)
                await self.stream_manager.handle_websocket(connection_id)

            finally:
                await self.stream_manager.disconnect(connection_id)

        @self.app.get("/api/stream")
        async def stream():

            try:

                # consume stream
                consumer_id = uuid.uuid4()
                return StreamingResponse(
                    self.stream_manager.handle_consumer(consumer_id),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except ServerFullException as ex:
                return HTTPException('Server is full')

        # if not os.path.exists("public"):
        #     os.makedirs("public")

        # # serve FE
        # fe_path = os.environ.get(
        #     'STATIC_SERVE',
        #     '/Users/himmelroman/projects/speechualizer/StreamDiffusion/demo/realtime-img2img/frontend/public')
        # print(f"FE Path: {fe_path}")
        # if Path(fe_path).exists():
        #     self.app.mount("/", StaticFiles(directory=fe_path, html=True), name="public")
        #     print(f"Mounted static: {fe_path}")


app = App().app

if __name__ == "__main__":
    freeze_support()

    import uvicorn

    uvicorn.run(
        "stream_app:app",
        host="0.0.0.0",
        port=8080,
        # reload=config.reload,
        # ssl_certfile=config.ssl_certfile,
        # ssl_keyfile=config.ssl_keyfile,
    )
