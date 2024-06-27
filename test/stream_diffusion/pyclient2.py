import io
import json
import logging
import time
import uuid
import asyncio
import aiohttp
import argparse
import websockets

from pathlib import Path
from json import JSONDecodeError

from tmunan.common.utils import load_image


class TmunanStreamTester:

    def __init__(self, host, port, input_mode, input_source):

        # client id
        self.client_id = str(uuid.uuid4())

        # input
        self.input_mode = input_mode
        self.input_source = input_source

        # server
        self.host = host
        self.port = port

        # throttling
        self.send_fps = 6

        self.websocket = None
        self.connected = False

    # async def start_streaming(self):
        # Run sender and stream tasks concurrently
        # sender_task = asyncio.create_task(self.sender_func())
        # stream_task = asyncio.create_task(self.stream_func())
        # await asyncio.gather(sender_task, stream_task)
        # await asyncio.gather(self.sender_func(), self.stream_func())

    async def on_message(self, websocket, message):
        print(f"Received message: {message}")
        try:
            message = json.loads(message)
            if message.get('status') == 'connected':
                self.connected = True

        except JSONDecodeError as not_json:
            pass

    async def on_error(self, websocket, error):
        print(f"Encountered error: {error}")

    async def on_close(self, websocket, close_status_code, close_msg):
        print(f"Connection closed! {close_status_code=}, {close_msg=}")

    async def on_open(self, websocket):
        print("Connection opened")
        await websocket.send(json.dumps({
            'prompt': 'Winged lions in the sky',
            'strength': 1.2
        }))

    async def input_generator(self, input_source, mode='image'):

        if mode == 'image':

            # load image (assuming load_image is async)
            img = load_image(str(input_source))

            for _ in range(self.send_fps * 10):
                yield img

    async def handle_messages(self, websocket):
        async for message in websocket:
            await self.on_message(websocket, message)

    async def send_frames(self, websocket):

        while not self.connected:
            await asyncio.sleep(0.1)

        try:
            async for img in self.input_generator(input_source=self.input_source, mode=self.input_mode):
                print('Sending frame')
                await websocket.send(self.frame_to_bytes(img))
                await asyncio.sleep(1 / self.send_fps)
        except websockets.ConnectionClosedError as ex:
            logging.exception('Connection closed')

    async def websocket_main_func(self):

        # Use websockets for asynchronous connection
        async with websockets.connect(f"ws://{self.host}:{self.port}/api/ws/{self.client_id}") as websocket:
            self.websocket = websockets.connect(f"ws://{self.host}:{self.port}/api/ws/{self.client_id}")

            # Run message handling and frame sending concurrently
            await asyncio.gather(
                self.send_frames(websocket),
                self.handle_messages(websocket),
                self.stream_func()
            )

    def frame_to_bytes(self, img):
        frame_data = io.BytesIO()
        img.save(frame_data, format="WEBP", quality=100, method=6)
        frame_data = frame_data.getvalue()
        return frame_data

    async def stream_func(self):

        while not self.connected:
            await asyncio.sleep(0.1)

        async with aiohttp.ClientSession() as session:
            async with session.get(url=f"http://{self.host}:{self.port}/api/stream/{self.client_id}") as response:
                if response.status == 200:
                    reader = aiohttp.MultipartReader.from_response(response)
                    while True:

                        part = await reader.next()
                        image_data = await part.read(decode=False)
                        print(f'Image data: {len(image_data)}')

                        if part is None:
                            break

                else:
                    print(f"Error getting stream: {response.status}")

    async def run(self):
        # await asyncio.gather(self.sender_func(), self.stream_func())
        await self.websocket_main_func()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Stream Tmunan Tester")
    parser.add_argument("--host", default="localhost", help="Host for Tmunan server")
    parser.add_argument("--port", type=int, default=8080, help="Port for Tmunan server")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input_image", '-i', type=Path,
                       default='/Users/himmelroman/Desktop/Bialik/me_canny.png',
                       help="Input image path")
    group.add_argument("--input_video", '-v', type=Path, help="Input video path")

    args = parser.parse_args()

    # create tester
    tester = TmunanStreamTester(args.host, args.port, input_mode='image', input_source=args.input_image)

    asyncio.run(tester.run())
