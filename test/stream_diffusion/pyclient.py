import io
import json
import time
import uuid
import argparse
import threading

from pathlib import Path
from json import JSONDecodeError

import websocket
import requests
from requests_toolbelt.multipart import decoder

from tmunan.utils.image import load_image


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
        self.ws = websocket.WebSocketApp(
            url=f"ws://{self.host}:{self.port}/api/ws/{self.client_id}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close)

        # IO threads
        self.send_fps = 6
        self.sender_thread = None
        self.stream_thread = None

    def start_streaming(self):
        self.sender_thread = threading.Thread(target=self.sender_thread_func)
        self.stream_thread = threading.Thread(target=self.stream_thread_func)
        self.sender_thread.start()
        self.stream_thread.start()

    def on_message(self, ws, message):
        print(f"Received message: {message}")
        try:
            message = json.loads(message)
            if message.get('status') == 'connected':
                self.start_streaming()

        except JSONDecodeError as not_json:
            pass

    def on_error(self, ws, error):
        print(f"Encountered error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed! {close_status_code=}, {close_msg=}")

    def on_open(self, ws):
        print("Connection opened")
        ws.send_text(json.dumps({
            'prompt': 'Winged lions in the sky',
            'strength': 1.2
        }))

    def input_generator(self, input_source, mode='image'):

        if mode == 'image':

            # load image
            img = load_image(str(input_source))

            for _ in range(self.send_fps * 10):
                yield img

    def sender_thread_func(self):

        # iterate frames
        for img in self.input_generator(input_source=self.input_source, mode=self.input_mode):
            print('Sending frame')
            self.send_frame(img)
            time.sleep(1 / self.send_fps)

    def send_frame(self, img):

        frame_data = io.BytesIO()
        img.save(frame_data, format="WEBP", quality=100, method=6)
        frame_data = frame_data.getvalue()

        self.ws.send_bytes(frame_data)

    def stream_thread_func(self):

        time.sleep(2)
        stream_response = requests.get(url=f"http://{self.host}:{self.port}/api/stream/{self.client_id}", stream=True)
        multipart_data = decoder.MultipartDecoder.from_response(stream_response)
        for part in multipart_data.parts:
            print(part.headers['content-type'])
            print(part)


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

    tester.ws.run_forever()
