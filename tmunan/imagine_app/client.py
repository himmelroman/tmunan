import io
import time
import queue
import threading

import requests
from PIL import Image

from tmunan.utils.event import Event
from tmunan.utils.log import get_logger
from tmunan.utils.image import pil_to_bytes, pil_to_frame
from tmunan.common.models import ImageParameters


class ImagineClient:

    def __init__(self, host: str, port: int, secure: bool = False):

        # service address
        scheme = 'https' if secure else 'http'
        self.service_url = f'{scheme}://{host}:{port}/api/img2img'

        # io
        self.input_queue: queue.Queue | None = None
        self.on_image_ready = Event()

        # worker thread
        self._worker_thread = threading.Thread(target=self._thread_func)

        # env
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f'Initialized ImagineClient! {self.service_url=}')

    def watch_queue(self, input_queue):

        self.input_queue = input_queue
        self._worker_thread.start()

    def _thread_func(self):

        while True:
            try:
                item = self.input_queue.get(timeout=0.01)
                if item:

                    # take time
                    req_time = item.pop('timestamp', None)
                    self.logger.info(f'Processing request from: {req_time}, which arrived {time.time() - req_time} ago')

                    # post image
                    input_image = item.pop('image')
                    self.logger.info(f'Sending request with params: {item}')
                    new_image = self.post_image(input_image, item)
                    self.logger.info(f'Finished processing request at: {req_time}, which arrived {time.time() - req_time} ago')

                    # output
                    frame = pil_to_frame(new_image, format='webp')
                    self.on_image_ready.notify(req_time, frame)

            except queue.Empty:
                pass

            except requests.exceptions.ConnectionError:
                self.logger.exception(f'Error connecting to server at {self.service_url}')

    def post_image(self, image: Image, params: ImageParameters) -> Image:

        # prepare post
        files = {
            'image': pil_to_bytes(image)
        }
        # params = params.model_dump()

        # execute post
        response = requests.post(
            url=self.service_url,
            files=files,
            params=params
        )
        response.raise_for_status()

        # load image from response
        return Image.open(io.BytesIO(response.content))
