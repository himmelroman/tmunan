import io
import time
import queue
import threading

import requests
from PIL import Image
from requests.exceptions import Timeout, HTTPError, RequestException

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
        self._worker_thread = threading.Thread(target=self._thread_func, daemon=True)
        self._stop_requested = False

        # env
        self.logger = get_logger(self.__class__.__name__)

    def watch_queue(self, input_queue):

        self.input_queue = input_queue
        self._worker_thread.start()

    def stop(self):
        self._stop_requested = True

        if self._worker_thread.is_alive():
            self._worker_thread.join()

    def _thread_func(self):

        while not self._stop_requested:
            try:
                item = self.input_queue.get(timeout=0.01)
                if item:

                    # take time
                    req_id = item.pop('req_id', None)
                    req_time = item.pop('timestamp', None)
                    self.logger.info(f"ReqTrace - Sending to Imagine: {req_id} at {time.time()}, delay: {time.time() - req_time}")

                    # post image
                    input_image = item.pop('image')
                    new_image = self.post_image(input_image, item)
                    self.logger.info(f"ReqTrace - Response from Imagine: {req_id} at {time.time()}, delay: {time.time() - req_time}")

                    # output
                    frame = pil_to_frame(new_image, format='jpeg', quality=95)
                    self.logger.info(f"ReqTrace - Frame ready: {req_id} at {time.time()}, delay: {time.time() - req_time}")
                    self.on_image_ready.notify(req_id, req_time, frame)

            except queue.Empty:
                pass

            except (ConnectionError, Timeout, HTTPError, RequestException) as ex:
                self.logger.exception('Request error')
                pass

    def post_image(self, image: Image, params: ImageParameters) -> Image:

        # loopback when strength is zero
        if params.strength < 1 or params.strength > 2.95:
            return image

        # prepare post
        files = {
            'image': pil_to_bytes(image, format='jpeg', quality=95)
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
