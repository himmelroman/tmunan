from concurrent.futures import ThreadPoolExecutor

import logging
import requests
from PIL import Image

from tmunan.api.pydantic_models import Prompt, ImageInstructions
from tmunan.common.event import Event
from tmunan.common.exec import BackgroundExecutor
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.sd_lcm.lcm_bg_task import LCMBackgroundTask, TaskType


class ImageGenerator:

    def __init__(self, model_size):

        # executor
        self.lcm_executor = BackgroundExecutor(LCMBackgroundTask, model_size=model_size)

        # events
        self.on_image_ready = Event()
        self.on_startup = Event()
        self.on_shutdown = Event()

    def start(self):
        self.lcm_executor.start()
        self.lcm_executor.on_output_ready += lambda res: self.on_image_ready.notify(res)
        self.lcm_executor.on_worker_ready += self.on_startup.notify
        self.lcm_executor.on_exit += self.on_shutdown.notify
        self.lcm_executor.on_error += lambda ex: print(ex)

    def stop(self):
        self.lcm_executor.stop()

    def request_image(self, **kwargs):
        self.lcm_executor.push_input(kwargs)

    @staticmethod
    def get_random_seed():
        return LCM.get_random_seed()


class RemoteImageGenerator:

    def __init__(self, api_address, api_port):

        # http
        self.imagine_address = api_address
        self.imagine_port = api_port
        self.exec_pool = ThreadPoolExecutor(max_workers=1)

        # events
        self.on_image_ready = Event()
        self.on_startup = Event()
        self.on_shutdown = Event()

    def start(self):
        pass

    def stop(self):
        pass

    def txt2img(self, prompt: str, **img_config):

        # prepare request
        url = f'{self.imagine_address}:{self.imagine_port}/api/imagine/txt2img'
        data = {
            'prompt': Prompt(text=prompt).model_dump(),
            'img_config': ImageInstructions(**img_config).model_dump()
        }

        # submit work
        self.exec_pool.submit(self._download_image, url, data)

    def img2img(self, prompt: str, image_id: str, **img_config):

        # prepare request
        url = f'{self.imagine_address}/api/imagine/img2img'
        params = {
            'image_id': image_id
        }
        data = {
            'prompt': Prompt(text=prompt).model_dump_json(),
            'img_config': ImageInstructions(**img_config).model_dump_json()
        }

        # submit work
        self.exec_pool.submit(self._download_image, url, data, params)

    def _download_image(self, url: str, data: dict, params=None):

        try:
            # post request
            resp = requests.post(
                url=url,
                params=params,
                json=data,
                timeout=30
            )
            resp.raise_for_status()

            # get image url from response
            resp_image = resp.json()
            image_url = resp_image.get('image_url')
            if image_url:
                # download image and load into PIL.Image object
                img = Image.open(requests.get(image_url, stream=True).raw)

                # fire result event
                self.on_image_ready.fire(img)

        except Exception as ex:
            logging.exception('Error requesting for image!')

    @staticmethod
    def get_random_seed():
        return LCM.get_random_seed()
