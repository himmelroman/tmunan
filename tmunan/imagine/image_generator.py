from abc import ABC
from concurrent.futures import ThreadPoolExecutor

import logging
import requests
from PIL import Image

from tmunan.api.pydantic_models import Prompt, ImageInstructions, BaseImage
from tmunan.common.event import Event
from tmunan.common.exec import BackgroundExecutor
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.sd_lcm.lcm_bg_task import LCMBackgroundTask, TaskType


class ImageGenerator(ABC):

    def __init__(self):

        # events
        self.on_image_ready = Event()
        self.on_startup = Event()
        self.on_shutdown = Event()

    def start(self):
        pass

    def stop(self):
        pass

    def txt2img(self, **kwargs):
        pass

    def img2img(self, **kwargs):
        pass

    @staticmethod
    def get_random_seed():
        return LCM.get_random_seed()


class ImageGeneratorLocal(ImageGenerator):

    def __init__(self, model_size):
        super().__init__()

        # executor
        self.lcm_executor = BackgroundExecutor(LCMBackgroundTask, model_size=model_size)

    def start(self):
        self.lcm_executor.start()
        self.lcm_executor.on_output_ready += lambda res: self.on_image_ready.notify(res)
        self.lcm_executor.on_worker_ready += self.on_startup.notify
        self.lcm_executor.on_exit += self.on_shutdown.notify
        self.lcm_executor.on_error += lambda ex: print(ex)

    def stop(self):
        self.lcm_executor.stop()

    def txt2img(self, **kwargs):
        kwargs['task'] = TaskType.Text2Image.value
        self.lcm_executor.push_input(kwargs)

    def img2img(self, **kwargs):
        kwargs['task'] = TaskType.Image2Image.value
        self.lcm_executor.push_input(kwargs)


class ImageGeneratorRemote(ImageGenerator):

    def __init__(self, api_address, api_port):
        super().__init__()

        # http
        self.imagine_address = api_address
        self.imagine_port = api_port
        self.http_exec_pool = ThreadPoolExecutor(max_workers=1)

    def start(self):
        self.on_startup.fire()

    def stop(self):
        self.on_shutdown.fire()

    def txt2img(self, prompt: str, **img_config):

        # prepare request
        url = f'{self.imagine_address}:{self.imagine_port}/api/imagine/txt2img'
        data = {
            'prompt': Prompt(text=prompt).model_dump(),
            'img_config': ImageInstructions(**img_config).model_dump()
        }

        # submit work
        self.http_exec_pool.submit(self._download_image, url, data)

    def img2img(self, prompt: str, image_url: str, **img_config):

        # prepare request
        url = f'{self.imagine_address}:{self.imagine_port}/api/imagine/img2img'
        data = {
            'prompt': Prompt(text=prompt).model_dump(),
            'base_image': BaseImage(image_url=image_url).model_dump(),
            'img_config': ImageInstructions(**img_config).model_dump()
        }

        # submit work
        self.http_exec_pool.submit(self._download_image, url, data)

    def _download_image(self, url: str, data: dict):

        try:
            # post request
            resp = requests.post(
                url=url,
                json=data,
                timeout=60
            )
            resp.raise_for_status()

            # get image details from response
            resp_image = resp.json()
            image_url = resp_image.get('image_url')

            # verify image is available
            if image_url:

                # download image and load into PIL.Image object
                img = Image.open(requests.get(image_url, stream=True).raw)

                # fire result event
                self.on_image_ready.fire(image_url, img)

        except Exception as ex:
            logging.exception('Error requesting for image!')
