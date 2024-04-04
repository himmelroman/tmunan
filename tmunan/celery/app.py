import numpy as np
from celery import Celery, Task, signals
from celery.app import task

from tmunan.common.log import get_logger
from tmunan.imagine.sd_lcm.lcm import LCM

# from tmunan.celery.workers import ImagineWorker
# from tmunan.celery.tasks import txt2img, img2img

# logging
log = get_logger('TmunanImagineCelery')


# celery app
celery_app = Celery('imagine', backend='rpc://', broker='pyamqp://guest@localhost//')
celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'pickle'
celery_app.conf.accept_content = ['pickle']


class LCMTask(Task):
    _lcm = None

    def __init__(self):
        signals.worker_before_create_process.connect(self.load)

    def load(self, **kwargs):
        self._lcm = LCM(model_size='small')
        self._lcm.load()

    @property
    def lcm(self):
        return self._lcm


@celery_app.task(base=LCMTask, bind=True)
def txt2img(self: task,
            prompt: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 4,
            guidance_scale: float = 0.0,
            seed: int = 0,
            randomize_seed: bool = False):

    # generate image
    images = self.lcm.txt2img(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        randomize_seed=randomize_seed
    )

    # select first
    result_image = images[0]

    # convert to nparray
    # np_image = np.array(result_image, dtype=np.uint8)

    return result_image


@celery_app.task(base=LCMTask, bind=True)
def img2img(self: task,
            image_url: str,
            prompt: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 4,
            guidance_scale: float = 1.0,
            strength: float = 0.6):

    # generate image
    images = self.lcm.img2img(
        image_url=image_url,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )

    # select first
    result_image = images[0]

    # convert to nparray
    # np_image = np.array(result_image, dtype=np.uint8)

    return result_image


# if __name__ == '__main__':
#     custom_worker = ImagineWorker(app=celery_app)
#     custom_worker.start()
