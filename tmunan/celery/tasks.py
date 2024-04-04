from celery import Task
from celery.app import task

from tmunan.celery.app import celery_app
from tmunan.celery.workers import ImagineWorker
from tmunan.imagine.sd_lcm.lcm import LCM


class LCMTask(Task):
    _lcm = None

    @property
    def lcm(self):
        if self._lcm is None:
            self._lcm = LCM(model_size='small')
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
    images = ImagineWorker.lcm.img2img(
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
