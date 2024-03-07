from enum import Enum

from tmunan.common.log import get_logger
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.common.exec import BackgroundTask


class TaskType(str, Enum):
    Text2Image = 'txt2img'
    Image2Image = 'img2img'


class LCMBackgroundTask(BackgroundTask):

    def __init__(self, model_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # models
        self.lcm = None
        self.model_size = model_size

        # arguments
        self._args = args
        self._kwargs = kwargs
        self._task_map = None

        # env
        self.logger = None

    def setup(self):

        self.logger = get_logger(self.__class__.__name__)

        try:
            # load model
            self.lcm = LCM(txt2img_size=self.model_size, img2img_size=self.model_size)
            self.lcm.load()

            # init task map
            self._task_map = {
                TaskType.Text2Image: self.lcm.txt2img,
                TaskType.Image2Image: self.lcm.img2img
            }

        except Exception as ex:
            self.logger.exception(f'Error in {self.__class__.__name__} setup phase!')

    def exec(self, img_gen_args):

        try:

            # parse task
            task_type = img_gen_args.pop('task')

            # run task
            images = self._task_map[task_type](**img_gen_args)
            return images[0]

        except Exception as ex:
            self.logger.exception(f'Error in {self.__class__.__name__} exec!')
