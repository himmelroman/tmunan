from tmunan.common.log import get_logger
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.common.exec import BackgroundTask


class LCMBackgroundTask(BackgroundTask):

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # models
        self.lcm = None
        self.model_id = model_id

        # arguments
        self._args = args
        self._kwargs = kwargs

        # env
        self.logger = None

    def setup(self):

        self.logger = get_logger(self.__class__.__name__)

        try:
            # load model
            self.lcm = LCM(txt2img_size=self.model_id)
            self.lcm.load()

        except Exception as ex:
            self.logger.exception()

    def exec(self, txt2img_args):

        try:
            print(f'Running self.lcm.txt2img with: {txt2img_args=}')
            images = self.lcm.txt2img(**txt2img_args)
            return images[0]

        except Exception as ex:
            self.logger.exception()
