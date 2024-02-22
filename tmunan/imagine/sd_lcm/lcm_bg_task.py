from tmunan.common.exec import BackgroundTask
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.listen.distil_whisper.whisper import DistilWhisper


class LCMBackgroundTask(BackgroundTask):

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = args
        self._kwargs = kwargs

        self.model_id = model_id
        self.lcm = None

    def setup(self):

        try:
            # load model
            self.lcm = LCM(txt2img_size=self.model_id)
            self.lcm.load()

        except Exception as e:
            print(e)

    def exec(self, txt2img_args):
        images = self.lcm.txt2img(**txt2img_args)
        return images[0]
