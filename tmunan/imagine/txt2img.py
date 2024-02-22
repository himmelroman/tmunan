from tmunan.common.event import Event
from tmunan.common.exec import BackgroundExecutor
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.sd_lcm.lcm_bg_task import LCMBackgroundTask


class Txt2Img:

    def __init__(self, model_id):
        self.lcm_executor = BackgroundExecutor(LCMBackgroundTask, model_id=model_id)
        self.on_image_ready = Event()

    def start(self):
        self.lcm_executor.start()
        self.lcm_executor.on_output_ready += lambda res: self.on_image_ready.notify(res)

    def txt2img(self, **kwargs):
        self.lcm_executor.push_input(kwargs)

    def get_random_seed(self):
        return LCM.get_random_seed()
