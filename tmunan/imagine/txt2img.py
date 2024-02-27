from tmunan.common.event import Event
from tmunan.common.exec import BackgroundExecutor
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.sd_lcm.lcm_bg_task import LCMBackgroundTask


class Txt2Img:

    def __init__(self, model_id):
        self.lcm_executor = BackgroundExecutor(LCMBackgroundTask, model_id=model_id)
        self.on_image_ready = Event()
        self.on_startup = Event()
        self.on_shutdown = Event()

    def start(self):
        self.lcm_executor.start()
        self.lcm_executor.on_output_ready += lambda res: self.on_image_ready.notify(res)
        self.lcm_executor.on_worker_ready += self.on_startup.notify
        self.lcm_executor.on_exit += self.on_shutdown.notify

    def stop(self):
        self.lcm_executor.stop()

    def request_txt2img(self, **kwargs):
        print(f'Pushing to self.lcm_executor: {kwargs=}')
        self.lcm_executor.push_input(kwargs)

    @staticmethod
    def get_random_seed():
        return LCM.get_random_seed()
