from tmunan.common.event import Event
from tmunan.common.exec import BackgroundExecutor
from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.sd_lcm.lcm_bg_task import LCMBackgroundTask


class ImageGenerator:

    def __init__(self, model_size):
        self.lcm_executor = BackgroundExecutor(LCMBackgroundTask, model_size=model_size)
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
