from celery.apps.worker import Worker

from tmunan.imagine.sd_lcm.lcm import LCM


class ImagineWorker(Worker):

    lcm = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # LCM
        self.lcm = LCM(model_size='small')

    def on_start(self):
        super().on_start()

        # Your one-time setup code here
        print("Worker starting, performing setup tasks...")
        self.lcm.load()
