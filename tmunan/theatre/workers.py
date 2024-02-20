from enum import Enum

from tmunan.common.exec import BackgroundExecutor

from tmunan.imagine.lcm import LCM
from tmunan.display.hls_worker_process import HLSEncoderProcess
from tmunan.listen.whisper_worker_process import WhisperBackgroundTask


class AppWorkers:

    class WorkerType(Enum):
        Imagine = 0
        Listen = 1
        Display = 2

    def __init__(self):

        self.imagine = None
        self.listen = None
        self.display = None

    def init_imagine(self, model_size='large'):
        if not self.imagine:
            self.imagine = LCM(txt2img_size=model_size)
            self.imagine.load()

    def init_listen(self):
        if not self.listen:
            self.listen = BackgroundExecutor(WhisperBackgroundTask, model_id='distil-medium')
            self.listen.start()

    def init_display(self, output_dir, image_height, image_width, fps=12):
        self.stop_display()
        self.display = HLSEncoderProcess(output_dir / 'hls' / 'manifest.m3u8', image_height, image_width, fps)
        self.display.start()

    def stop_display(self):
        if self.display:
            self.display.stop()