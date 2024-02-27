from enum import Enum

from tmunan.display.hls import HLS
from tmunan.imagine.txt2img import Txt2Img
from tmunan.listen.asr import ASR


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
            self.imagine = Txt2Img(model_id=model_size)
            self.imagine.start()

    def init_listen(self):
        if not self.listen:
            self.listen = ASR()
            self.listen.start()

    def init_display(self, output_dir, image_height, image_width, fps=12):
        self.stop_display()
        self.display = HLS(input_shape=(image_height, image_width),
                           input_fps=1,
                           output_fps=fps,
                           hls_path=output_dir / 'hls' / 'manifest.m3u8')
        self.display.start()

    def stop_display(self):
        if self.display:
            self.display.stop()
