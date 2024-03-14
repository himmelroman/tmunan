from enum import Enum

from tmunan.display.hls import HLS
from tmunan.imagine.image_generator import ImageGenerator
from tmunan.listen.asr import ASR
from tmunan.read.text_generator import TextGenerator


class AppWorkers:

    class WorkerType(Enum):
        Imagine = 0
        Listen = 1
        Display = 2
        Text = 3

    def __init__(self):

        self.imagine = None
        self.listen = None
        self.display = None
        self.read = None

    def init_read(self):
        if not self.read:
            self.read = TextGenerator()
            self.read.start()

    def init_imagine(self, model_size='small'):
        if not self.imagine:
            self.imagine = ImageGenerator(model_size=model_size)
            self.imagine.start()

    def init_listen(self):
        if not self.listen:
            self.listen = ASR()
            self.listen.start()

    def init_display(self, output_dir, image_height, image_width,
                     kf_duration, kf_repeat, fps=12):
        self.stop_display()
        self.display = HLS(input_shape=(image_height, image_width),
                           kf_duration=kf_duration,
                           kf_repeat=kf_repeat,
                           output_fps=fps,
                           hls_path=output_dir / 'hls' / 'manifest.m3u8')
        self.display.start()

    def stop_display(self):
        if self.display:
            self.display.stop()
