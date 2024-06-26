import shutil
from enum import Enum
from pathlib import Path

from tmunan.display.hls import HLS
from tmunan.imagine.image_generator import ImageGenerator, ImageGeneratorRemote
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

    def init_imagine(self, api_base_address=None, api_port=None):
        if not self.imagine:
            self.imagine: ImageGenerator = ImageGeneratorRemote(api_base_address, api_port)
            self.imagine.start()

    def init_listen(self):
        if not self.listen:
            self.listen = ASR()
            self.listen.start()

    def init_display(self, output_dir, image_height, image_width,
                     kf_period, kf_repeat, fps=12):

        # stop if running
        self.stop_display()

        # prepare hls dir
        hls_dir = output_dir / 'hls'
        shutil.rmtree(hls_dir, ignore_errors=True)
        Path.mkdir(hls_dir, parents=True, exist_ok=True)

        # HLS generator
        self.display = HLS(input_shape=(image_height, image_width),
                           kf_period=kf_period,
                           kf_repeat=kf_repeat,
                           output_fps=fps,
                           hls_path=output_dir / 'hls' / 'manifest.m3u8')
        self.display.start()

    def stop_display(self):
        if self.display:
            self.display.stop()
