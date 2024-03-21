from pathlib import Path

from tmunan.common.exec import BackgroundExecutor
from tmunan.display.hls_bg_task import Image2HLSBackgroundTask


class HLS:

    def __init__(self,
                 input_shape: tuple[int, int],
                 kf_period: int,
                 kf_repeat: int,
                 output_fps: int,
                 hls_path: Path):

        # create HLS executor
        self.hls_executor = BackgroundExecutor(Image2HLSBackgroundTask,
                                               input_shape=input_shape,
                                               kf_period=kf_period,
                                               kf_repeat=kf_repeat,
                                               output_fps=output_fps,
                                               hls_path=hls_path)

    def start(self):
        self.hls_executor.start()

    def stop(self):
        self.hls_executor.stop()

    def push_image(self, image):
        self.hls_executor.push_input(image)
