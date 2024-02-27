import enum
import numpy as np
from typing import Any
from pathlib import Path

import ffmpeg

from tmunan.common.exec import BackgroundTask
from tmunan.common.log import get_logger


class HLSPresets(enum.Enum):
    # For preset settings
    # https://obsproject.com/blog/streaming-with-x264#:~:text=x264%20has%20several%20CPU%20presets,%2C%20slower%2C%20veryslow%2C%20placebo

    DEFAULT_CPU = {
        "vcodec": "libx264",
        "preset": "veryfast",
        "video_bitrate": "6M",
        "maxrate": "6M",
        "bufsize": "6M",
    }
    DEFAULT_CUDA = {
        "vcodec": "h264_nvenc",
        "preset": "p3",  # https://gist.github.com/nico-lab/e1ba48c33bf2c7e1d9ffdd9c1b8d0493
        "tune": "ll",
        "video_bitrate": "6M",
        "maxrate": "6M",
        "bufsize": "6M",
    }
    DEFAULT_MPS = {
        "vcodec": "h264_videotoolbox",
        "video_bitrate": "4M",
        # "maxrate": "6M",
        # "bufsize": "6M",
    }


class Image2HLSBackgroundTask(BackgroundTask):

    def __init__(self,
                 input_shape: tuple[int, int],
                 input_fps: int,
                 output_fps: int,
                 hls_path: Path,
                 preset: HLSPresets = HLSPresets.DEFAULT_CPU,
                 **hls_kwargs):
        super().__init__()

        # save arguments
        self.input_shape = input_shape
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.preset = preset
        self.hls_kwargs = hls_kwargs
        self.hls_path = hls_path

        # Prepare HLS dir
        self.out_path = Path(self.hls_path)
        if self.out_path.is_dir():
            self.out_path = self.out_path / 'hls.m3u8'
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        # FFMPEG arguments
        self.ffmpeg_process = None
        self.input_settings = {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
            "s": f"{self.input_shape[1]}x{self.input_shape[0]}",
            "framerate": f"1/{self.input_fps}",
        }
        self.output_settings = {
            "g": self.output_fps,
            "sc_threshold": 0,
            "format": "hls",
            "hls_time": 1,
            "hls_list_size": 2 * 60 / 2,             # 10 minutes keep
            "hls_flags": "independent_segments",     # "split_by_time", "delete_segments"
            "flush_packets": 1,
            **preset.value,
            **hls_kwargs,
        }

        # env
        self.logger = None

    def setup(self):

        # setup logger
        self.logger = get_logger(self.__class__.__name__)

        # ffmpeg process
        self.logger.info(f"Starting ffmpeg: {self.input_settings=}, {self.output_settings=}")
        self.ffmpeg_process = (
            ffmpeg.input("pipe:", **self.input_settings)
            .filter("minterpolate", fps=self.output_fps, mi_mode="mci", mc_mode="aobmc", me_mode="bidir", vsbmc=0.9)
            # .filter("cas", strength=0.8)
            .output(str(self.out_path), **self.output_settings)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def cleanup(self):
        self.logger.info('Cleanup started...')
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
        self.ffmpeg_process = None
        self.logger.info('Cleanup finished.')

    def exec(self, image: np.ndarray[np.uint8, Any]):

        # push image to ffmpeg's stdin
        self.logger.info('Pushing image')
        self.ffmpeg_process.stdin.write(image.tobytes())
