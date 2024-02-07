import enum
import time
from queue import Empty

import numpy as np
from typing import Any
from pathlib import Path

import ffmpeg


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
    DEFAULT_GPU = {
        "vcodec": "h264_nvenc",
        "preset": "p3",  # https://gist.github.com/nico-lab/e1ba48c33bf2c7e1d9ffdd9c1b8d0493
        "tune": "ll",
        "video_bitrate": "6M",
        "maxrate": "6M",
        "bufsize": "6M",
    }


class HLSEncoder:
    def __init__(
        self,
        out_path: Path,
        shape: tuple[int, int] = (1080, 1920),
        input_fps: int = 30,
        preset: HLSPresets = HLSPresets.DEFAULT_CPU,
        **hls_kwargs,
    ) -> None:

        # Input image
        self.shape = shape

        # Output HLS dir
        self.out_path = Path(out_path)
        if self.out_path.is_dir():
            self.out_path = self.out_path / 'hls.m3u8'
        # if self.out_path.is_file() and self.out_path.suffix != '.m3u8':
        #     self.out_path = self.out_path / 'hls.m3u8'

        print(self.out_path.parent)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Output path for HLS: {self.out_path}')

        # Input queue
        self.frame_queue = None

        # FFMPEG Config
        self.inp_settings = {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
            "s": f"{shape[1]}x{shape[0]}",
            "r": input_fps
        }
        self.enc_settings = {
            "format": "hls",
            "pix_fmt": "yuv420p",
            "hls_time": 1,
            "hls_list_size": 2 * 60 / 2,  # 10 minutes keep
            "hls_flags": "independent_segments+split_by_time",  # "delete_segments",  # remove outdated segments from disk
            "flush_packets": 1,
            **preset.value,
            **hls_kwargs,
        }
        self.fps = input_fps

    def __enter__(self) -> "HLSEncoder":
        self.proc = (
            ffmpeg.input("pipe:", **self.inp_settings)
            .output(str(self.out_path), **self.enc_settings)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return self

    def __exit__(self, type, value, traceback):
        self.proc.stdin.close()
        self.proc = None

    def __call__(self, rgb24: np.ndarray[np.uint8, Any]):
        self.proc.stdin.write(rgb24.tobytes())

    def stop(self):
        print('HLS Encoder: Stopping!')
        if self.frame_queue:
            print('HLS Encoder: Putting None')
            self.frame_queue.put(None)

    def run(self, frame_queue):

        print(f'Running: {self.fps=}')

        # save
        self.frame_queue = frame_queue

        # context
        with self:

            # loop
            while True:

                try:
                    # get next image
                    img = frame_queue.get(timeout=1)
                    print(f'HLS Encoder: Got image from queue')

                    # push to ffmpeg
                    if img is None:
                        print('Breaking')
                        break

                    print('HLS Encoder: Pushing image into ffmpeg')
                    for _ in range(self.fps):
                        self(img)

                except Empty:
                    print('HLS Encoder: Queue is empty...')
                    continue

        print('HLS Existing')