import enum
import time
from queue import Queue

import numpy as np
from typing import Any
from pathlib import Path

import ffmpeg

from tmunan.display.utils import duplicate_frames


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
            "hls_flags": "independent_segments",    # "split_by_time",  # "delete_segments",  # remove outdated segments from disk
            "flush_packets": 1,
            **preset.value,
            **hls_kwargs,
        }
        self.fps = input_fps

    def __enter__(self) -> "HLSEncoder":
        self.proc = (
            ffmpeg.input("pipe:", **self.inp_settings)
            # .filter("fade", d=1, t="in", alpha=1)
            .output(str(self.out_path), **self.enc_settings)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return self

    def __exit__(self, type, value, traceback):
        self.proc.stdin.close()
        self.proc.wait()
        self.proc = None

    def __call__(self, rgb24: np.ndarray[np.uint8, Any]):
        self.proc.stdin.write(rgb24.tobytes())

    def stop(self):
        print('HLS Encoder: Stopping!')
        if self.frame_queue:
            print('HLS Encoder: Putting None')
            self.frame_queue.put(None)

    def run(self, frame_queue: Queue):

        print(f'Running: {self.fps=}')

        # save input queue
        self.frame_queue = frame_queue

        # context
        with self:

            # loop
            end_reached = False
            last_frame = None
            while not end_reached:

                # let images aggregate
                time.sleep(1)

                # get all frames from queue
                input_frames = self.get_all(frame_queue)
                # print(f'HLS: Got {len(input_frames)} frames from queue')

                # reuse last frame if no new frames arrived
                if len(input_frames) == 0 and last_frame is not None:
                    input_frames = [last_frame]

                # check if we have any input to push
                if len(input_frames) > 0:

                    # check if death-pill received
                    if input_frames[-1] is None:

                        # mark
                        end_reached = True

                        # take all real frames for last round
                        if len(input_frames) > 1:
                            input_frames = input_frames[:-1]

                    filled_frames = duplicate_frames(input_frames, target_fps=self.fps)
                    # print(f'HLS: Got {len(filled_frames)} after interpolation')

                    # push frames to ffmpeg
                    for frame in filled_frames:
                        self(frame)
                        last_frame = frame

        print('HLS Exiting')

    @staticmethod
    def get_all(queue):
        all_items = []
        while not queue.empty():
            all_items.append(queue.get_nowait())

        return all_items
