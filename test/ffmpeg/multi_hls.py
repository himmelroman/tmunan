import time
import argparse

import ffmpeg
import numpy as np
from pathlib import Path
from diffusers.utils import load_image

from tmunan.display.hls import HLS


def append_to_hls(images, do_interpolation, input_fps, output_fps):

    input_settings = {
        "format": "rawvideo",
        "pix_fmt": "rgb24",
        "s": "768x768",
        "framerate": f"1/{input_fps}",
    }
    output_settings = {
        "g": output_fps,
        "sc_threshold": 0,
        "format": "hls",
        "hls_time": 1,
        "hls_list_size": 2 * 60 / 2,  # 10 minutes keep
        "hls_flags": "independent_segments+append_list",  # "split_by_time", "delete_segments"
        "flush_packets": 1,
        "vcodec": "libx264",
        "r": output_fps,
        # "preset": "veryfast",
        "video_bitrate": "6M",
        "maxrate": "6M",
        "bufsize": "6M",
    }

    if do_interpolation:
        ffmpeg_process = (
            ffmpeg
            .input("pipe:", **input_settings)
            .filter("minterpolate", fps=output_fps, mi_mode="mci", mc_mode="aobmc", me_mode="bidir", vsbmc=0.9)
            # .filter("minterpolate", fps=output_fps, mi_mode="blend")
            # .filter("cas", strength=0.8)
            .output('/tmp/multi_hls/hls.m3u8', **output_settings)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        ffmpeg_process = (
            ffmpeg
            .input("pipe:", **input_settings)
            # .filter("tblend")
            # .filter("minterpolate", fps=output_fps, mi_mode="blend")
            # .filter("cas", strength=0.8)
            .output('/tmp/multi_hls/hls.m3u8', **output_settings)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    for img_path in images:
        print(f'Pushing image: {img_path=}')
        image = load_image(str(img_path))
        image_np = np.array(image, dtype=np.uint8)
        ffmpeg_process.stdin.write(image_np.tobytes())
        # time.sleep(1)

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


if __name__ == "__main__":

    input_fps = 3
    output_fps = 25

    image_paths = Path('/Users/himmelroman/.cache/theatre/script_093ea0ec/seq_1d8f9b2d').glob('*.png')
    image_paths = sorted(image_paths)

    for i, image_path in enumerate(image_paths):
        # append_to_hls([image_paths[i]], do_interpolation=False, input_fps=2, output_fps=25)
        if len(image_paths) > i + 1:
            append_to_hls([image_paths[i], image_paths[i + 1]], do_interpolation=True, input_fps=2, output_fps=25)
        # if len(image_paths) > i + 1:
        #     append_to_hls([image_paths[i], image_paths[i+1]], do_interpolation=True, input_fps=1, output_fps=25)
