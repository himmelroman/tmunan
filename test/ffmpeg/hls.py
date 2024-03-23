import time
import argparse
import numpy as np
from pathlib import Path
from diffusers.utils import load_image

from tmunan.display.hls import HLS


def dir2hls(in_fps, out_fps, key_frame_repeat, input_dir, output_dir):

    images = Path(input_dir).glob('*.png')
    hlser = HLS(input_shape=(512, 512), kf_period=in_fps, kf_repeat=key_frame_repeat, output_fps=out_fps,
                hls_path=Path(output_dir))
    hlser.start()

    time.sleep(5)

    for img in sorted(images):
        for _ in range(5):
            print(f'Pushing image: {img=}')
            image = load_image(str(img))
            image_np = np.array(image, dtype=np.uint8)
            hlser.push_image(image_np)
            time.sleep(1)

    time.sleep(15)
    print('Stopping test')
    hlser.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=False, type=Path,
                        default='/Users/himmelroman/.cache/theatre/script_d92e9879/seq_1c6678a8')
    parser.add_argument('-o', '--hls_output_path', required=False, type=Path,
                        default='/tmp/multi_hls')
    args = parser.parse_args()

    # dir2hls(in_fps=2, out_fps=12, input_dir=args.input_dir, output_dir=args.hls_output_path)

    dir2hls(in_fps=3, key_frame_repeat=2, out_fps=24,
            input_dir=args.input_dir,
            output_dir=args.hls_output_path / 'hls.m3u8')
