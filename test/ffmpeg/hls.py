import time
import argparse
import numpy as np
from pathlib import Path
from diffusers.utils import load_image

from tmunan.display.hls import HLS


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=False, type=Path,
                        default='/Users/himmelroman/.cache/theatre/script_093ea0ec/seq_1d8f9b2d')
    parser.add_argument('-o', '--hls_output_path', required=False, type=Path,
                        default='/tmp/multi_hls/hls.m3u8')
    args = parser.parse_args()

    in_fps = 2
    out_fps = 12
    images = Path(args.input_dir).glob('*.png')
    hlser = HLS(input_shape=(768, 768), input_fps=in_fps, output_fps=out_fps,
                hls_path=Path(args.hls_output_path))
    hlser.start()
    time.sleep(5)

    for img in sorted(images):
        for _ in range(3):
            print(f'Pushing image: {img=}')
            image = load_image(str(img))
            image_np = np.array(image, dtype=np.uint8)
            hlser.push_image(image_np)
            # time.sleep(1)

    time.sleep(20)
    print('Stopping test')
    hlser.stop()
