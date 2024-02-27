import time
import numpy as np
from pathlib import Path
from diffusers.utils import load_image

from tmunan.display.hls import HLS


if __name__ == "__main__":

    hls_path = Path('/tmp/test/hls')
    hls_path.mkdir(parents=True, exist_ok=True)

    in_fps = 3
    out_fps = 12
    images = Path('/Users/himmelroman/projects/speechualizer/tmunan/test/ffmpeg/blend/sheep_sequence').glob('*.png')
    hlser = HLS(input_shape=(768, 768), input_fps=in_fps, output_fps=out_fps,
                hls_path=Path('/Users/himmelroman/projects/speechualizer/tmunan/test/ffmpeg/blend/hls/playlist.m3u8'))
    hlser.start()
    time.sleep(5)

    for _ in range(5):
        for img in sorted(images):
            print(f'Pushing image: {img=}')
            image = load_image(str(img))
            image_np = np.array(image, dtype=np.uint8)
            hlser.push_image(image_np)
            # time.sleep(1)

    time.sleep(20)
    print('Stopping test')
    hlser.stop()
