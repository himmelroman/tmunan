import time
from pathlib import Path

import numpy as np
from diffusers.utils import load_image

from tmunan.display.hls_worker_process import HLSEncoderProcess

#
# def wait_and_stop():
#     print('waiting')
#     time.sleep(5)
#     print('stopping')
#     hls_enc.stop()


if __name__ == "__main__":

    hls_path = Path('/tmp/test/hls')
    hls_path.mkdir(parents=True, exist_ok=True)

    fps = 25
    images = Path('/Users/himmelroman/projects/speechualizer/tmunan/test/ffmpeg/blend/').glob('*.png')
    hlser = HLSEncoderProcess('/Users/himmelroman/projects/speechualizer/tmunan/test/ffmpeg/blend/hls/playlist.m3u8', 768, 768, fps)
    hlser.start()
    time.sleep(2)

    for _ in range(5):
        for img in images:
            image = load_image(str(img))
            image_np = np.array(image, dtype=np.uint8)
            hlser.push_input(image_np)
            time.sleep(1)

    print('Stopping test')
    hlser.stop()
