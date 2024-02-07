import time
from pathlib import Path

import numpy as np
from diffusers.utils import load_image

from tmunan.pack.hls_worker_process import HLSEncoderProcess

#
# def wait_and_stop():
#     print('waiting')
#     time.sleep(5)
#     print('stopping')
#     hls_enc.stop()


if __name__ == "__main__":

    hls_path = Path('/tmp/test/hls')
    hls_path.mkdir(parents=True, exist_ok=True)

    frames = Path('/Users/himmelroman/.cache/theatre/seq_8ceec6c1/').glob('*.png')
    hlser = HLSEncoderProcess('/tmp/test/hls', 512, 512, 12)
    hlser.start()
    time.sleep(3)

    for f in frames:
        image = load_image(str(f))
        image_np = np.array(image, dtype=np.uint8)

        hlser.push_input(image_np)
        # time.sleep(1)

    # time.sleep(1)

    # time.sleep(1)
    print('Stopping test')
    hlser.stop()
