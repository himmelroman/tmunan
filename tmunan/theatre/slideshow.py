import shutil
from pathlib import Path
from queue import Queue

import numpy as np

from tmunan.pack.hls_worker_process import HLSEncoderProcess
from tmunan.theatre.performance import Performance
from tmunan.tasks.image_script import ImageScript

from tmunan.imagine.lcm import LCM, load_image
from tmunan.api.pydantic_models import ImageSequence, ImageInstructions


class Slideshow(Performance):

    def __init__(self, app):
        super().__init__()

        # Host app
        self.app = app

        # Sequence info
        self.img_seq = None
        self.img_config = None

        # LCM
        self.lcm = LCM(txt2img_size='large')
        self.lcm.load()

        # Image management
        self.image_map = dict()
        self.image_queue = Queue()

        # Task
        self.image_script_task = ImageScript(self.lcm, self.cache_dir)

        # start HLS generation task
        self.hls_encoder = None

    def register_image(self, image_info):

        # register image into map
        self.image_map[image_info['image_id']] = image_info

        # put on queue
        image = load_image(image_info['image_path'])
        print('Pushing image to encoder')
        self.hls_encoder.push_input(np.array(image, dtype=np.uint8))

    def run(self, img_seq: ImageSequence, img_config: ImageInstructions, seq_id: str):

        # Sequence info
        seq_dir = Path(self.cache_dir) / f'seq_{seq_id}'
        self.img_seq = img_seq
        self.img_config = img_config

        # subscribe to events
        self.image_script_task.on_image_ready = self.register_image
        # self.image_script_task.on_sequence_finished = self.broadcast_message

        # start HLS encoder
        if (Path(seq_dir) / 'hls').exists():
            shutil.rmtree(str(Path(seq_dir) / 'hls'))
        self.hls_encoder = HLSEncoderProcess(Path(seq_dir) / 'hls' / 'manifest.m3u8', img_config.height, img_config.width, 12)
        self.hls_encoder.start()

        # run
        self.image_script_task.run_image_sequence(self.img_seq, self.img_config, seq_id)

        # stop hls
        self.hls_encoder.stop()
