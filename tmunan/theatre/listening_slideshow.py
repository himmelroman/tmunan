import numpy as np

from tmunan.theatre.performance import Performance
from tmunan.tasks.image_script import ImageScript

from tmunan.imagine.sd_lcm.lcm import load_image
from tmunan.api.pydantic_models import ImageSequence, ImageInstructions


class Slideshow(Performance):

    def __init__(self, app):
        super().__init__()

        # Host app
        self.app = app

        # Imagine & display
        self.imagine = app.workers.imagine
        self.display = app.workers.display
        self.listen = app.workers.listen

        # Sequence info
        self.img_seq = None
        self.img_config = None

        # Task
        self.image_script_task = ImageScript(self.imagine, self.listen, self.cache_dir)

    def display_image(self, image_url, image_info):

        # put on queue
        image = load_image(image_info['image_path'])
        self.display.push_image(np.array(image, dtype=np.uint8))

    def run(self, img_seq: ImageSequence, img_config: ImageInstructions, seq_id: str):

        # Sequence info
        self.img_seq = img_seq
        self.img_config = img_config

        # subscribe to events
        self.image_script_task.on_image_ready += self.display_image

        # run
        self.image_script_task.run_image_sequence(self.img_seq, self.img_config, seq_id)

        # stop display
        self.app.workers.stop_display()

        # unsubscribe events
        self.image_script_task.on_image_ready -= self.display_image
