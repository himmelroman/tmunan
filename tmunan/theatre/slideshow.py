import numpy as np

from tmunan.theatre.performance import Performance
from tmunan.tasks.image_script import ImageScript

from tmunan.imagine.sd_lcm.lcm import load_image
from tmunan.api.pydantic_models import ImageSequence, ImageInstructions, ImageSequenceScript


class Slideshow(Performance):

    def __init__(self, app):
        super().__init__()

        # Host app
        self.app = app

        # Workers
        self.read = app.workers.read
        self.imagine = app.workers.imagine
        self.display = app.workers.display

        # Script info
        self.img_script = None
        self.img_config = None

        # Imagine Task
        self.image_script_task = ImageScript(self.imagine, self.cache_dir)

        # Bind events
        self.read.on_prompt_ready += self.push_text

    def push_text(self, text_prompt):
        print(f'on_prompt_ready fired with: {text_prompt}')
        self.image_script_task.set_text_prompt(text_prompt)

    def display_image(self, image_info):
        print(f'on_image_ready fired with: {image_info}')

        # put on queue
        image = load_image(image_info['image_path'])
        self.display.push_image(np.array(image, dtype=np.uint8))

    def run(self, img_script: ImageSequenceScript, img_config: ImageInstructions, seq_id: str):

        # Sequence info
        self.img_script = img_script
        self.img_config = img_config

        # subscribe to events
        self.image_script_task.on_image_ready += self.display_image

        # run
        self.image_script_task.run_script(self.img_script, self.img_config, seq_id)

        # stop display
        self.app.workers.stop_display()

        # unsubscribe events
        self.image_script_task.on_image_ready -= self.display_image
