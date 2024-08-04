import logging

from tmunan.common.event import Event
from tmunan.common.log import get_logger
from tmunan.common.exec import BackgroundExecutor, BackgroundTask

from tmunan.imagine.sd_lcm.lcm_control import ControlLCM
from tmunan.imagine.sd_lcm.lcm_stream import StreamLCM


class ImageGeneratorWorker:

    def __init__(self, model_id, diff_type):

        # events
        self.on_image_ready = Event()
        self.on_startup = Event()
        self.on_shutdown = Event()

        # executor
        self.bg_exec = BackgroundExecutor(
            task_class=ImageGeneratorBGTask,
            proc_method=BackgroundExecutor.ProcessCreationMethod.Spawn,
            model_id=model_id,
            diff_type=diff_type
        )

    def start(self):

        # subscribe to events
        self.bg_exec.on_output_ready += lambda res: self.on_image_ready.notify(res)
        self.bg_exec.on_worker_ready += self.on_startup.notify
        self.bg_exec.on_exit += self.on_shutdown.notify
        self.bg_exec.on_error += lambda ex: logging.error(f'Error: {ex}')

        # start
        self.bg_exec.start()

    def stop(self):
        self.bg_exec.stop()

    def img2img(self, handle_id, **kwargs):
        kwargs['handle_id'] = handle_id
        self.bg_exec.push_input(kwargs)

    # def txt2img(self, **kwargs):
    #     kwargs['task'] = TaskType.Text2Image.value
    #     self.bg_exec.push_input(kwargs)


class ImageGeneratorBGTask(BackgroundTask):

    def __init__(self, model_id, diff_type):
        super().__init__()

        # models
        self.lcm = None
        self.model_id = model_id
        self.diff_type = diff_type

        # env
        self.logger = None

    def setup(self):

        self.logger = get_logger(self.__class__.__name__)

        try:

            # check diffusion type
            if self.diff_type == 'stream':
                self.lcm = StreamLCM(model_id=self.model_id)

            elif self.diff_type == 'control':
                self.lcm = ControlLCM(model_id=self.model_id)

            self.lcm.load()

        except Exception as ex:
            self.logger.exception(f'Error in {self.__class__.__name__} setup!')

    def exec(self, img_gen_args):

        try:

            # handle id
            handle_id = img_gen_args.pop('handle_id', None)

            # run img2img
            images = self.lcm.img2img(**img_gen_args)
            return handle_id, images[0]

        except Exception as ex:
            self.logger.exception(f'Error in {self.__class__.__name__} exec!')
