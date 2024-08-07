from tmunan.utils.exec import BackgroundTask
from tmunan.listen.distil_whisper.whisper import DistilWhisper


class WhisperBackgroundTask(BackgroundTask):

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = args
        self._kwargs = kwargs

        self.model_id = model_id
        self.whisper = None

    def setup(self):

        try:
            # load model
            self.whisper = DistilWhisper(model_id=self.model_id)
            self.whisper.load()

        except Exception as e:
            print(e)

    def exec(self, item):
        return self.whisper.transcribe(item)
