import multiprocessing
from queue import Empty

from tmunan.listen.whisper import DistilWhisper
from tmunan.pack.hls_encoder import HLSEncoder


class WhisperProcess:

    def __init__(self, model_id):

        # mp
        self.audio_queue = multiprocessing.Queue()
        self.transcript_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._worker_loop)

        # whisper
        self.model_id = model_id
        self.whisper = None

    def start(self):
        self.process.start()

    def push_input(self, chunk):
        self.audio_queue.put(chunk)

    def stop(self):
        self.audio_queue.put(None)
        self.process.join()

    def _worker_loop(self):

        # run hls encoder
        self.whisper = DistilWhisper(model_id=self.model_id)
        self.whisper.load()

        while True:
            try:
                audio_chunk = self.audio_queue.get(block=True, timeout=1)
                if audio_chunk is None:
                    break

                transcript = self.whisper.transcribe(audio_chunk)
                self.transcript_queue.put(transcript)

            except Empty:
                continue
