import multiprocessing

from tmunan.display.hls_encoder import HLSEncoder


class HLSEncoderProcess:

    def __init__(self, hls_dir, width, height, fps):

        # mp
        self.mp_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._worker_loop)

        # hls
        self.hls_encoder = HLSEncoder(hls_dir, (height, width), fps)

    def start(self):
        self.process.start()

    def push_input(self, item):
        self.mp_queue.put(item)

    def stop(self):
        self.mp_queue.put(None)
        self.process.join()

    def _worker_loop(self):

        # run hls encoder
        self.hls_encoder.run(self.mp_queue)
