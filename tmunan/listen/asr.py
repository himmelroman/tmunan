from tmunan.common.exec import BackgroundExecutor
from tmunan.listen.text_buffer import TextBuffer
from tmunan.listen.distil_whisper.whisper_bg_task import WhisperBackgroundTask


class ASR:

    def __init__(self):
        self.audio_buffer = []
        self.text_buffer = TextBuffer()
        self.asr_executor = BackgroundExecutor(WhisperBackgroundTask, model_id='distil-large')

    def start(self):

        # start executor
        self.asr_executor.start()

        # subscribe to event
        self.asr_executor.on_output_ready += self.buffer_text

    def push_audio(self, audio_chunk: bytes):

        # add item to buffer
        self.audio_buffer.append(audio_chunk)

        # Check if enough audio has accumulated
        if len(self.audio_buffer) >= 1:

            # join audio
            audio_sample = b''.join(self.audio_buffer)

            # clean buffer
            self.audio_buffer.clear()

            # push to asr
            print(f'Pushing audio into asr_executor: {len(audio_sample)}')
            self.asr_executor.push_input(audio_sample)

    def stop(self):
        self.asr_executor.stop()

    def buffer_text(self, text):
        print(f'Pushing text into TextBuffer: {text}')
        self.text_buffer.push_text(text)

    def consume_text(self):
        return self.text_buffer.consume_text()
