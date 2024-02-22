import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class DistilWhisper:

    model_map = {
        'distil-medium': {
            'model': "distil-whisper/distil-medium",
        },
        'distil-large': {
            'model': "distil-whisper/distil-large-v2",
        }
    }

    def __init__(self, model_id=None):

        # model
        self.model_id = model_id

        # pipeline
        self.whisper_pipeline = None

        # comp device
        self.device = self.get_device()

    @classmethod
    def get_device(cls):

        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load(self):

        # load model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_map[self.model_id]['model'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        # load pipeline
        processor = AutoProcessor.from_pretrained(self.model_map[self.model_id]['model'])
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype= torch.float16,
            device=self.device,
        )

        print(f'Done')

    def transcribe(self, audio_sample):

        # transcribe
        result = self.whisper_pipeline(audio_sample)

        # return transcript
        return result["text"]
