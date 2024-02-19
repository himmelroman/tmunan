import torch
import numpy as np
from scipy.io import wavfile

from tmunan.listen.whisper import DistilWhisper


if __name__ == '__main__':

    torch.mps.empty_cache()

    dw = DistilWhisper(model_id='distil-medium')
    dw.load()

    # Load the audio file
    sample_rate, audio_data = wavfile.read("/Users/himmelroman/projects/speechualizer/tmunan/test/test.wav")

    # Convert data to float64 if needed
    if audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float64)

    print(audio_data.shape)  # Outputs: (n_samples,) for mono, (n_samples, n_channels) for stereo

    res = dw.transcribe(audio_data)
    print(res)
