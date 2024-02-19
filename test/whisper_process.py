import torch
import numpy as np
from scipy.io import wavfile

from tmunan.listen.whisper_worker_process import WhisperProcess


if __name__ == '__main__':

    torch.mps.empty_cache()

    wp = WhisperProcess(model_id='distil-medium')
    wp.start()

    # Load the audio file
    sample_rate, audio_data = wavfile.read("/Users/himmelroman/projects/speechualizer/tmunan/test/test.wav")

    # Convert data to float64 if needed
    if audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float64)

    wp.push_input(audio_data)
    wp.push_input(audio_data)
    wp.push_input(audio_data)

    for i in range(3):
        res = wp.transcript_queue.get(block=True)
        print(f'Result {i}: {res}')

    wp.stop()
