import time

import torch
import numpy as np
from scipy.io import wavfile

from tmunan.common.exec import BackgroundExecutor
from tmunan.listen.distil_whisper.whisper_bg_task import WhisperBackgroundTask

if __name__ == '__main__':

    torch.mps.empty_cache()

    # create executor
    wp_exec = BackgroundExecutor(WhisperBackgroundTask, model_id='distil-medium')
    wp_exec.on_output_ready += lambda tran: print(f'Result: {tran}')
    wp_exec.on_error += lambda e: print(f'Error!')
    wp_exec.on_exit += lambda: print(f'Exit!')
    wp_exec.start()

    # Load the audio file
    sample_rate, audio_data = wavfile.read("/Users/himmelroman/projects/speechualizer/tmunan/test/test.wav")

    # Convert data to float64 if needed
    if audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float64)

    time.sleep(5)

    wp_exec.push_input(audio_data)
    wp_exec.push_input(audio_data)
    wp_exec.push_input(audio_data)

    time.sleep(5)
    wp_exec.stop()
