import os
from whisper_live.client import TranscriptionClient

os.environ["TERM"] = 'xterm'

if __name__ == '__main__':
    client = TranscriptionClient(host='localhost', port=9090,
                                 is_multilingual=True, lang='he', translate=True)
    client()
