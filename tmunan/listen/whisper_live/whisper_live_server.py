from tmunan.listen.whisper_live.TranscriptServer import TmunanTranscriptionServer

if __name__ == '__main__':
    server = TmunanTranscriptionServer()
    server.run("0.0.0.0", 9090)
