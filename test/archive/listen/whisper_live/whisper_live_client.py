import time

from tmunan.listen.whisper_live.TranscriptClient import TranscriptClient


sentences = []


def process_message(message):

    print(message)

    # last_processed_time = sentences[-1]['segments'][-1]['end'] if sentences else 0
    # for seg in message['segments']:
    #
    #     # skip segments which were already processed
    #     if float(seg['end']) < float(last_processed_time):
    #         continue
    #
    #     # verify last open result
    #     if not sentences or sentences[-1]['finalized']:
    #         sentences.append({
    #             'segments': [],
    #             'finalized': False
    #         })
    #
    #     # add segment
    #     sentences[-1]['segments'].append(seg)
    #
    #     # check for sentence finalization
    #     if '.' in seg['text']:
    #         sentences[-1]['finalized'] = True
    #         print(' '.join(seg['text'] for seg in sentences[-1]['segments']))


if __name__ == '__main__':
    client = TranscriptClient(host='localhost', port=9090,
                              is_multilingual=True, lang='en', translate=True,
                              on_message=process_message)
    time.sleep(2)
    client.record()
