import aiortc
import threading

# Create a lock
lock = threading.Lock()


def patch_vpx():

    # get original method
    original_encode = aiortc.codecs.vpx.Vp8Encoder.encode

    # New encode method with lock
    def patched_encode_method(self, *args, **kwargs):
        with lock:
            return original_encode(self, *args, **kwargs)

    # Apply the patch
    aiortc.codecs.vpx.Vp8Encoder.encode = patched_encode_method
