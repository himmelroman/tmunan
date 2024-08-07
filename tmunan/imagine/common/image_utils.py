import io
from PIL import Image


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_bytes(image: Image.Image, format="webp", quality=100) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format=format, quality=quality)
    return frame_data.getvalue()


def pil_to_frame(image: Image.Image, format="webp", quality=100) -> bytes:

    # convert to bytes
    frame_data = pil_to_bytes(image, format, quality)

    return (
            b"--frame\r\n"
            + b"Content-Type: image/webp\r\n"
            + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
            + frame_data
            + b"\r\n"
    )
