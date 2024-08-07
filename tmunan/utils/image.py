import io
import os
from io import BytesIO
from typing import Union, Callable

import requests
import PIL.Image
import PIL.ImageOps


def bytes_to_pil(image_bytes: bytes) -> PIL.Image.Image:
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_bytes(image: PIL.Image.Image, format="webp", quality=100) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format=format, quality=quality)
    return frame_data.getvalue()


def pil_to_frame(image: PIL.Image.Image, format="webp", quality=100) -> bytes:

    # convert to bytes
    frame_data = pil_to_bytes(image, format, quality)

    return (
            b"--frame\r\n"
            + b"Content-Type: image/webp\r\n"
            + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
            + frame_data
            + b"\r\n"
    )


def load_image(
        image: Union[str, PIL.Image.Image],
        convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(BytesIO(requests.get(image, stream=True).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image
