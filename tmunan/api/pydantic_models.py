from typing import List

from pydantic import BaseModel


class Txt2ImgInstructions(BaseModel):
    prompt_list: List[str]
    prompt_weights: List[float]
    num_images: int = 8
    height: int | None = 768
    width: int | None = 768
    num_inference_steps: int | None = 10
    guidance_scale: float | None = 0.0
    seed: int | None = None


class ImageSequence(BaseModel):
    txt2img: Txt2ImgInstructions
    num_images: int = 8