from typing import List

from pydantic import BaseModel


class ImageInstructions(BaseModel):
    height: int | None = 768
    width: int | None = 768
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.5
    strength: float | None = 0.3
    seed: int | None = None


class SequencePrompt(BaseModel):
    text: str
    start_weight: float = 1
    end_weight: float = 1


class ImageSequence(BaseModel):
    prompts: List[SequencePrompt]
    num_images: int = 8


class ImageSequenceScript(BaseModel):
    sequences: List[ImageSequence]
    loop: bool = False
