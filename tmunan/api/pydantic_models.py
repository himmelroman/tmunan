from typing import List

from pydantic import BaseModel

from tmunan.imagine.sd_lcm.lcm_bg_task import TaskType


class ImageInstructions(BaseModel):
    height: int | None = 768
    width: int | None = 768
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.5
    strength: float | None = 0.3
    seed: int | None = None
    images_per_second: int = 2


class SequencePrompt(BaseModel):
    text: str
    start_weight: float = 1
    end_weight: float = 1


class ImageSequence(BaseModel):
    prompts: List[SequencePrompt]
    num_images: int = 10
    transition: TaskType


class ImageSequenceScript(BaseModel):
    sequences: List[ImageSequence]
    loop: bool = False
    transition: TaskType
