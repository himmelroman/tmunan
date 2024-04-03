from typing import List

from pydantic import BaseModel

from tmunan.imagine.sd_lcm.lcm_bg_task import TaskType


class Prompt(BaseModel):
    text: str


class BaseImage(BaseModel):
    image_url: str


class ImageInstructions(BaseModel):
    height: int | None = 768
    width: int | None = 768
    num_inference_steps: int | None = 4
    guidance_scale: float | None = 0.5
    strength: float | None = 0.3
    seed: int | None = None
    key_frame_period: int = 3
    key_frame_repeat: int = 2
    output_fps: int = 12


class ReadTextPrompt(BaseModel):
    text: str


class TextInstructions(BaseModel):
    start_weight: float = 1
    end_weight: float = 1


class SequencePrompt(BaseModel):
    text: str
    negative_text: str
    start_weight: float = 1
    end_weight: float = 1


class ImageSequence(BaseModel):
    transition: TaskType
    prompts: List[SequencePrompt]
    base_image_url: str | None = None
    num_images: int = 10


class ImageSequenceScript(BaseModel):
    sequences: List[ImageSequence]
    loop: bool = False
    loop_count: int = 10
    keep_rtf: bool | None = False
