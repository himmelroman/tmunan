import sys

from pydantic import BaseModel, Field


class StreamInputParams(BaseModel):
    prompt: str = Field(
        'Default prompt text - change in the code!',
        title="Prompt",
        field="textarea",
        id="prompt",
    )
    strength: float = Field(
        1.0, min=1.0, max=2.5, title="Strength", disabled=True, hide=True, id="strength"
    )
    guidance_scale: float = Field(
        1.0, min=1.0, max=2.5, title="Guidance Scale", disabled=True, hide=True, id="guidance_scale"
    )
    seed: int = Field(
        0, min=0, max=sys.maxsize, title="Seed", disabled=True, hide=True, id="seed"
    )
    # negative_prompt: str = Field(
    #     default_negative_prompt,
    #     title="Negative Prompt",
    #     field="textarea",
    #     id="negative_prompt",
    # )

    width: int = Field(
        512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
    )
    height: int = Field(
        512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
    )