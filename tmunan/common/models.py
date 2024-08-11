import sys

from fastapi import Form
from pydantic import BaseModel


class ImageParameters(BaseModel):
    prompt: str = Form(default='')
    negative_prompt: str = Form(default='')
    guidance_scale: float = Form(default=1.0, ge=0.0, le=2.0)
    strength: float = Form(default=1.5, ge=0.0, le=3.0)
    height: int = Form(default=512, ge=0)
    width: int = Form(default=904, ge=0)
    seed: int = Form(default=1, ge=0, le=sys.maxsize)
