from fastapi import Form
from pydantic import BaseModel


class ImageParameters(BaseModel):
    prompt: str = Form()
    guidance_scale: float       # = Form(ge=0.0, le=2.0)
    strength: float             # = Form(ge=0.0, le=2.0)
    height: int                 # = Form(ge=0)
    width: int                  # = Form(ge=0)
    seed: int                   # = Form(ge=0, le=sys.maxsize)
