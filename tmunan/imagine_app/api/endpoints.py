import io
import logging

from PIL import Image
from fastapi import Request, Response, Form, APIRouter, UploadFile, HTTPException, Depends

from tmunan.utils.image import pil_to_bytes
from tmunan.common.models import ImageParameters

router = APIRouter()


@router.post("/api/txt2img")
async def img2img(req: Request, data: ImageParameters = Form(...)):

    # check busy state
    if req.state.in_progress:
        raise HTTPException(status_code=503, detail="Server busy, image generation is in progress")

    # flag in progress
    req.state.in_progress.in_progress = True

    try:

        # generate image
        images = req.state.img_gen.txt2img(
            prompt=data.prompt,
            num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale,
            strength=data.strength,
            height=data.height, width=data.width,
            seed=data.seed,
            randomize_seed=bool(data.seed == 0)
        )

        # convert image to bytes
        image_format = "webp"
        image_bytes = pil_to_bytes(images[0], format=image_format)

        # respond
        return Response(content=image_bytes, media_type=f"image/{image_format}")

    except Exception as ex:
        logging.exception('Error in txt2img')
        raise HTTPException(status_code=500, detail="Image generation failed")


@router.post("/api/img2img")
async def img2img(req: Request, image: UploadFile, data: ImageParameters = Depends()):

    # check busy state
    if req.state.in_progress:
        raise HTTPException(status_code=503, detail="Server busy, image generation is in progress")

    # flag in progress
    req.state.in_progress = True

    try:
        # read uploaded image
        file_content = await image.read()
        img = Image.open(io.BytesIO(file_content))

        # generate image
        images = req.state.img_gen.img2img(
            image=img,
            prompt=data.prompt,
            # num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale,
            strength=data.strength,
            height=data.height, width=data.width,
            seed=data.seed,
            randomize_seed=bool(data.seed == 0)
        )

        req.state.in_progress = False

        # convert image to bytes
        image_format = "webp"
        image_bytes = pil_to_bytes(images[0], format=image_format)

        # respond
        return Response(content=image_bytes, media_type=f"image/{image_format}")

    except Exception as ex:
        logging.exception('Error in img2img')
        raise HTTPException(status_code=500, detail="Image generation failed")
