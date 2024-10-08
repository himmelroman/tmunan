import io

from PIL import Image
from fastapi import Request, Response, APIRouter, UploadFile, HTTPException, Depends, status

from tmunan.utils.image import pil_to_bytes
from tmunan.common.models import ImageParameters

router = APIRouter()


@router.get("/api/health")
async def health(req: Request, status_code=status.HTTP_200_OK):
    return {
        "status": "ok",
        "model": {
            "id": req.state.img_gen.model_id
        }
    }


@router.post("/api/txt2img")
async def txt2img(req: Request, data: ImageParameters = Depends()):

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
        image_format = "jpeg"
        image_quality = 95
        image_bytes = pil_to_bytes(images[0], format=image_format, quality=image_quality)

        # respond
        return Response(content=image_bytes, media_type=f"image/{image_format}")

    except Exception as ex:
        req.state.logger.exception('Error in txt2img')
        raise HTTPException(status_code=500, detail="Image generation failed")


@router.post("/api/img2img")
async def img2img(req: Request, image: UploadFile, data: ImageParameters = Depends()):

    # check busy state
    if req.state.in_progress:
        req.state.logger.info(f'Imagine Server is busy! Rejecting request')
        raise HTTPException(status_code=503, detail="Server busy, image generation is in progress")

    # flag in progress
    req.state.in_progress = True

    try:
        # read uploaded image
        file_content = await image.read()
        img = Image.open(io.BytesIO(file_content))

        # generate image
        req.state.logger.info(f'Executing request with params: {data.model_dump()}')
        images = req.state.img_gen.img2img(
            image=img,
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            # num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale,
            strength=data.strength,
            height=data.height,
            width=data.width,
            seed=data.seed,
            randomize_seed=bool(data.seed == 0)
        )

        req.state.in_progress = False

        # convert image to bytes
        image_format = "jpeg"
        image_quality = 95
        image_bytes = pil_to_bytes(images[0], format=image_format, quality=image_quality)

        # respond
        return Response(content=image_bytes, media_type=f"image/{image_format}")

    except Exception as ex:
        req.state.logger.exception('Error in img2img')
        raise HTTPException(status_code=500, detail="Image generation failed")
