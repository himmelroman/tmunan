import os
import time
import logging

from typing import TypedDict, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from tmunan.imagine_app.api.endpoints import router
from tmunan.imagine_app.sd.lcm_stream import StreamLCM
from tmunan.utils.log import get_logger


class AppState(TypedDict):
    img_gen: StreamLCM
    in_progress: bool
    logger: logging.Logger


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[AppState]:

    # load image generator
    model_id = os.environ.get('MODEL_ID') or 'sd-turbo'
    img_gen = StreamLCM(model_id=model_id)
    img_gen.load()

    # yield fastapi state
    yield AppState(img_gen=img_gen, in_progress=False, logger=get_logger('ImagineApp'))

    # cleanup
    pass


# app instance
middleware = [
    Middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*']
    ),
    # Middleware(
    #     GZipMiddleware,
    #     minimum_size=500
    # )
]

# app instance
app = FastAPI(middleware=middleware, lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    request.state.logger.info(f"ReqTrace - Imagine Request exec time: {time.time() - start_time}")
    return response


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8090)
