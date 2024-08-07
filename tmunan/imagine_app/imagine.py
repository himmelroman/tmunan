import os

from typing import TypedDict, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from tmunan.imagine_app.api.endpoints import router
from tmunan.imagine_app.sd.lcm_stream import StreamLCM


class AppState(TypedDict):
    img_gen: StreamLCM
    in_progress: bool


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[AppState]:

    # load image generator
    model_id = os.environ.get('MODEL_ID') or 'sd-turbo'
    img_gen = StreamLCM(model_id=model_id)
    img_gen.load()

    # yield fastapi state
    yield AppState(img_gen=img_gen, in_progress=False)

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


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8090)
