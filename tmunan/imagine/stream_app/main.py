from contextlib import asynccontextmanager
from typing import TypedDict, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from tmunan.imagine.stream_app.api.endpoints import router
from tmunan.imagine.stream_app.stream_manager import StreamManager


class AppState(TypedDict):
    stream_manager: StreamManager


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[AppState]:

    # initialize
    pass

    # yield fastapi state
    stream_mgr = StreamManager(max_streams=1)
    yield AppState(stream_manager=stream_mgr)

    # cleanup
    pass


# middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*']
    ),
    Middleware(
        GZipMiddleware,
        minimum_size=500
    )
]

# app instance
app = FastAPI(middleware=middleware, lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8080)
