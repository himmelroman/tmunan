import logging
import os
import time
import signal
import asyncio
from contextlib import asynccontextmanager
from typing import TypedDict, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from tmunan.stream_app.api.endpoints import router
from tmunan.stream_app.webrtc.stream_manager import WebRTCStreamManager


logger = logging.getLogger()
logger.setLevel("INFO")


async def shutdown_monitor(stream_manager: WebRTCStreamManager):

    # read idle timeout value from env (default is 15 min)
    idle_timeout_seconds = float(os.environ.get("IDLE_TIMEOUT_SECONDS", 60 * 15))

    # loop forever
    while True:

        # log
        logger.info(f'Shutdown Monitor - Activity: {stream_manager.last_activity}, Max Idle: {idle_timeout_seconds}')

        # check if idle timeout reached
        if time.time() - stream_manager.last_activity > idle_timeout_seconds:

            logger.info(f'Shutdown Monitor - Shutting down')

            # shutdown stream manager
            await stream_manager.shutdown(reason="inactivity")

            # shutdown the application gracefully
            os.kill(os.getpid(), signal.SIGKILL)

        # check every minute
        await asyncio.sleep(60)


class AppState(TypedDict):
    stream_manager: WebRTCStreamManager


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[AppState]:

    # create stream manager
    stream_manager = WebRTCStreamManager()
    await asyncio.create_task(shutdown_monitor(stream_manager))

    # yield fastapi state
    yield AppState(stream_manager=stream_manager)

    # cleanup
    await stream_manager.cleanup()


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
