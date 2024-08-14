import argparse
import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from typing import Optional


import aiohttp_cors
from asyncer import asyncify

from aiohttp import web
from av import VideoFrame

import aiortc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

from tmunan.utils.log import get_logger
from tmunan.common.models import ImageParameters
from tmunan.imagine_app.client import ImagineClient

ROOT = os.path.dirname(__file__)

logger = get_logger('WebRTC Server')
pcs = set()
relay = MediaRelay()


class VideoTransformTrack(MediaStreamTrack):

    kind = "video"

    def __init__(self, input_track=None):
        super().__init__()

        # input track
        self.input_track = input_track

        # I/O
        self.input_frame_queue = asyncio.Queue(maxsize=1)
        self.output_frame_queue = asyncio.Queue(maxsize=100)
        self._task_input = None
        self._task_output = None

        # events
        self.on_image_ready_callback = None

        # synchronization flag
        self.is_running = False

        # env
        self.logger = get_logger(self.__class__.__name__)
        self._log_input_frame = False
        self._log_output_frame = False
        self._log_output_recv = False

    async def start_tasks(self):
        if not self.is_running:
            self.is_running = True
            self._task_input = asyncio.create_task(self._task_consume_input())
            self._task_output = asyncio.create_task(self._task_produce_output())

    async def stop_tasks(self):
        self.is_running = False

    async def enqueue_input_frame(self, frame: VideoFrame):

        # try to add the frame to the queue, dropping the oldest if full
        if self.input_frame_queue.full():
            try:
                self.input_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        # put on queue
        await self.input_frame_queue.put(frame)

    async def _task_consume_input(self):

        self.logger.debug('Input - Starting _task_consume_input')
        while self.is_running:

            try:

                # wait if input track is not initialized
                if not self.input_track:
                    raise ValueError

                # get frame from source track
                frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)

                # enqueue on image generation queue
                await self.enqueue_input_frame(frame)
                if not self._log_input_frame:
                    self.logger.debug(f'Input - Put frame on input queue, qsize={self.input_frame_queue.qsize()}')
                    self._log_input_frame = True

            # timeout waiting for input track's recv()
            except asyncio.TimeoutError:
                continue

            # input track not initialized
            except ValueError:
                await asyncio.sleep(0.1)
                continue

            # error reading from input track's recv()
            except aiortc.mediastreams.MediaStreamError:
                await asyncio.sleep(0.1)
                continue

            except Exception:
                self.logger.exception(f"Input - Error in _task_consume_input")

        self.logger.debug('Input - Exiting _task_consume_input')

    async def _task_produce_output(self):

        self.logger.info('Output - Starting _task_produce_output')
        while self.is_running:

            try:

                # get input frame
                frame = await asyncio.wait_for(self.input_frame_queue.get(), timeout=0.1)

                # run transform
                transformed_frame = await self.on_image_ready_callback(frame)

                # put on output queue
                await self.output_frame_queue.put(transformed_frame)
                if not self._log_output_frame:
                    self.logger.debug(f'Output - Put frame on output queue, qsize={self.output_frame_queue.qsize()}')
                    self._log_output_frame = True

            # timeout waiting for input queue's get()
            except asyncio.TimeoutError:
                pass

            except Exception:
                self.logger.exception(f"Output - Error in _task_produce_output")

        self.logger.debug('Output - Exiting _task_produce_output')

    async def recv(self):

        # self.logger.debug('Track - Executing recv()')

        # start background tasks if they are not initialized yet
        if self._task_input is None and self._task_output is None:
            await self.start_tasks()

        # get transformed frame from output queue
        frame = await self.output_frame_queue.get()

        if not self._log_output_recv:
            self._log_output_recv = True
            self.logger.debug(f"Track - Returning output frame to track consumer")

        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()

    # pc = RTCPeerConnection(
    #     configuration=RTCConfiguration([
    #         RTCIceServer("stun:stun.relay.metered.ca:80"),
    #         RTCIceServer(
    #             urls=[
    #                 "turn:global.relay.metered.ca:80",
    #                 "turn:global.relay.metered.ca:80?transport=tcp",
    #                 "turn:global.relay.metered.ca:443",
    #                 "turns:global.relay.metered.ca:443?transport=tcp"
    #             ],
    #             username="f78886871923839a14bf4731",
    #             credential="9ZUQ3gDC/0/kvKJ8"
    #         )
    #     ])
    # )
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # @pc.on('stats')
    # def on_stats(stats):
    #     print(f"Frame rate: {stats.framerateMean}")
    #     print(f"Packets lost: {stats.packetsLost}")

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")

        if track.kind == "video":
            pass
            # pc.addTrack(
            #     VideoTransformTrack(
            #         relay.subscribe(track), transform=params["video_transform"]
            #     )
            # )

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)
            # if track.kind == "video":
            #     await video_transform_track.stop()

    # video_transform_track.start()
    pc.addTrack(aiortc.mediastreams.VideoStreamTrack())
    logger.info(f"Added video output to pc")

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_startup(app):
    # asyncio.create_task(ip.processing_task())
    pass


async def on_shutdown(app):

    # close peer connections and stop video processing
    coros = [pc.close() for pc in pcs]
    for pc in pcs:
        for sender in pc.getSenders():
            if sender.track and isinstance(sender.track, VideoTransformTrack):
                coros.append(sender.track.stop())
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    # # load LCM
    # lcm = StreamLCM(model_id='sd-turbo')
    # lcm.load()

    # image processor
    # ip = ImageProcessor()

    app = web.Application()

    # `aiohttp_cors.setup` returns `aiohttp_cors.CorsConfig` instance.
    # The `cors` instance will store CORS configuration for the
    # application.

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
