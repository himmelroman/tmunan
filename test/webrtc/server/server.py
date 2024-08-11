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

import cv2
from aiohttp import web
from av import VideoFrame
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
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, input_track, transform):
        super().__init__()

        self.transform = transform
        self.input_track = input_track

        # imagine client
        self.img_client = ImagineClient(host='localhost', port=8090, secure=False)

        # I/O
        self.input_frame_queue = asyncio.Queue(maxsize=1)
        self.output_frame_queue = asyncio.Queue(maxsize=100)
        self._task_input = None
        self._task_output = None

        # synchronization flag
        self.is_running = True

    async def start(self):
        self.is_running = True
        self._task_input = asyncio.create_task(self._task_consume_input())
        self._task_output = asyncio.create_task(self._task_produce_output())

    async def stop(self):
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

        logger.info('INPUT - Starting _task_consume_input')
        while self.is_running:

            try:

                # get frame from source track
                frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)

                # enqueue on image generation queue
                await self.enqueue_input_frame(frame)
                logger.info(f'INPUT - Put frame on queue, qsize={self.input_frame_queue.qsize()}')

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logging.exception(f"Error in _task_consume_input")

    async def _task_produce_output(self):

        logger.info('OUTPUT - Starting _task_produce_output')
        while self.is_running:

            try:

                # get input frame
                frame = await asyncio.wait_for(self.input_frame_queue.get(), timeout=0.1)

                # run transform
                transformed_frame = await self.transform_frame(frame)

                # put on output queue
                await self.output_frame_queue.put(transformed_frame)
                logger.info(f'OUTPUT - Put frame on queue, qsize={self.output_frame_queue.qsize()}')

            except asyncio.TimeoutError:
                pass

            except Exception as e:
                logging.exception(f"Error in _task_produce_output")

    async def recv(self):

        logger.info('RECV - Running recv')
        if self._task_input is None and self._task_output is None:
            await self.start()

        # get transformed frame from output queue
        frame = await self.output_frame_queue.get()

        logger.info('RECV - Output frame!')
        return frame

    def test(self, frame):

        image = frame.to_image()
        new_image = self.img_client.post_image(
            image=image,
            params=ImageParameters(
                prompt='evil nina simone',
                guidance_scale=1.0,
                strength=1.5,
                height=640,
                width=480,
                seed=123
        ))
        new_frame = VideoFrame.from_image(new_image)
        return new_frame

    async def transform_frame(self, frame):

        if self.transform == "cartoon":

            # gen image
            logger.info('TRANS - Sending frame to img2img')
            new_frame = await asyncify(self.test)(frame)

            # assign timestamps
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

        if self.transform == "edges":

            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        elif self.transform == "rotate":

            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        else:
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
            video_transform_track = VideoTransformTrack(track, transform='cartoon')
            # video_transform_track.start()
            pc.addTrack(video_transform_track)

            logger.info(f"Added track to pc")
            # pc.addTrack(
            #     VideoTransformTrack(
            #         relay.subscribe(track), transform=params["video_transform"]
            #     )
            # )

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)
            if track.kind == "video":
                await video_transform_track.stop()

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
