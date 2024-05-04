import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from pathlib import Path

import cv2
import torch
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# from tmunan.imagine.sd_lcm.lcm import LCM

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

        # img2img
        self.in_progress = False
        self.last_frame = None
        self.frame_index = 0
        self.work_dir = Path('/tmp/img2img')
        self.work_dir.mkdir(exist_ok=True, parents=True)

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "cartoon":

            if self.in_progress:
                new_frame = self.last_frame
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                print(f'Recycling previous frame: {frame.pts=}, {frame.time_base=}')
                return new_frame
            else:
                # print('Generating new frame')
                self.in_progress = True

            # dump frame to image
            image_path = self.work_dir.with_name(f'img_{self.frame_index}.png')
            frame.to_image().save(str(image_path))
            self.frame_index += 1

            # gen image
            # res_images = lcm.img2img(
            #     prompt='cartoon',
            #     image_url=str(image_path),
            #     num_inference_steps=1,
            #     guidance_scale=0.6,
            #     strength=0.3,
            #     height=640, width=480
            # )
            res = pipe(
                prompt='frida kahlo, self portrait',
                image=frame.to_image(),
                width=512,
                height=512,
                guidance_scale=0.5,
                num_inference_steps=1,
                num_images_per_prompt=1,
                output_type="pil",
                controlnet_conditioning_scale=1.2
            )
            res_image = res.images[0]   # .resize(640, 480)

            # pack resulting frame
            print(f'Outputing new frame: {frame.pts=}, {frame.time_base=}')
            new_frame = VideoFrame.from_image(res_image)
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            # return
            self.last_frame = new_frame
            self.in_progress = False
            return new_frame

            # img = frame.to_ndarray(format="bgr24")
            #
            # # prepare color
            # img_color = cv2.pyrDown(cv2.pyrDown(img))
            # for _ in range(6):
            #     img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            # img_color = cv2.pyrUp(cv2.pyrUp(img_color))
            #
            # # prepare edges
            # img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img_edges = cv2.adaptiveThreshold(
            #     cv2.medianBlur(img_edges, 7),
            #     255,
            #     cv2.ADAPTIVE_THRESH_MEAN_C,
            #     cv2.THRESH_BINARY,
            #     9,
            #     2,
            # )
            # img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
            #
            # # combine color and edges
            # img = cv2.bitwise_and(img_color, img_edges)
            #
            # rebuild a VideoFrame, preserving timing information
            # new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            # new_frame.pts = frame.pts
            # new_frame.time_base = frame.time_base
            # return new_frame

        elif self.transform == "edges":
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

    pc = RTCPeerConnection(
        configuration=RTCConfiguration([
            RTCIceServer("stun:stun.l.google:19302"),
            RTCIceServer("turn:global.relay.metered.ca:80", "f78886871923839a14bf4731", "9ZUQ3gDC/0/kvKJ8"),
            RTCIceServer("turn:global.relay.metered.ca:80?transport=tcp", "f78886871923839a14bf4731", "9ZUQ3gDC/0/kvKJ8"),
            RTCIceServer("turn:global.relay.metered.ca:443", "f78886871923839a14bf4731", "9ZUQ3gDC/0/kvKJ8"),
            RTCIceServer("turns:global.relay.metered.ca:443?transport=tcp", "f78886871923839a14bf4731", "9ZUQ3gDC/0/kvKJ8"),
        ])
    )
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

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
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
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
    parser.add_argument("--record-to", help="Write received media to a file.")
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

    # load LCM
    # lcm = LCM(model_size='small')
    # lcm.load()
    device = "cuda"
    weight_type = torch.float16

    controlnet = ControlNetModel.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper-sketch", torch_dtype=weight_type
    ).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "IDKiro/sdxs-512-dreamshaper", controlnet=controlnet, torch_dtype=weight_type
    )
    pipe.to(device)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
