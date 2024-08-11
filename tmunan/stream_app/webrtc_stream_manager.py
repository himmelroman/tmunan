import os
import json
import logging
import asyncio
from typing import Dict
from uuid import UUID

import aiortc
from asyncer import asyncify, syncify

from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from tmunan.utils.log import get_logger
from tmunan.common.models import ImageParameters
from tmunan.imagine_app.client import ImagineClient


class StreamPeerConnection:

    def __init__(self, id: UUID, name: str, pc: RTCPeerConnection, data_channel: RTCDataChannel = None):

        self.id: UUID = id
        self.name: str = name
        self.pc: RTCPeerConnection = pc
        self.data_channel: RTCDataChannel = data_channel


class FrameTransformTrack(MediaStreamTrack):

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
        self.is_running = True

        # env
        self.logger = get_logger(self.__class__.__name__)

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

        self.logger.info('INPUT - Starting _task_consume_input')
        while self.is_running:

            try:

                # wait if input track is not initialized
                if not self.input_track:
                    await asyncio.sleep(0.1)
                    continue

                # get frame from source track
                frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.01)

                # enqueue on image generation queue
                await self.enqueue_input_frame(frame)
                # self.logger.info(f'INPUT - Put frame on queue, qsize={self.input_frame_queue.qsize()}')

            except aiortc.mediastreams.MediaStreamError:
                continue

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logging.exception(f"Error in _task_consume_input")

    async def _task_produce_output(self):

        self.logger.info('OUTPUT - Starting _task_produce_output')
        while self.is_running:

            try:

                # get input frame
                frame = await asyncio.wait_for(self.input_frame_queue.get(), timeout=0.1)

                # run transform
                transformed_frame = await self.on_image_ready_callback(frame)

                # put on output queue
                await self.output_frame_queue.put(transformed_frame)
                # self.logger.info(f'OUTPUT - Put frame on queue, qsize={self.output_frame_queue.qsize()}')

            except asyncio.TimeoutError:
                pass

            except Exception as e:
                logging.exception(f"Error in _task_produce_output")

    async def recv(self):

        # self.logger.info('RECV - Running recv')
        if self._task_input is None and self._task_output is None:
            await self.start()

        # get transformed frame from output queue
        frame = await self.output_frame_queue.get()

        # self.logger.info('RECV - Output frame!')
        return frame


class WebRTCStreamManager:

    def __init__(self):

        # peer registry
        self.media_relay = MediaRelay()
        self.video_transform_track = FrameTransformTrack()
        self.peer_connections: Dict[str, StreamPeerConnection] = dict()
        self.active_connection_name = None

        # img generation
        host = os.environ.get("IMAGINE_HOST", "localhost")
        port = os.environ.get("IMAGINE_PORT", "8090")
        secure = bool(os.environ.get("IMAGINE_SECURE", False))
        self.img_client = ImagineClient(host=host, port=port, secure=secure)
        self.parameters = ImageParameters()

        # env
        self.logger = get_logger(self.__class__.__name__)

    async def add_peer_connection(self, spc):

        # close existing
        if spc.name in self.peer_connections:
            await self.peer_connections[spc.name].pc.close()
            self.peer_connections.pop(spc.name, None)

        # add to registry
        self.peer_connections[spc.name] = spc

    async def set_active_peer_connection(self, name):

        # update connection
        self.active_connection_name = name
        self.logger.info(f"Active connection set: {self.active_connection_name}")

        # publish state
        await self.publish_state()

    async def cleanup(self):

        # collection stop() coroutines all peer connections
        task_list = [spc.pc.close() for spc in list(self.peer_connections.values())]

        # add video transform track to close() list
        if self.video_transform_track:
            task_list.append(self.video_transform_track.stop())

        # join all peer close coroutines
        await asyncio.gather(*task_list)

        # clear peer dict
        self.peer_connections.clear()

    async def publish_state(self):

        state = {
                    "type": "state",
                    "payload": {
                        "connections": [
                            {
                                "id": str(spc.id),
                                # "info": conn.info
                            }
                            for spc in self.peer_connections.values()
                        ],
                        "active_connection_name": self.active_connection_name,
                        "parameters": self.parameters.model_dump()
                    }
                }

        # iterate all stream peer connections
        for spc in list(self.peer_connections.values()):
            try:

                # check if data channel is connected
                if spc.data_channel:

                    # send state
                    spc.data_channel.send(json.dumps(state))

            except Exception as ex:
                self.logger.exception(f'Error in publish state!')
                #self.peer_connections.pop(conn.name)

    async def handle_offer(self, id: UUID, name: str, output: bool, sdp, type):

        # parse incoming offer
        offer = RTCSessionDescription(sdp=sdp, type=type)

        # create peer connection
        pc = RTCPeerConnection()
        spc = StreamPeerConnection(id=id, name=name, pc=pc)

        # add to registry
        await self.add_peer_connection(spc)

        @pc.on("datachannel")
        async def on_datachannel(channel):

            # save data channel
            spc.data_channel = channel

            # check if there is no active peer
            if not self.active_connection_name:
                await self.set_active_peer_connection(spc.name)

            # publish change
            await self.publish_state()

            @channel.on("message")
            async def on_message(message):

                self.logger.info(f'Message from {name}: {message}')
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])
                else:
                    await self.handle_control_message(message)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.logger.info(f"Connection state is {pc.connectionState}")

            # handle failed connection
            if pc.connectionState == "failed":

                # close connection
                await pc.close()

                # pop from registry
                self.peer_connections.pop(spc.name)

            elif pc.connectionState == "connected":
                pass

            else:
                self.logger.warning(f"Unhandled connection state: {pc.connectionState}")

        @pc.on("track")
        def on_track(track):
            self.logger.info(f"Track received: {track.kind}")

            # handle video track
            if track.kind == "video":

                # check if connecting peer is active
                if self.active_connection_name == spc.name:
                    # syncify(self.set_active_peer_connection)(spc.name)
                    self.video_transform_track.input_track = track

                self.video_transform_track.on_image_ready_callback = self.transform_frame
                self.logger.info(f"Added {track.kind} track to pc")

                # check if video feed requested
                if output:
                    pc.addTrack(self.media_relay.subscribe(self.video_transform_track, buffered=False))

            @track.on("ended")
            async def on_ended():
                self.logger.info(f"Track {track.kind} ended")

                if track.kind == "video":
                    await self.video_transform_track.stop()

        # handle offer
        await pc.setRemoteDescription(offer)

        # create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }

    async def handle_control_message(self, app_msg):

        # parse json
        app_msg = json.loads(app_msg)
        self.logger.info(f"Incoming message: {app_msg}")

        # extract app message
        if app_msg['type'] == "set_parameters":
            if self.parameters.prompt == '' or app_msg['payload'].get('override', False) is True:
                self.parameters = self.parameters.model_copy(update=app_msg['payload'])
                await self.publish_state()
                self.logger.info(f"Parameters set: {self.parameters.model_dump()}")

        elif app_msg['type'] == "set_active_name":

            await self.set_active_peer_connection(app_msg['payload']['name'])
            await self.publish_state()

        # elif app_msg['type'] == "set_connection_info":
        #     if conn := self.connections.get(UUID(app_msg['payload']['connection_id'])):
        #         conn.info.update(app_msg['payload']['info'])
        #         await self.publish_state()
        #         self.logger.info(f"Connection info updated: {conn.id=} - {conn.info}")

        else:
            self.logger.info(f"Unrecognized message: {app_msg}")

    async def transform_frame(self, frame):

        # gen image
        # self.logger.info('TRANS - Sending frame to img2img')
        new_frame = await asyncify(self.request_img2img)(frame)

        # assign timestamps
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

    def request_img2img(self, frame):

        try:
            image = frame.to_image()
            new_image = self.img_client.post_image(
                image=image,
                params=self.parameters
            )
            new_frame = VideoFrame.from_image(new_image)
            return new_frame

        except Exception as ex:
            self.logger.exception('Exception in request_img2img')