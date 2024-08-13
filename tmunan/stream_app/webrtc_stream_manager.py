import json
import asyncio
from typing import Dict
from uuid import UUID

import aiortc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from av import VideoFrame
from asyncer import asyncify

from tmunan.utils.log import get_logger
from tmunan.common.models import ImageParameters
from tmunan.imagine_app.client import ImagineClient


class StreamClient:

    def __init__(self, id: UUID, name: str, pc: RTCPeerConnection, data_channel: RTCDataChannel = None):

        self.id: UUID = id
        self.name: str = name
        self.pc: RTCPeerConnection = pc
        self.data_channel: RTCDataChannel = data_channel


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
        self.is_running = True

        # env
        self.logger = get_logger(self.__class__.__name__)
        self._log_output_frame = False
        self._log_input_frame = False

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

        self.logger.debug('Input - Starting _task_consume_input')
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
                if not self._log_input_frame:
                    self.logger.debug(f'Input - Put frame on input queue, qsize={self.input_frame_queue.qsize()}')
                    self._log_input_frame = True

            except asyncio.TimeoutError:
                continue

            except aiortc.mediastreams.MediaStreamError:
                self.logger.exception(f"Input - MediaStreamError in _task_consume_input")
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

            except asyncio.TimeoutError:
                pass

            except Exception:
                self.logger.exception(f"Output - Error in _task_produce_output")

        self.logger.debug('Output - Exiting _task_produce_output')

    async def recv(self):

        # self.logger.debug('Track - Executing recv() from input track')
        if self._task_input is None and self._task_output is None:
            await self.start()

        # get transformed frame from output queue
        frame = await self.output_frame_queue.get()

        # self.logger.debug('Track - Outputting frame from output queue to track')
        return frame


class WebRTCStreamManager:

    def __init__(self, imagine_host, imagine_port, imagine_secure):

        # env
        self.logger = get_logger(self.__class__.__name__)

        # tracks and media
        self.media_relay = MediaRelay()
        self.video_transform_track = VideoTransformTrack()
        self.video_transform_track.on_image_ready_callback = self.transform_frame

        # peer connections
        self.peer_connections: Dict[str, StreamClient] = dict()
        self.active_connection_name = None

        # img generation
        self.parameters = ImageParameters()
        self.img_client = None
        if imagine_host and imagine_port:
            self.img_client = ImagineClient(host=imagine_host, port=imagine_port, secure=imagine_secure)
            self.logger.info(f"ImagineClient initialized: URL={self.img_client.service_url}")
        else:
            self.logger.warning(f"ImagineClient not initialized! WebRTCStreamManager in frame loopback mode.")

    async def add_peer_connection(self, sc: StreamClient):

        # close existing
        if sc.name in self.peer_connections:
            await self.peer_connections[sc.name].pc.close()
            self.peer_connections.pop(sc.name, None)

        # add to registry
        self.peer_connections[sc.name] = sc
        self.logger.info(f"StreamClient added to peer registry: {sc.id=}, {sc.name}")

    async def set_active_peer_connection(self, name):

        # update active
        self.active_connection_name = name

        # log
        self.logger.info(f"Active StreamClient set: {self.active_connection_name}")
        sc = self.peer_connections.get(name, None)
        if sc:
            self.logger.info(f"Active client found in registry: {sc.id=}, {sc.name}")
        else:
            self.logger.warning(f"Active client not found in registry!")

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
        self.logger.info(f"StreamClient registry cleanup complete")

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
        for sc in list(self.peer_connections.values()):
            try:

                # check if data channel is connected
                if sc.data_channel:

                    # send state
                    sc.data_channel.send(json.dumps(state))

                    # log
                    self.logger.debug(f"Published state to StreamClient: {sc.id=}, {sc.name}")
                else:

                    # log
                    self.logger.warning(f"Could not publish state to StreamClient because DataChannel is not initialized: {sc.id=}, {sc.name}")

            except Exception as ex:
                self.logger.exception(f'Error while publishing state to StreamClient: {sc.id=}, {sc.name}')

    async def handle_offer(self, id: UUID, name: str, output: bool, sdp, type):

        self.logger.info(f"HandleOffer - Handling new offer: {id=}, {name=}, {output=}")

        # parse incoming offer
        offer = RTCSessionDescription(sdp=sdp, type=type)

        # create stream client
        sc = StreamClient(id=id, name=name, pc=RTCPeerConnection())

        # add to registry
        await self.add_peer_connection(sc)

        @sc.pc.on("datachannel")
        async def on_datachannel(channel):

            # save data channel
            sc.data_channel = channel
            self.logger.info(f"DataChannel - Established for StreamClient: {sc.id=}, {sc.name}")

            # check if there is no active peer
            if not self.active_connection_name:
                await self.set_active_peer_connection(sc.name)

            # publish change
            await self.publish_state()

            @channel.on("message")
            async def on_message(message):

                self.logger.debug(f'DataChannel - Message from {sc.id=}, {sc.name}: {message=}')
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])
                else:
                    await self.handle_control_message(message)

        @sc.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.logger.info(f"ConnectionState - State changed: {sc.pc.connectionState=}, {sc.id=}, {sc.name=}")

            # handle failed connection
            if sc.pc.connectionState == "failed":

                # close connection
                await sc.pc.close()

                # pop from registry
                self.peer_connections.pop(sc.name)
                self.logger.info(f"ConnectionState - StreamClient removed from registry: {sc.id=}, {sc.name=}")

                # publish state update
                await self.publish_state()

            elif sc.pc.connectionState == "connected":
                pass

            else:
                self.logger.debug(f"Unhandled connection state: {sc.pc.connectionState}")

        @sc.pc.on("track")
        async def on_track(track):
            self.logger.info(f"MediaTrack - Received Track: {track.kind=}, {sc.id=}, {sc.name=}")

            # handle video track
            if track.kind == "video":

                # check if no active peer yet
                if not self.active_connection_name:
                    await self.set_active_peer_connection(sc.name)

                # check if this is the active peer - take its track as input
                if self.active_connection_name == sc.name:

                    self.logger.info(
                        f"MediaTrack - Received Track: StreamClient matches active connection name: "
                        f"{self.active_connection_name=}, {sc.id=}, {sc.name=}")
                    self.video_transform_track.input_track = track

                # check if video feed requested
                if output:
                    sc.pc.addTrack(self.media_relay.subscribe(self.video_transform_track, buffered=False))
                    self.logger.info(f"MediaTrack - Received Track: Output track requested and added. {sc.id=}, {sc.name=}")

            @track.on("ended")
            async def on_ended():
                self.logger.info(f"MediaTrack - Track ended: {track.kind=}, {sc.id=}, {sc.name=}")

                # if track.kind == "video":
                #     await self.video_transform_track.stop()

        # handle offer
        await sc.pc.setRemoteDescription(offer)
        self.logger.info(f"HandleOffer -Remote description created: {sc.id=}, {sc.name=}")

        # create answer
        answer = await sc.pc.createAnswer()
        await sc.pc.setLocalDescription(answer)
        self.logger.info(f"HandleOffer - Answer created: {sc.id=}, {sc.name=}")

        return {
            'sdp': sc.pc.localDescription.sdp,
            'type': sc.pc.localDescription.type
        }

    async def handle_control_message(self, app_msg):

        # parse json
        app_msg = json.loads(app_msg)
        self.logger.info(f"DataChannel - Incoming message: {app_msg}")

        # extract app message
        if app_msg['type'] == "set_parameters":
            self.logger.info(f"DataChannel - Handling {app_msg['type']} control message")

            # check if parameters should be updated
            if self.parameters.prompt == '' or app_msg['payload'].get('override', False) is True:

                # update params
                self.parameters = self.parameters.model_copy(update=app_msg['payload'])
                self.logger.info(f"Parameters set: {self.parameters.model_dump()}")

                # publish state update
                await self.publish_state()

        elif app_msg['type'] == "set_active_name":
            self.logger.info(f"DataChannel - Handling {app_msg['type']} control message")

            # update active
            await self.set_active_peer_connection(app_msg['payload']['name'])

            # publish state update
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

            # loopback mode - return input frame if ImagineClient is not initialized
            if not self.img_client:
                return frame

            # serialize frame
            image = frame.to_image()
            # image = frame.to_ndarray(format='rgb24')

            # post to image generation
            new_image = self.img_client.post_image(
                image=image,
                params=self.parameters
            )

            # convert PIL back to frame
            new_frame = VideoFrame.from_image(new_image)
            # new_frame = VideoFrame.from_ndarray(new_image, format='rgb24')

            # return processed frame
            return new_frame

        except Exception as ex:
            self.logger.exception('Exception in request_img2img')
