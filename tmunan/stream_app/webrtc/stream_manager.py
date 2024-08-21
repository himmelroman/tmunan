import os
import json
import asyncio
from typing import Dict
from functools import partial

import aiortc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from av import VideoFrame
from asyncer import asyncify

from tmunan.utils.log import get_logger
from tmunan.common.models import ImageParameters
from tmunan.imagine_app.client import ImagineClient
from tmunan.stream_app.webrtc.aiortc_monkey_patch import patch_vpx
from tmunan.stream_app.webrtc.signaling import AblySignalingChannel, Offer, Answer

# global shit
import faulthandler
faulthandler.enable()
patch_vpx()


class StreamClient:

    def __init__(self, id: str, name: str, pc: RTCPeerConnection, data_channel: RTCDataChannel = None):

        self.id: str = id
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


class WebRTCStreamManager:

    def __init__(self):

        # env
        self.logger = get_logger(self.__class__.__name__)

        # signaling
        self.signaling_channel_name = os.environ.get('SESSION_SIGNALING_CHANNEL', 'tmunan_dev')
        self.signaling_client = AblySignalingChannel()
        self.signaling_client.listen_to_offers(
            channel_name=self.signaling_channel_name,
            on_offer=self.handle_offer
        )

        # peers
        self.peer_connections: Dict[str, StreamClient] = dict()
        self.active_connection_name = None

        # tracks and media
        self.media_relay = MediaRelay()
        self.video_transform_track = VideoTransformTrack()
        self.video_transform_track.on_image_ready_callback = self.transform_frame

        # parameters
        self.diffusion_parameters = ImageParameters()
        self.client_parameters = dict()

        # img generation
        self.img_client = self._init_imagine_client()

    def _init_imagine_client(self):

        # check loopback mode
        if 'IMAGINE_LOOPBACK' in os.environ:
            self.logger.warning(f"ImagineClient not initialized! WebRTCStreamManager in frame loopback mode.")
            return None
        else:

            # get Imagine address
            imagine_host = os.environ.get("IMAGINE_HOST", "localhost")
            imagine_port = os.environ.get("IMAGINE_PORT", "8090")
            imagine_secure = bool(os.environ.get("IMAGINE_SECURE", False))

            # init client
            img_client = ImagineClient(host=imagine_host, port=imagine_port, secure=imagine_secure)
            self.logger.info(f"ImagineClient initialized: URL={img_client.service_url}")

            # return client
            return img_client

    async def add_peer_connection(self, sc: StreamClient):

        # close existing
        if sc.name in self.peer_connections:
            await self.peer_connections[sc.name].pc.close()
            self.peer_connections.pop(sc.name, None)

        # add to registry
        self.peer_connections[sc.name] = sc
        self.logger.info(f"StreamClient added to peer registry: {sc.id=}, {sc.name}")

    def set_active_peer_connection(self, name):

        # update active
        self.active_connection_name = name
        self.logger.info(f"Active StreamClient set: {self.active_connection_name}")

        # consumer track
        self.consume_active_peer_track()

        # publish presence
        self.publish_presence()

    def consume_active_peer_track(self):

        # get active client
        if sc := self.peer_connections.get(self.active_connection_name, None):

            # get incoming video track
            video_track = next((recv.track for recv in sc.pc.getReceivers() if recv.track.kind == 'video'), None)
            if video_track:
                self.logger.info(f"Consuming video track from client: {sc.id=}, {sc.name}")
                self.video_transform_track.input_track = video_track
            else:
                self.logger.warning(f"Cannot find video track in PeerConnection for client: {sc.id=}, {sc.name}")
        else:
            self.logger.warning(f"Active client not found in registry!")

    async def on_datachannel(self, sc: StreamClient, channel: RTCDataChannel):

        # save data channel
        sc.data_channel = channel
        sc.data_channel.on('message', partial(self.on_datachannel_message, sc))
        self.logger.info(f"DataChannel - Established for StreamClient: {sc.id=}, {sc.name=}")

        # check if there is no active peer
        if not self.active_connection_name:
            self.set_active_peer_connection(sc.name)

        # publish change
        self.publish_presence(self.get_client_list(include_peer_list=[sc.name]))
        self.publish_parameters(self.get_client_list(include_peer_list=[sc.name]))

    async def on_datachannel_message(self, sc: StreamClient, message):

        self.logger.debug(f'DataChannel - Message from {sc.id=}, {sc.name}: {message=}')
        await self.handle_control_message(message, sc)

    async def on_connectionstatechange(self, sc: StreamClient):

        # log state change
        self.logger.info(f"ConnectionState - State changed: {sc.pc.connectionState=}, {sc.id=}, {sc.name=}")

        # handle failed connection
        if sc.pc.connectionState in ["new", "connected", "connecting"]:

            # update presence
            self.publish_presence()

        elif sc.pc.connectionState in ["failed", "disconnected", "closed"]:

            # get peer connection
            if spc := self.peer_connections.pop(sc.name, None):

                # close connection (this should be harmless if already closed)
                await spc.pc.close()

                # log
                self.logger.info(f"ConnectionState - StreamClient removed from registry: {spc.id=}, {spc.name=}")

                # update presence
                self.publish_presence()

        else:
            self.logger.warning(f"Unhandled connection state: {sc.pc.connectionState}")

    def on_track(self, sc: StreamClient, track: MediaStreamTrack):

        # log
        self.logger.info(f"MediaTrack - Received Track: {track.kind=}, {sc.id=}, {sc.name=}")

        # handle video track
        if track.kind == "video":

            # no active peer, this is the first connection
            if not self.active_connection_name:
                self.set_active_peer_connection(sc.name)

            # new connection from already active peer
            elif self.active_connection_name == sc.name:
                self.consume_active_peer_track()

        @track.on("ended")
        async def on_ended():
            self.logger.info(f"MediaTrack - Track ended: {track.kind=}, {sc.id=}, {sc.name=}")

    async def cleanup(self):

        # collection stop() coroutines all peer connections
        task_list = [spc.pc.close() for spc in list(self.peer_connections.values())]

        # add video transform track to stop_tasks() list
        if self.video_transform_track:
            task_list.append(self.video_transform_track.stop_tasks())

        # join all peer close coroutines
        await asyncio.gather(*task_list)

        # clear peer dict
        self.peer_connections.clear()
        self.logger.info(f"StreamClient registry cleanup complete")

    def get_client_list(self, include_peer_list=None, exclude_peer_list=None):

        # include
        if include_peer_list:
            return [c for c in self.peer_connections.values() if c.name in include_peer_list]

        # exclude
        elif exclude_peer_list:
            return [c for c in self.peer_connections.values() if c.name not in exclude_peer_list]

        # all
        else:
            return list(self.peer_connections.values())

    def _publish(self, message, client_list=None):

        # init client list if empty
        if not client_list:
            client_list = self.get_client_list()

        # log
        self.logger.info(f"Publishing message to {[c.name for c in client_list]}: {message=}")

        # iterate all stream peer connections
        for sc in client_list:
            try:

                # check if data channel is connected
                if sc.data_channel:

                    # send state
                    sc.data_channel.send(json.dumps(message))

                    # log
                    self.logger.debug(f"Sent message to StreamClient: {sc.id=}, {sc.name=}")

                else:

                    # log
                    self.logger.warning(
                        f"Could not send message because DataChannel is not initialized: {sc.id=}, {sc.name=}")

            except Exception as ex:
                self.logger.exception(f'Error while sending message: {sc.id=}, {sc.name=}')

    def publish_presence(self, client_list=None):

        presence = {
                    "type": "presence",
                    "payload": {
                        "connections": [
                            {
                                "id": str(spc.id),
                                "name": spc.name
                            }
                            for spc in self.peer_connections.values()
                        ],
                        "active_connection_name": self.active_connection_name
                    }
                }

        # publish
        self._publish(presence, client_list)

    def publish_parameters(self, client_list=None):

        parameters = {
                    "type": "parameters",
                    "payload": {
                        "diffusion": self.diffusion_parameters.model_dump(),
                        "client": self.client_parameters
                    }
                }

        # publish
        self._publish(parameters, client_list)

    async def handle_offer(self, offer: Offer) -> Answer:

        self.logger.info(f"HandleOffer - Handling new offer: {offer.id=}, {offer.name=}, {offer.output=}")

        # parse incoming offer
        remote_peer_sdp = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

        # create stream client
        pc = RTCPeerConnection(
            configuration=RTCConfiguration([RTCIceServer("stun:stun.l.google:19302")])
        )
        sc = StreamClient(id=offer.id, name=offer.name, pc=pc)

        # add to registry
        await self.add_peer_connection(sc)

        # register events
        sc.pc.on("track", partial(self.on_track, sc))
        sc.pc.on("datachannel", partial(self.on_datachannel, sc))
        sc.pc.on("connectionstatechange", partial(self.on_connectionstatechange, sc))

        # check if video feed requested
        if offer.output:
            sc.pc.addTrack(self.media_relay.subscribe(self.video_transform_track, buffered=False))
            self.logger.info(f"HandleOffer - Output track requested and added. {sc.id=}, {sc.name=}")

        # handle offer
        await sc.pc.setRemoteDescription(remote_peer_sdp)
        self.logger.info(f"HandleOffer - Remote description created: {sc.id=}, {sc.name=}")

        # create answer
        answer = await sc.pc.createAnswer()
        await sc.pc.setLocalDescription(answer)
        self.logger.info(f"HandleOffer - Answer created: {sc.id=}, {sc.name=}")

        return Answer(
            id=offer.id,
            name=offer.name,
            sdp=sc.pc.localDescription.sdp,
            type=sc.pc.localDescription.type
        )

    async def handle_control_message(self, app_msg, sc):

        # log
        self.logger.info(f"DataChannel - Incoming message from {sc.name}: {app_msg}")

        try:

            # parse json
            app_msg = json.loads(app_msg)

        except json.decoder.JSONDecodeError:
            self.logger.warning(f"DataChannel - Message is not a valid JSON, discarding.")
            return

        # extract app message
        if app_msg['type'] == "parameters":

            # check if parameters should be updated
            if self.diffusion_parameters.prompt == '' or app_msg['payload'].get('override', False) is True:

                # params updated flag
                updated = False

                # check for diffusion params update
                if 'diffusion' in app_msg['payload']:

                    # update params
                    updated = True
                    self.diffusion_parameters = self.diffusion_parameters.model_copy(update=app_msg['payload']['diffusion'])
                    self.logger.info(f"Parameters set: {self.diffusion_parameters.model_dump()}")

                # check for client params update
                if 'client' in app_msg['payload']:

                    # update params
                    updated = True
                    self.client_parameters.update(app_msg['payload']['client'])

                # publish params update
                if updated:
                    self.publish_parameters()

        elif app_msg['type'] == "set_active_name":

            # update active
            self.set_active_peer_connection(app_msg['payload']['name'])

        else:
            self.logger.info(f"Unrecognized message: {app_msg}")

    async def transform_frame(self, frame):

        # gen image
        # self.logger.info('TRANS - Sending frame to img2img')
        new_frame = await asyncify(self.request_img2img)(frame)
        if new_frame:

            # assign timestamps
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

        else:

            self.logger.warning(f"ImagineClient - Request to img2img failed, returning original frame.")
            return frame

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
                params=self.diffusion_parameters
            )

            # convert PIL back to frame
            new_frame = VideoFrame.from_image(new_image)
            # new_frame = VideoFrame.from_ndarray(new_image, format='rgb24')

            # return processed frame
            return new_frame

        except ConnectionRefusedError:
            self.logger.error(f"Error in request_img2img: ConnectionRefusedError! {self.img_client.service_url=}")

        except Exception:
            self.logger.exception('Exception in request_img2img')
