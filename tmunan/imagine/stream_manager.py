import copy
import json
import typing
import asyncio
import logging
import uuid

from uuid import UUID
from typing import Dict
from json import JSONDecodeError

from fastapi.websockets import WebSocket, WebSocketState

from tmunan.common.event import Event
from tmunan.common.log import get_logger
from tmunan.common.fixed_size_queue import AsyncFixedSizeQueue
from tmunan.imagine.common.image_utils import bytes_to_pil, bytes_to_frame, pil_to_bytes
from tmunan.imagine.common.pydantic_models import StreamInputParams


class ServerFullException(Exception):
    """Exception raised when the server is full."""
    pass


class WebSocketConnection:

    def __init__(self, id: UUID, stream_id: UUID, websocket: WebSocket):

        # connection
        self.id: UUID = id
        self.stream_id: UUID = stream_id
        self.websocket: WebSocket = websocket


class StreamConsumer:

    def __init__(self, id: UUID, stream_id: UUID):

        # connection
        self.id: UUID = id
        self.stream_id: UUID = stream_id

        # io
        self.output_queue = asyncio.Queue()


class ImageStream:

    def __init__(self):

        # io
        self.input_queue = AsyncFixedSizeQueue(maxsize=1)

        # consumers and connections
        self.consumers: Dict[UUID, StreamConsumer] = {}
        self.connections: Dict[UUID, WebSocketConnection] = {}

        # params cache
        default_params = {
            'prompt': 'Lions in the sky',
            'strength': 1.0
        }
        self.param_cache = StreamInputParams(**default_params)

    @property
    def has_consumers(self) -> bool:
        return bool(self.consumers)

    @property
    def has_connections(self) -> bool:
        return bool(self.connections)

    def add_consumer(self, cons: StreamConsumer):
        self.consumers[cons.id] = cons

    def remove_consumer(self, cons_id: UUID):
        self.consumers.pop(cons_id, None)

    def add_connection(self, conn: WebSocketConnection):
        self.connections[conn.id] = conn

    def remove_connection(self, conn_id: UUID):
        self.connections.pop(conn_id, None)

    def distribute_output(self, data):

        # iterate all connection registered to this stream
        for cons_id, cons in list(self.consumers.items()):

            # enqueue output
            cons.output_queue.put_nowait(data)


class StreamManager:

    def __init__(self, max_streams=1):

        # streams
        self.max_streams = max_streams
        self.streams: Dict[UUID, ImageStream] = {}

        # events
        self.on_input_ready = Event()

        # env
        self.logger = get_logger(self.__class__.__name__)

    def get_stream(self, stream_id):

        self.streams[stream_id] = self.streams.get(stream_id, ImageStream())
        return self.streams[stream_id]

    @property
    def stream_count(self) -> int:
        return len(self.streams)

    async def connect(self, stream_id: UUID, connection_id: UUID, websocket: WebSocket):

        # accept incoming ws connection
        self.logger.info(f"Incoming WS Connection!")
        await websocket.accept()

        # check if this is a new stream
        if stream_id not in self.streams:

            # check max concurrent streams
            if self.stream_count >= self.max_streams:

                self.logger.info(f"Server is full! Dropping connection")
                await websocket.send_json({"status": "error", "message": "Server is full"})
                await websocket.close()
                raise ServerFullException("Server is full")

        # register new connection
        self.logger.info(f"New WebSocket Connection established to stream: {stream_id}")
        ws_conn = WebSocketConnection(id=connection_id, stream_id=stream_id, websocket=websocket)
        self.streams[stream_id].add_connection(ws_conn)

        # send connection ack
        await websocket.send_json({"status": "connected", "message": "Connected"})

        return ws_conn

    async def disconnect(self, stream_id: UUID, connection_id: UUID):
        if stream := self.streams.pop(stream_id, None):
            if conn := stream.connections.get(connection_id, None):
                await conn.websocket.close()
            stream.remove_connection(connection_id)

    async def handle_websocket(self, stream_id: UUID, conn_id: UUID):

        try:

            while True:

                message = await self.receive(stream_id, conn_id)
                # self.logger.info(f'WS message arrived: {message}')
                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                elif message['type'] == 'json':

                    self.streams[stream_id].param_cache = StreamInputParams(**message['data'])
                    # self.logger.info(f'WS message arrived: {message}')

                elif message['type'] == 'bytes':

                    if self.streams[stream_id].param_cache is None:
                        self.logger.warning('Image arrived, but params not initialized')
                        continue

                    if message['data'] is None or len(message['data']) == 0:
                        self.logger.warning('Got empty data blob')
                        continue

                    stream_request = copy.deepcopy(self.streams[stream_id].param_cache)
                    stream_request = stream_request.model_dump()
                    stream_request['image'] = bytes_to_pil(message['data'])

                    # enqueue image on stream input queue
                    if stream := self.streams[stream_id]:
                        # self.logger.info(f'WS message, put on queue')
                        await stream.input_queue.put(stream_request)

                        # TODO: This is ugly, we notify here for someone else to take the item from the queue
                        self.on_input_ready.notify(stream_id)

        except Exception as e:
            self.logger.exception(f"Websocket Error on {stream_id=}, {conn_id=} - {e}")
            await self.disconnect(stream_id, conn_id)

    async def enqueue_request(self, stream_id: UUID, request_data):

        # enqueue new request on specified connection's queue
        if stream := self.streams.get(stream_id):
            await stream.input_queue.put(request_data)

    async def dequeue_request(self, stream_id: UUID):

        if stream := self.streams.get(stream_id):
            try:
                return await stream.input_queue.get()
            except asyncio.QueueEmpty:
                return None

    async def handle_consumer(self, stream_id: UUID, cons_id: UUID):

        # get relevant stream
        if stream := self.streams.get(stream_id, None):
            try:
                # register consumer
                cons = StreamConsumer(cons_id, stream_id)
                stream.add_consumer(cons)
                self.logger.info(f"Incoming Stream Consumer: {stream_id=}{cons_id=}")

                while True:

                    # read from output queue
                    image = await cons.output_queue.get()
                    if image is None:
                        await asyncio.sleep(0.01)
                        continue

                    # convert image to multipart frame
                    frame = bytes_to_frame(pil_to_bytes(image, format='WEBP'))
                    yield frame

            finally:
                # de-register consumer
                stream.remove_consumer(cons_id)
                self.logger.info(f"Removed Stream Consumer: {stream_id=}{cons_id=}")

    def get_websocket(self, stream_id: UUID, conn_id: UUID) -> WebSocket | None:
        if stream := self.streams.get(stream_id):
            if conn := stream.connections.get(conn_id, None):
                if conn.websocket.client_state == WebSocketState.CONNECTED:
                    return conn.websocket

    async def send_json(self, stream_id: UUID, conn_id: UUID, data: Dict):
        try:
            if websocket := self.get_websocket(stream_id, conn_id):
                await websocket.send_json(data)
        except Exception as e:
            self.logger.exception(f"Error sending json to {stream_id}-{conn_id}")

    async def receive_json(self, stream_id: UUID, conn_id: UUID) -> Dict:
        try:
            if websocket := self.get_websocket(stream_id, conn_id):
                return await websocket.receive_json()
        except Exception as e:
            self.logger.exception(f"Error receiving json from {stream_id}-{conn_id}")

    async def receive_bytes(self, stream_id: UUID, conn_id: UUID) -> bytes:
        try:
            if websocket := self.get_websocket(stream_id, conn_id):
                return await websocket.receive_bytes()
        except Exception as e:
            self.logger.exception(f"Error receiving bytes from {stream_id}-{conn_id}")

    async def receive(self, stream_id: UUID, conn_id: UUID):

        message = None
        try:
            if websocket := self.get_websocket(stream_id, conn_id):

                # try to receive a message
                message = await websocket.receive()
                if message is None:
                    return

                # check if received message is text
                if message.get('text', None) is not None:
                    try:
                        return {
                            'type': 'json',
                            'data': json.loads(message["text"])
                        }
                    except JSONDecodeError as not_json:
                        pass

                # check if received message is bytes
                elif message.get('bytes', None) is not None:
                    return {
                        'type': 'bytes',
                        'data': typing.cast(bytes, message["bytes"])
                    }

        except Exception as e:
            self.logger.exception(f"Error in Receive! {message=}")
