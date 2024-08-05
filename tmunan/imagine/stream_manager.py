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

    def __init__(self, id: UUID, websocket: WebSocket):

        # connection
        self.id: UUID = id
        self.websocket: WebSocket = websocket


class StreamConsumer:

    def __init__(self, id: UUID):

        # connection
        self.id: UUID = id

        # io
        self.output_queue = asyncio.Queue()


class ImageStream:

    def __init__(self):

        # consumers and connections
        self.consumers: Dict[UUID, StreamConsumer] = {}
        self.connections: Dict[UUID, WebSocketConnection] = {}
        self.active_connection: UUID | None = None

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
        self.stream: ImageStream = ImageStream()

        # events
        self.on_input_ready = Event()

        # env
        self.logger = get_logger(self.__class__.__name__)

    async def publish_state(self):

        state = {
                    "type": "state",
                    "stream": {
                        "connections": [
                            {
                                "id": str(conn.id),
                                "active": bool(conn.id == self.stream.active_connection),
                            }
                            for conn in self.stream.connections.values()
                        ],
                        "consumers": [
                            {
                                "id": str(cons.id)
                            }
                            for cons in self.stream.consumers.values()
                        ],
                        "active_connection": str(self.stream.active_connection),
                        "parameters": self.stream.param_cache.model_dump()
                    }
                }

        for conn in list(self.stream.connections.values()):
            if conn.websocket.client_state == WebSocketState.CONNECTED:
                await conn.websocket.send_json(state)
            else:
                self.stream.remove_connection(conn.id)

    async def connect(self, connection_id: UUID, websocket: WebSocket):

        # accept incoming ws connection
        self.logger.info(f"Incoming WS Connection!")
        await websocket.accept()

        # register new connection
        self.logger.info(f"New WebSocket Connection established to stream: {connection_id=}")
        ws_conn = WebSocketConnection(id=connection_id, websocket=websocket)
        self.stream.add_connection(ws_conn)

        # check if this is the first connection
        if len(self.stream.connections) == 1:
            self.stream.active_connection = connection_id

        await self.publish_state()

        return ws_conn

    async def disconnect(self, connection_id: UUID):
        if conn := self.stream.connections.get(connection_id, None):
            await conn.websocket.close()
        self.stream.remove_connection(connection_id)

        await self.publish_state()

    async def handle_websocket(self, conn_id: UUID):

        try:

            while True:

                message = await self.receive(conn_id)
                # self.logger.info(f'WS message arrived: {message}')
                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                elif message['type'] == 'json':

                    app_msg = dict(message['data'])
                    if app_msg['type'] == "parameters":
                        self.stream.param_cache = StreamInputParams(**message['data'])
                        await self.publish_state()

                    elif app_msg['type'] == "set_active":
                        if app_msg['connection_id'] in self.stream.connections.get(app_msg['connection_id']):
                            self.stream.active_connection = app_msg['connection_id']

                elif message['type'] == 'bytes':

                    if conn_id != self.stream.active_connection:
                        self.logger.warning("Non active connection sent payload...")
                        await self.send_json(conn_id, {
                            "type": "error",
                            "code": "non_active_publish"
                        })
                        continue

                    if self.stream.param_cache is None:
                        self.logger.warning('Image arrived, but params not initialized')
                        continue

                    if message['data'] is None or len(message['data']) == 0:
                        self.logger.warning('Got empty data blob')
                        continue

                    # process incoming payload
                    stream_request = copy.deepcopy(self.stream.param_cache)
                    stream_request = stream_request.model_dump()
                    stream_request['image'] = bytes_to_pil(message['data'])

                    # fire event with new imag generation request
                    self.on_input_ready.notify(stream_request)

        except Exception as e:
            self.logger.exception(f"Websocket Error on {conn_id=}")
            await self.disconnect(conn_id)

    async def handle_consumer(self, cons_id: UUID):

        try:
            # register consumer
            cons = StreamConsumer(cons_id)
            self.stream.add_consumer(cons)
            self.logger.info(f"Stream Consumer added: {cons_id=}")
            await self.publish_state()

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
            self.stream.remove_consumer(cons_id)
            self.logger.info(f"Removed Stream Consumer: {cons_id=}")
            await self.publish_state()

    def get_websocket(self, conn_id: UUID) -> WebSocket | None:
        if conn := self.stream.connections.get(conn_id, None):
            if conn.websocket.client_state == WebSocketState.CONNECTED:
                return conn.websocket

    async def send_json(self, conn_id: UUID, data: Dict):
        try:
            if websocket := self.get_websocket(conn_id):
                await websocket.send_json(data)
        except Exception as e:
            self.logger.exception(f"Error sending json to {conn_id}")

    async def receive_json(self, conn_id: UUID) -> Dict:
        try:
            if websocket := self.get_websocket(conn_id):
                return await websocket.receive_json()
        except Exception as e:
            self.logger.exception(f"Error receiving json from {conn_id}")

    async def receive_bytes(self, conn_id: UUID) -> bytes:
        try:
            if websocket := self.get_websocket(conn_id):
                return await websocket.receive_bytes()
        except Exception as e:
            self.logger.exception(f"Error receiving bytes from {conn_id}")

    async def receive(self, conn_id: UUID):

        message = None
        try:
            if websocket := self.get_websocket(conn_id):

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
