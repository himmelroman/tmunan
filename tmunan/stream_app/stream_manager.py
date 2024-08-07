import os
import time
import copy
import json
import typing
import asyncio
from queue import Queue

from uuid import UUID
from typing import Dict
from json import JSONDecodeError

from websockets.exceptions import ConnectionClosedOK
from fastapi.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from tmunan.utils.log import get_logger
from tmunan.utils.image import bytes_to_pil
from tmunan.utils.fixed_size_queue import FixedSizeQueue
from tmunan.common.models import ImageParameters
from tmunan.imagine_app.client import ImagineClient


class ServerFullException(Exception):
    """Exception raised when the server is full."""
    pass


class WebSocketConnection:

    def __init__(self, id: UUID, websocket: WebSocket, info: dict = None):

        self.id: UUID = id
        self.info = info
        self.websocket: WebSocket = websocket


class StreamConsumer:

    def __init__(self, id: UUID):

        self.id: UUID = id

        # io
        self.output_queue = asyncio.Queue()


class ImageStream:

    def __init__(self):

        # consumers and connections
        self.consumers: Dict[UUID, StreamConsumer] = {}
        self.connections: Dict[UUID, WebSocketConnection] = {}
        self.active_connection_name: str | None = None

        # params cache
        self.parameters = ImageParameters()

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

    def distribute_output(self, req_time, data):

        # iterate all connection registered to this stream
        for cons_id, cons in list(self.consumers.items()):

            # enqueue output
            cons.output_queue.put_nowait((req_time, data))


class StreamManager:

    def __init__(self, max_streams=1):

        # streams
        self.max_streams = max_streams
        self.stream: ImageStream = ImageStream()
        self.input_queue: Queue = FixedSizeQueue(max_size=2)

        # img generation
        host = os.environ.get("IMAGINE_HOST", "localhost")
        port = os.environ.get("IMAGINE_PORT", "8090")
        secure = bool(os.environ.get("IMAGINE_SECURE", False))
        self.img_client = ImagineClient(host=host, port=port, secure=secure)
        self.img_client.on_image_ready += self.stream.distribute_output
        self.img_client.watch_queue(self.input_queue)

        # env
        self.logger = get_logger(self.__class__.__name__)

    async def publish_welcome(self, conn: WebSocketConnection):

        welcome = {
            "type": "connected",
            "payload": {
                "id": str(conn.id),
                "info": conn.info,
                "active": conn.info.get('name', '') == self.stream.active_connection_name
            }
        }

        await conn.websocket.send_json(welcome)

    async def publish_state(self):

        state = {
                    "type": "state",
                    "payload": {
                        "connections": [
                            {
                                "id": str(conn.id),
                                "info": conn.info,
                                "active": conn.info.get('name', '') == self.stream.active_connection_name,
                            }
                            for conn in self.stream.connections.values()
                        ],
                        "consumers": [
                            {
                                "id": str(cons.id)
                            }
                            for cons in self.stream.consumers.values()
                        ],
                        "active_connection_name": self.stream.active_connection_name,
                        "parameters": self.stream.parameters.model_dump()
                    }
                }

        for conn in list(self.stream.connections.values()):
            try:
                await conn.websocket.send_json(state)

            except (ConnectionClosedOK, WebSocketDisconnect):
                self.stream.remove_connection(conn.id)

    async def connect(self, name: str, connection_id: UUID, websocket: WebSocket):

        # register new connection
        self.logger.info(f"WebSocket connected: {connection_id=}, host={websocket.client.host}")
        ws_conn = WebSocketConnection(
            id=connection_id,
            info={'name': name, 'host': websocket.client.host},
            websocket=websocket
        )
        self.stream.add_connection(ws_conn)

        # check if this is the first connection
        if len(self.stream.connections) == 1:
            self.stream.active_connection_name = name

        await self.publish_welcome(ws_conn)
        await self.publish_state()

        return ws_conn

    async def disconnect(self, connection_id: UUID):

        # deregister connection
        if conn := self.stream.connections.get(connection_id, None):
            self.stream.remove_connection(connection_id)
            self.logger.info(f'WebSocket disconnected: {connection_id=}, {conn.info}')

        await self.publish_state()

    async def handle_websocket(self, conn_id: UUID):

        try:

            while True:

                socket_message = await self.receive(conn_id)
                if socket_message is None:
                    await asyncio.sleep(0.01)
                    continue

                # check for JSON messages
                elif socket_message['type'] == 'json':

                    # parse and handle message
                    app_msg = dict(socket_message['data'])
                    await self.handle_json_message(app_msg)

                elif socket_message['type'] == 'bytes':

                    if conn := self.stream.connections.get(conn_id, None):
                        if conn.info['name'] != self.stream.active_connection_name:
                            self.logger.warning("Non active connection sent payload...")
                            await self.send_json(conn_id, {
                                "type": "error",
                                "payload": {
                                    "code": "non_active_publish"
                                }
                            })
                            continue

                    if socket_message['data'] is None or len(socket_message['data']) == 0:
                        self.logger.warning('Got empty data blob')
                        continue

                    # parse and handle message
                    app_msg = socket_message['data']
                    await self.handle_bytes_message(app_msg)

        except WebSocketDisconnect as disconnect_ex:
            pass

        except Exception as e:
            self.logger.exception(f"Websocket Error on {conn_id=}")

    async def handle_json_message(self, app_msg):

        # extract app message
        if app_msg['type'] == "set_parameters":
            if self.stream.parameters.prompt == '' or app_msg['payload'].get('override', False) is True:
                self.stream.parameters = self.stream.parameters.model_copy(update=app_msg['payload'])
                await self.publish_state()
                self.logger.info(f"Parameters set: {self.stream.parameters.model_dump()}")

        elif app_msg['type'] == "set_active_name":

            self.stream.active_connection_name = app_msg['payload']['name']
            await self.publish_state()
            self.logger.info(f"Active connection set: {self.stream.active_connection_name}")

        elif app_msg['type'] == "set_connection_info":
            if conn := self.stream.connections.get(UUID(app_msg['payload']['connection_id'])):
                conn.info.update(app_msg['payload']['info'])
                await self.publish_state()
                self.logger.info(f"Connection info updated: {conn.id=} - {conn.info}")

    async def handle_bytes_message(self, app_msg):

        # process incoming payload
        stream_request = copy.deepcopy(self.stream.parameters)
        stream_request = stream_request.model_dump()
        stream_request['image'] = bytes_to_pil(app_msg)
        stream_request['timestamp'] = time.time()

        # fire event with new imag generation request
        self.logger.info(f"Enqueue request at: {stream_request['timestamp']}")
        self.input_queue.put(stream_request)

    async def handle_consumer(self, cons_id: UUID):

        try:
            # register consumer
            cons = StreamConsumer(cons_id)
            self.stream.add_consumer(cons)
            self.logger.info(f"Stream Consumer added: {cons_id=}")
            await self.publish_state()

            while True:

                # read from output queue
                req_time, frame = await cons.output_queue.get()
                # if frame is None:
                #     await asyncio.sleep(0.01)
                #     continue

                # convert image to multipart frame
                self.logger.info(f'Sending to consumer {cons_id=}, image which was requested {time.time() - req_time} ago')
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

        if websocket := self.get_websocket(conn_id):

            # try to receive a message
            message = await websocket.receive()
            if message is None:
                return

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message["code"], message.get("reason"))

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
