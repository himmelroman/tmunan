import asyncio
import json
import logging
import typing

from uuid import UUID
from typing import Dict, Union
from json import JSONDecodeError

from fastapi import WebSocket
from starlette.websockets import WebSocketState


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class AsyncFixedSizeQueue(asyncio.Queue):
    """An asynchronous queue with a fixed size of 1 that overwrites on put."""

    async def put(self, item):
        """Overrides the default put method. If full, empty the queue asynchronously."""

        while not self.empty():
            try:
                await self.get()  # Discard the existing item if possible
            except asyncio.QueueEmpty:
                break

        await super().put(item)  # Add the new item
        print(f'Put success!, items in queue: {self.qsize()}')


Connections = Dict[UUID, Dict[str, Union[WebSocket, AsyncFixedSizeQueue]]]


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(
            self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0
    ):
        await websocket.accept()

        # user_count = self.get_user_count()
        # print(f"User count: {user_count}")
        # if max_queue_size > 0 and user_count >= max_queue_size:
        #     print("Server is full")
        #     await websocket.send_json({"status": "error", "message": "Server is full"})
        #     await websocket.close()
        #     raise ServerFullException("Server is full")

        print(f"New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            # "queue": asyncio.Queue(),
            "queue": AsyncFixedSizeQueue()
        }
        await websocket.send_json({"status": "connected", "message": "Connected"})

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            await queue.put(new_data)

    async def get_latest_data(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                return await queue.get()
            except asyncio.QueueEmpty:
                return None

    def delete_user(self, user_id: UUID):
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            if websocket.client_state == WebSocketState.CONNECTED:
                return user_session["websocket"]
        return None

    async def disconnect(self, user_id: UUID):
        websocket = self.get_websocket(user_id)
        if websocket:
            await websocket.close()
        self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: Dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.send_json(data)
        except Exception as e:
            logging.error(f"Error: Send json: {e}")

    async def receive_json(self, user_id: UUID) -> Dict:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_json()
        except Exception as e:
            logging.error(f"Error: Receive json: {e}")

    async def receive_bytes(self, user_id: UUID) -> bytes:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_bytes()
        except Exception as e:
            logging.error(f"Error: Receive bytes: {e}")

    async def receive(self, user_id: UUID):

        message = None
        try:
            websocket = self.get_websocket(user_id)
            if websocket:

                message = await websocket.receive()
                if message is None:
                    return

                # check if text was sent
                if message.get('text', None) is not None:
                    try:
                        return {
                            'type': 'json',
                            'data': json.loads(message["text"])
                        }
                    except JSONDecodeError as not_json:
                        pass

                # check if bytes were sent
                elif message.get('bytes', None) is not None:
                    return {
                        'type': 'bytes',
                        'data': typing.cast(bytes, message["bytes"])
                    }

        except Exception as e:
            logging.exception(f"Error: In Receive! {message=}")
