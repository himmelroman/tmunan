import asyncio

from fastapi import WebSocket


class WebSocketConnectionManager:
    def __init__(self):

        # set up event loop
        self.loop = asyncio.new_event_loop()

        # connection management
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    def sync_broadcast(self, message):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(
            self.broadcast(message)
        )

    def sync_send_personal_message(self, message, websocket):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(
            self.send_message(message, websocket)
        )

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message):
        for connection in self.active_connections:
            await connection.send_json(message)
