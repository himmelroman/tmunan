import os

from fastapi import WebSocket
from pydantic_settings import BaseSettings

from tmunan.imagine.lcm_large import LCMLarge


class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class Context(BaseSettings):
    app_name: str = "Tmunan"
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache")
    lcm: LCMLarge | None = None
    ws_manager: WebSocketConnectionManager | None = None

