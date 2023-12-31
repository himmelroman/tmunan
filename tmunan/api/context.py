import os
from pathlib import Path
from typing import Any

from fastapi import WebSocket
from pydantic_settings import BaseSettings

from tmunan.imagine.lcm import LCM


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

    async def broadcast(self, message):
        for connection in self.active_connections:
            await connection.send_json(message)


class Context(BaseSettings):

    app_name: str = "Tmunan"
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", 'tmunan')
    lcm: LCM | None = None
    ws_manager: WebSocketConnectionManager | None = None

    def __init__(self, **values: Any):
        super().__init__(**values)
        Path.mkdir(Path(self.cache_dir), parents=True, exist_ok=True)


# init app context
context = Context()
