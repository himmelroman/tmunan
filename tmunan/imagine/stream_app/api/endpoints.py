import uuid

from fastapi import WebSocket
from fastapi import Request, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from tmunan.imagine.stream_app.stream_manager import ServerFullException

router = APIRouter()


@router.websocket("/api/ws")
async def websocket(name: str, ws: WebSocket, req: Request):

    # accept incoming ws connection
    await ws.accept()

    connection_id = uuid.uuid4()
    try:
        await req.state.stream_manager.connect(name=name, connection_id=connection_id, websocket=ws)
        await req.state.stream_manager.handle_websocket(connection_id)

    finally:
        await req.state.stream_manager.disconnect(connection_id)


@router.get("/api/stream")
async def stream(req: Request):

    try:

        # consume stream
        consumer_id = uuid.uuid4()
        return StreamingResponse(
            req.state.stream_manager.handle_consumer(consumer_id),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers={"Cache-Control": "no-cache"},
        )
    except ServerFullException as ex:
        return HTTPException(503)
