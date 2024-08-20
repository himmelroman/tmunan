import uuid

from fastapi import Request, WebSocket, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from tmunan.stream_app.webrtc.signaling import Offer

router = APIRouter()


@router.websocket("/api/ws")
async def websocket(name: str, ws: WebSocket):

    # accept incoming ws connection
    await ws.accept()

    connection_id = str(uuid.uuid4())
    try:
        await ws.state.stream_manager.connect(name=name, connection_id=connection_id, websocket=ws)
        await ws.state.stream_manager.handle_websocket(connection_id)

    finally:
        await ws.state.stream_manager.disconnect(connection_id)


@router.get("/api/stream")
async def stream(req: Request):

    # consume stream
    consumer_id = str(uuid.uuid4())
    return StreamingResponse(
        req.state.stream_manager.handle_consumer(consumer_id),
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/api/offer")
async def offer(name: str, output: bool, req: Request):

    # get offer
    request_params = await req.json()

    # assign id
    peer_id = str(uuid.uuid4())

    # prepare offer
    offer = Offer(
        id=peer_id,
        name=name,
        output=output,
        sdp=request_params["sdp"],
        type=request_params["type"]
    )

    # handle offer
    answer = await req.state.stream_manager.handle_offer(offer)

    return JSONResponse(
        {"sdp": answer.sdp, "type": answer.type},
    )
