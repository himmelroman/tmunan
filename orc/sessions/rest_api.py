from mangum import Mangum
from fastapi import FastAPI, HTTPException

from orc.sessions.db import DynamoDBSessionManager
from orc.sessions.models import SessionItem, SessionInfo, UsageData

app = FastAPI()
session_manager = DynamoDBSessionManager('YourTableName')


@app.get("/sessions/{user_id}/{session_id}")
async def get_session(user_id: str, session_id: str):
    session = session_manager.get_session(user_id, session_id)
    if session:
        return session
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions")
async def create_session(session: SessionItem):
    if session_manager.create_session(session):
        return {"message": "Session created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create session")


@app.put("/sessions/{user_id}/{session_id}")
async def update_session(user_id: str, session_id: str, info: SessionInfo, usage: UsageData):

    session = SessionItem(
        user_id=user_id,
        session_id=session_id,
        info=info,
        usage=usage
    )
    if session_manager.update_session(session):
        return {"message": "Session updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update session")

handler = Mangum(app)
