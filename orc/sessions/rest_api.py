import logging
import os

from mangum import Mangum
from fastapi import FastAPI, Request, HTTPException

from orc.sessions.launch import launch_session
from orc.sessions.models import UsageData, SessionItem
from orc.sessions.db import DynamoDBSessionManager

app = FastAPI()
session_manager = DynamoDBSessionManager(os.environ['DYNAMODB_TABLE'])

logger = logging.getLogger()
logger.setLevel("INFO")


@app.get("/users/me")
async def read_user_me(request: Request):

    # Access the request context, specifically the authorizer context
    claims = request.scope["aws.event"]["requestContext"]["authorizer"]

    # You can now access claims like user_id, email, etc.
    user_id = claims.get("user_id")
    email = claims.get("email")

    return {'claims': claims, "user_id": user_id, "email": email}


@app.get("/sessions/{user_id}/{session_id}")
async def get_session(user_id: str, session_id: str):
    session = session_manager.get_session(user_id, session_id)
    if session:
        return session
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions")
async def create_session(session: SessionItem):
    if session_manager.create_session(session.user_id, session.session_id):
        return {"message": "Session created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create session")


@app.post("/sessions/{user_id}/{session_id}/launch")
async def update_session(user_id: str, session_id: str):

    # get session
    session = session_manager.get_session(user_id, session_id)
    if session:

        # generate signaling channel
        signaling_channel = f'tmunan_session_{session_id}'

        # launch the session
        return launch_session(user_id, session_id, signaling_channel)
    else:
        raise HTTPException(status_code=400, detail="Failed to update session usage")


@app.put("/sessions/{user_id}/{session_id}/usage")
async def update_session(user_id: str, session_id: str, usage_data: UsageData):
    if session_manager.update_session_usage(user_id, session_id, usage_data):
        return {"message": "Session usage updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update session usage")


handler = Mangum(app)
