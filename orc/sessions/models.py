from pydantic import BaseModel
from typing import Dict, Optional


class SessionData(BaseModel):
    creation_time: str
    start_time: str
    end_time: Optional[str] = None


class UsageData(BaseModel):
    duration: int  # in seconds


class SessionItem(BaseModel):
    user_id: str
    session_id: str
    session_data: SessionData
    usage_data: UsageData
