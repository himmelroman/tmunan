from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UsageData(BaseModel):
    duration: int = 0   # in seconds

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionItem(BaseModel):
    user_id: str
    session_id: str
    created_at: Optional[datetime] = None
    usage_data: Optional[UsageData] = None
