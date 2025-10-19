from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str


class CollectedData(BaseModel):
    order_number: Optional[int] = None
    problem_category: Optional[str] = Field(None, min_length=3, max_length=50)
    problem_description: Optional[str] = Field(None, min_length=5, max_length=500)
    urgency_level: Optional[UrgencyLevel] = None

    @field_validator("urgency_level", mode="before")
    @classmethod
    def normalize_urgency_level(cls, v):
        return v.lower() if isinstance(v, str) else v


class Conversation(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    collected_data: Optional[CollectedData] = None
    created_at: datetime
    updated_at: datetime
