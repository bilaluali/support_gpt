from typing import Optional
from pydantic import BaseModel, Field

from storage.models import CollectedData


class OpenAIResponse(BaseModel):
    reply: str
    collected_data: CollectedData


class ChatRequest(BaseModel):
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's text message to the AI agent",
    )
    transaction_id: Optional[str] = Field(
        default=None,
        description="Unique transaction identifier for the conversation",
    )


class ChatResponse(BaseModel):
    transaction_id: str
    response: str
    collected_data: CollectedData


class ChatSummaryRequest(BaseModel):
    transaction_id: str


class ChatSummaryResponse(BaseModel):
    summary: str
    collected_data: CollectedData
