import logging
import re
import uuid

from fastapi import APIRouter, HTTPException, Request
from openai.types.chat import (
    ChatCompletionAssistantMessageParam as OpenAIAssistantMessage,
)
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
from openai.types.chat import ChatCompletionUserMessageParam as OpenAIUserMessage
from pydantic import ValidationError

from chat.models import (
    ChatRequest,
    ChatResponse,
    ChatSummaryRequest,
    ChatSummaryResponse,
    OpenAIResponse,
)
from chat.prompts import CHAT_SYSTEM_MESSAGE, CHAT_SUMMARY_SYSTEM_MESSAGE
from config import limiter, openai_client, storage
from storage.models import CollectedData, Message, MessageRole

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


def parse_message(message: Message) -> OpenAIMessage:
    """
    Parse a Message object into an OpenAI message.
    """
    if message.role == MessageRole.USER:
        return OpenAIUserMessage(role="user", content=message.content)
    elif message.role == MessageRole.ASSISTANT:
        return OpenAIAssistantMessage(role="assistant", content=message.content)
    else:
        raise ValueError(f"Invalid message role: {message.role}")


def parse_response(response_content: str) -> OpenAIResponse:
    """
    Extract <COLLECTED_DATA> block from assistant response and parse it into
    an OpenAIResponse object.
    """

    match = re.search(
        r"<COLLECTED_DATA>(.*?)</COLLECTED_DATA>", response_content, re.DOTALL
    )
    collected_json = match.group(1).strip() if match else None

    # Initialize with defaults
    reply = (
        response_content.strip()
        if not collected_json
        else response_content.replace(match.group(0), "").strip()
    )
    openai_response = OpenAIResponse(reply=reply, collected_data=CollectedData())

    if not collected_json:
        logger.warning("No <COLLECTED_DATA> block found in response.")
        return openai_response

    try:
        openai_response.collected_data = CollectedData.model_validate_json(
            collected_json
        )
    except ValidationError as e:
        logger.error(f"Invalid JSON in <COLLECTED_DATA>: {collected_json} | Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing <COLLECTED_DATA>: {e}")

    return openai_response


def update_collected_data(
    collected_data: CollectedData, new_collected_data: CollectedData
) -> CollectedData:
    """
    Incrementally build up the conversation's collected data object as new information is provided
    by the user over time. This function merges the latest non-None values from the new collected data
    into the existing collected data, so fields are gradually filled in as they are obtained.
    """
    return collected_data.model_copy(
        update=new_collected_data.model_dump(exclude_none=True)
    )


@router.post("/chat")
@limiter.limit("10/minute")  # Max 10 requests per minute per IP
def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    POST endpoint to generate a response from the LLM for the given user message.
    """
    # 1. Clean message input
    user_message = chat_request.user_message.strip()
    if not user_message:
        raise HTTPException(400, "Message cannot be empty.")

    # 2. Avoid offensive content
    if openai_client.is_offensive_content(user_message):
        raise HTTPException(400, "Message contains offensive content.")

    # 3. Get transaction ID from request or generate a new one
    transaction_id = chat_request.transaction_id or str(uuid.uuid4())

    try:
        # 4. Get conversation history
        conversation = storage.get_or_create_conversation(transaction_id)

        # 5. Generate response from LLM
        messages = [
            CHAT_SYSTEM_MESSAGE,
            *[parse_message(message) for message in conversation.messages],
            parse_message(Message(role=MessageRole.USER, content=user_message)),
        ]
        response_content = openai_client.create_chat_completion(messages=messages)

        # 6. Extract order data from response
        openai_response = parse_response(response_content)
        conversation.collected_data = update_collected_data(
            conversation.collected_data, openai_response.collected_data
        )

        # 7. Append new pair user-assistant messages to conversation
        new_messages = [
            Message(role=MessageRole.USER, content=user_message),
            Message(role=MessageRole.ASSISTANT, content=openai_response.reply),
        ]
        conversation.messages.extend(new_messages)

        # 8. Update conversation in storage
        storage.update_conversation(transaction_id, conversation)

        return ChatResponse(
            transaction_id=transaction_id,
            response=openai_response.reply,
            collected_data=conversation.collected_data,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate response: {str(e)}")


@router.post("/chat/summary")
def chat_summary(
    request: Request, chat_summary_request: ChatSummaryRequest
) -> ChatSummaryResponse:
    """
    POST endpoint to get a summary of the conversation history.
    """
    # 1. Avoid empty transaction ID
    transaction_id = chat_summary_request.transaction_id.strip()
    if not transaction_id:
        raise HTTPException(400, "Transaction ID cannot be empty.")

    # 2. Retrieve conversation if it exists
    conversation = storage.get_conversation(transaction_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found.")

    try:
        # 3. Generate summary from LLM
        history_messages = [parse_message(message) for message in conversation.messages]
        response_content = openai_client.create_chat_completion(
            messages=[CHAT_SUMMARY_SYSTEM_MESSAGE, *history_messages]
        )

        return ChatSummaryResponse(
            summary=response_content,
            collected_data=conversation.collected_data or CollectedData(),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate summary: {str(e)}")
