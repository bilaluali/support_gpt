import logging
import re

from openai.types.chat import \
    ChatCompletionAssistantMessageParam as OpenAIAssistantMessage
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
from openai.types.chat import \
    ChatCompletionUserMessageParam as OpenAIUserMessage
from pydantic import ValidationError

from chat.models import OpenAIResponse
from storage.models import CollectedData, Message, MessageRole

# Initialize logger
logger = logging.getLogger(__name__)


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
