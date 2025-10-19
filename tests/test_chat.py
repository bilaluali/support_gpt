from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionUserMessageParam as OpenAIUserMessage

from chat import ChatResponse, OpenAIResponse, CHAT_SYSTEM_MESSAGE, parse_response
from main import app
from storage import CollectedData, Conversation


class TestChatAPI:
    def setup_method(self):
        self.client = TestClient(app)

        # Mock openai client methods
        self.offensive_patcher = patch("config.openai_client.is_offensive_content")
        self.completion_patcher = patch("config.openai_client.create_chat_completion")
        self.mock_is_offensive = self.offensive_patcher.start()
        self.mock_create_completion = self.completion_patcher.start()

        # Mock storage
        self.get_or_create_patcher = patch("config.storage.get_or_create_conversation")
        self.update_patcher = patch("config.storage.update_conversation")
        self.mock_get_or_create = self.get_or_create_patcher.start()
        self.mock_update = self.update_patcher.start()

    def teardown_method(self):
        self.offensive_patcher.stop()
        self.completion_patcher.stop()
        self.get_or_create_patcher.stop()
        self.update_patcher.stop()
        self.client.close()

    def test_chat_success(self):
        # Given
        conversation = Conversation(
            session_id="test-session-id",
            messages=[],
            collected_data=CollectedData(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_get_or_create.return_value = conversation
        self.mock_update.return_value = None
        self.mock_is_offensive.return_value = False
        self.mock_create_completion.return_value = "All good, how can I help you today?"

        expected_response = ChatResponse(
            transaction_id="test-transaction-id",
            response="All good, how can I help you today?",
            collected_data=CollectedData(),
        )

        # When
        response = self.client.post(
            "/chat",
            json={"user_message": "Hello, how are you?", "transaction_id": "test-transaction-id"},
        )

        # Then
        assert response.status_code == 200
        assert response.json() == expected_response.model_dump()

        self.mock_is_offensive.assert_called_once_with("Hello, how are you?")
        self.mock_get_or_create.assert_called_once_with("test-transaction-id")
        self.mock_update.assert_called_once_with(
            "test-transaction-id",
            conversation,
        )
        self.mock_create_completion.assert_called_once_with(
            messages=[
                CHAT_SYSTEM_MESSAGE,
                OpenAIUserMessage(role="user", content="Hello, how are you?"),
            ]
        )

    def test_chat_empty_message(self):
        empty_message = " " * 50
        response = self.client.post(
            "/chat",
            json={"user_message": empty_message, "transaction_id": "test-transaction-id"},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Message cannot be empty."}

    def test_chat_offensive_content(self):
        offensive_message = "fuck you"
        self.mock_is_offensive.return_value = True
        response = self.client.post(
            "/chat",
            json={"user_message": offensive_message, "transaction_id": "test-transaction-id"},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Message contains offensive content."}

    def test_chat_failure(self):
        self.mock_is_offensive.return_value = False
        self.mock_create_completion.side_effect = Exception(
            "Failed to create chat completion"
        )
        response = self.client.post(
            "/chat",
            json={"user_message": "Hello, how are you?", "transaction_id": "test-transaction-id"},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Failed to generate response: Failed to create chat completion"
        }


def test_parse_response():
    collected_data = CollectedData(
        order_number="1234567890",
        problem_category="technical",
        problem_description="The product is not working",
        urgency_level="high",
    )
    response_content = (
        "All good, how can I help you today?"
        + "<COLLECTED_DATA>"
        + collected_data.model_dump_json()
        + "</COLLECTED_DATA>"
    )
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == collected_data


@patch("chat.api.logger")
def test_parse_response_no_collected_data_tag(m_logger):
    response_content = "All good, how can I help you today?"
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == CollectedData()
    m_logger.warning.assert_called_once_with(
        "No <COLLECTED_DATA> block found in response."
    )


@patch("chat.api.logger")
def test_parse_response_invalid_collected_data_json(m_logger):
    response_content = "All good, how can I help you today? <COLLECTED_DATA>invalid json</COLLECTED_DATA>"
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == CollectedData()
    assert (
        "Invalid JSON in <COLLECTED_DATA>: invalid json"
        in m_logger.error.call_args[0][0]
    )
