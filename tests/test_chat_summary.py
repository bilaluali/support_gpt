from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionUserMessageParam as OpenAIUserMessage

from chat.models import ChatSummaryResponse
from chat.prompts import CHAT_SUMMARY_SYSTEM_MESSAGE
from main import app
from storage.models import CollectedData, Conversation, Message, MessageRole


class TestChatSummaryAPI:
    def setup_method(self):
        self.client = TestClient(app)

        self.completion_patcher = patch("config.openai_client.create_chat_completion")
        self.mock_create_completion = self.completion_patcher.start()

        self.get_conversation_patcher = patch("config.storage.get_conversation")
        self.mock_get_conversation = self.get_conversation_patcher.start()

    def teardown_method(self):
        self.completion_patcher.stop()
        self.get_conversation_patcher.stop()
        self.client.close()

    def test_chat_summary_success(self):
        # Given
        self.mock_create_completion.return_value = "Summary of the conversation"
        self.mock_get_conversation.return_value = Conversation(
            session_id="test-session-id",
            messages=[Message(role=MessageRole.USER, content="Fake message user")],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        expected_response = ChatSummaryResponse(
            summary="Summary of the conversation",
            collected_data=CollectedData(),
        )

        # When
        response = self.client.post(
            "/chat/summary",
            json={"transaction_id": "test-transaction-id"},
        )

        # Then
        assert response.status_code == 200
        assert response.json() == expected_response.model_dump()

        self.mock_get_conversation.assert_called_once_with("test-transaction-id")
        self.mock_create_completion.assert_called_once_with(
            messages=[
                CHAT_SUMMARY_SYSTEM_MESSAGE,
                OpenAIUserMessage(role="user", content="Fake message user"),
            ]
        )

    def test_chat_summary_empty_transaction_id(self):
        response = self.client.post(
            "/chat/summary",
            json={"transaction_id": " " * 50},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Transaction ID cannot be empty."}

    def test_chat_summary_conversation_not_found(self):
        self.mock_get_conversation.return_value = None
        response = self.client.post(
            "/chat/summary",
            json={"transaction_id": "test-transaction-id"},
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Conversation not found."}

    def test_chat_summary_failure(self):
        self.mock_create_completion.side_effect = Exception(
            "Failed to create chat completion"
        )
        response = self.client.post(
            "/chat/summary",
            json={"transaction_id": "test-transaction-id"},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Failed to generate summary: Failed to create chat completion"
        }
