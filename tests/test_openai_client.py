from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from openai import RateLimitError as OpenAIRateLimitError

from openai_client import OpenAIClient


class TestOpenAIClient:
    def setup_method(self):
        self.patcher = patch("openai_client.OpenAI")
        self.mock_openai_class = self.patcher.start()
        self.mock_client = Mock()
        self.mock_openai_class.return_value = self.mock_client
        self.client = OpenAIClient()

    def teardown_method(self):
        self.patcher.stop()

    ### Moderation API tests ###
    def test_is_offensive_content_success(self):
        mock_moderation = Mock()
        mock_moderation.results = [Mock(flagged=True)]
        self.mock_client.moderations.create.return_value = mock_moderation

        result = self.client.is_offensive_content("test text")

        assert result is True
        self.mock_client.moderations.create.assert_called_once_with(input="test text")

    def test_is_offensive_content_failure(self):
        self.mock_client.moderations.create.side_effect = Exception(
            "Failed to moderate content"
        )

        with pytest.raises(HTTPException) as exc_info:
            self.client.is_offensive_content("test text")
        assert exc_info.value.status_code == 500

    @patch("time.sleep")
    def test_is_offensive_content_rate_limit_error(self, m_sleep):
        self.mock_client.moderations.create.side_effect = OpenAIRateLimitError(
            "Rate limit", response=Mock(), body=Mock()
        )
        m_sleep.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            self.client.is_offensive_content("test text")
        assert exc_info.value.status_code == 429

    ### Chat completion API tests ###
    def test_create_chat_completion_success(self):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        self.mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.create_chat_completion(messages)

        assert result == "Test response"

    def test_create_chat_completion_failure(self):
        self.mock_client.chat.completions.create.side_effect = Exception(
            "Failed to create chat completion"
        )

        with pytest.raises(HTTPException) as exc_info:
            self.client.create_chat_completion(
                [{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.status_code == 500

    @patch("time.sleep")
    def test_create_chat_completion_rate_limit_error(self, m_sleep):
        self.mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
            "Rate limit", response=Mock(), body=Mock()
        )
        m_sleep.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            self.client.create_chat_completion(
                [{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.status_code == 429
