import os
import shutil
from datetime import datetime, timezone

from freezegun import freeze_time

from storage import Conversation, Message, MessageRole, Storage


class TestStorage:
    def setup_method(self):
        self.storage = Storage(db_path="tests/db")

    def teardown_method(self):
        # Delete the db path after each test
        shutil.rmtree(self.storage.db_path)

    def test_get_conversation(self):
        # Given a conversation
        session_id = "feca9559-dbc0-4b4e-a9e2-7de000907035"
        messages = [
            Message(role=MessageRole.USER, content="Hello, how are you?"),
            Message(role=MessageRole.ASSISTANT, content="I'm doing great, thank you!"),
        ]
        conversation = Conversation(
            session_id=session_id,
            messages=messages,
            collected_data=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Given a file in the db path
        os.makedirs(self.storage.db_path, exist_ok=True)
        with open(os.path.join(self.storage.db_path, f"{session_id}.json"), "w") as f:
            f.write(conversation.model_dump_json())

        # When getting the conversation
        conversation = self.storage.get_or_create_conversation(session_id)

        # Then the conversation is returned
        assert isinstance(conversation, Conversation)
        assert conversation.session_id == session_id
        assert conversation.messages == messages

    @freeze_time("2025-01-01 12:00:00")
    def test_create_conversation(self):
        session_id = "non-existent-session-id"

        conversation = self.storage.get_or_create_conversation(session_id)

        assert isinstance(conversation, Conversation)
        assert conversation.session_id == session_id
        assert conversation.created_at == datetime.now(timezone.utc)
        assert conversation.updated_at == datetime.now(timezone.utc)

    @freeze_time("2025-01-01 12:01:00")
    def test_update_conversation(self):
        # Given a conversation
        session_id = "feca9559-dbc0-4b4e-a9e2-7de000907035"
        conversation = Conversation(
            session_id=session_id,
            messages=[],
            collected_data=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Given a file in the db path
        os.makedirs(self.storage.db_path, exist_ok=True)
        with open(os.path.join(self.storage.db_path, f"{session_id}.json"), "w") as f:
            f.write(conversation.model_dump_json())

        # Given a new message
        new_messages = [
            Message(role=MessageRole.USER, content="Hello, how are you?"),
            Message(role=MessageRole.ASSISTANT, content="I'm doing great, thank you!"),
        ]
        conversation.messages.extend(new_messages)

        # When updating the conversation
        self.storage.update_conversation(session_id, conversation)

        # Then the conversation is updated
        assert isinstance(conversation, Conversation)
        assert conversation.session_id == session_id
        assert conversation.messages == new_messages
        assert conversation.collected_data is None
        assert conversation.updated_at == datetime(
            2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc
        )
