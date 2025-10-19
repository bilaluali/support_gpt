import os
from datetime import datetime, timezone
from typing import Optional

from .models import CollectedData, Conversation


class SimpleStorage:
    """
    Simple file-based conversation storage using JSON.
    In production, replace with a proper database (e.g., Redis, MongoDB).
    """

    def __init__(self, db_path: str = "db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get conversation from JSON file"""
        file_path = os.path.join(self.db_path, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return Conversation.model_validate_json(f.read())
        return None

    def get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get or create conversation from JSON file"""
        conversation = self.get_conversation(session_id)
        return conversation or Conversation(
            session_id=session_id,
            messages=[],
            collected_data=CollectedData()
            if conversation is None
            else conversation.collected_data,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def update_conversation(self, session_id: str, conversation: Conversation) -> None:
        """Update conversation in JSON file"""
        conversation.updated_at = datetime.now(timezone.utc)
        file_path = os.path.join(self.db_path, f"{session_id}.json")
        with open(file_path, "w") as f:
            f.write(conversation.model_dump_json())
