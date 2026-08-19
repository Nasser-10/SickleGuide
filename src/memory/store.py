from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


DEFAULT_DB_PATH = Path("data/sickleguide_memory.sqlite3")
MAX_MESSAGES = 24
MAX_CONTENT_LENGTH = 4000


class PersistentChatMemory:
    """Durable per-chat memory backed by SQLite.

    The database is local to the application and survives backend restarts.
    Only conversation turns are persisted; retrieved medical evidence is never
    treated as memory and must still come from the RAG retriever.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages ON chat_messages(chat_id, id)"
            )
            connection.commit()

    @staticmethod
    def _validate_chat_id(chat_id: str) -> str:
        value = str(chat_id or "").strip()
        if not value or len(value) > 100:
            raise ValueError("Invalid chat_id")
        return value

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for message in messages[-MAX_MESSAGES:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized.append({"role": role, "content": content[:MAX_CONTENT_LENGTH]})
        return normalized

    def load(self, chat_id: str) -> List[Dict[str, str]]:
        chat_id = self._validate_chat_id(chat_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM chat_messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, MAX_MESSAGES),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def append(self, chat_id: str, messages: List[Dict[str, str]]) -> None:
        chat_id = self._validate_chat_id(chat_id)
        normalized = self._normalize_messages(messages)
        if not normalized:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO chat_messages(chat_id, role, content) VALUES (?, ?, ?)",
                [(chat_id, item["role"], item["content"]) for item in normalized],
            )
            connection.commit()

    def append_turn(self, chat_id: str, user: str, assistant: str) -> None:
        self.append(chat_id, [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])

    def clear(self, chat_id: str) -> None:
        chat_id = self._validate_chat_id(chat_id)
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            connection.commit()

    def as_langchain_messages(self, chat_id: str) -> List[BaseMessage]:
        messages = self.load(chat_id)
        converted: List[BaseMessage] = []
        for item in messages:
            if item["role"] == "user":
                converted.append(HumanMessage(content=item["content"]))
            else:
                converted.append(AIMessage(content=item["content"]))
        return converted


_memory_store = PersistentChatMemory()


def get_memory_store() -> PersistentChatMemory:
    return _memory_store
