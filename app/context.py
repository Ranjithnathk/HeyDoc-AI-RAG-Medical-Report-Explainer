from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    role: Role
    content: str


def trim_history(history: List[ChatTurn], max_turns: int = 6) -> List[ChatTurn]:
    """
    Keep only the last max_turns turns (user+assistant messages).
    """
    if max_turns <= 0:
        return []
    return history[-max_turns:]


def history_to_messages(history: List[ChatTurn]) -> List[dict]:
    """
    Convert ChatTurn list into OpenAI-compatible messages.
    """
    return [{"role": h.role, "content": h.content} for h in history]
