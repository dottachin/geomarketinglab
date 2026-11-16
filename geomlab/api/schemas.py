"""Pydantic request/response models for the FastAPI app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatContext(BaseModel):
    """What the user has selected in the UI (sent along so 'here' / 'this hexagon' resolve)."""
    city: str | None = Field(None, max_length=64)
    business_type: str | None = Field(None, max_length=64)
    h3_id: str | None = Field(None, max_length=32)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    provider: Literal["anthropic", "ollama"] | None = None
    context: ChatContext | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    city: str | None = None
    doc_type: str | None = None
    year: int | None = None
    k: int = Field(5, ge=1, le=10)


class StatsRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=4000)


class SimulateRequest(BaseModel):
    business_type: str
    changes: dict[str, float] = Field(min_length=1, max_length=10)
