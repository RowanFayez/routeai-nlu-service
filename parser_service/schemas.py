from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    query: str = Field(min_length=1, description="User text in Arabic/Egyptian Arabic")


Intent = Literal["routing", "general_info", "other"]


class ParseResponse(BaseModel):
    intent: Intent
    origin: str | None = None
    destination: str | None = None
    mode: str | None = None
    constraints: list[str] = Field(default_factory=list)

    # Optional field used by some prompts/examples
    query_type: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    adapter_path: str
    device: str
    extra: dict[str, Any] = Field(default_factory=dict)
