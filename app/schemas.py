"""Request and response models for the /ask endpoint."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Route = Literal["superhero", "dataset", "both", "none", "greeting"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500,
                          examples=["Who created Batman and what are his powers?"])
    session_id: str | None = Field(
        None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Send the session_id from a previous response to ask follow-up questions.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("question must contain at least 2 non-space characters")
        return value


class Source(BaseModel):
    type: Literal["dataset", "superhero_api"]
    title: str                      # document title or hero name
    reference: str                  # file name, or the API endpoint used
    snippet: str | None = None      # short excerpt used as evidence


class AskResponse(BaseModel):
    answer: str
    route: Route
    sources: list[Source]
    session_id: str
    standalone_question: str | None = None   # set when a follow-up was rewritten using history
