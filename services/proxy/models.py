"""Pydantic request models mirroring api/decision-api.yaml (components/schemas)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_QUESTIONS = 32
MAX_STATE_BYTES = 32 * 1024
MAX_LABELS = 64
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_B64_CHARS = 11184812
DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_PAD_TO_SQUARE = False  # matches the image backend default

Instructions = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
QuestionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]


class _Question(BaseModel):
    # The spec does not close question objects; Laya gets them unchanged.
    model_config = ConfigDict(extra="allow", strict=True)
    instructions: Instructions


class ChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: Annotated[dict[str, str], Field(min_length=2, max_length=64)]


class ScoreQuestion(_Question):
    type: Literal["score"]
    criteria: Annotated[list[str], Field(min_length=2, max_length=16)]


class NoulQuestion(_Question):
    type: Literal["noul"]


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class TextDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: Annotated[str, StringConstraints(min_length=1)] | dict[str, Any]
    questions: Annotated[dict[QuestionId, Question], Field(min_length=1, max_length=MAX_QUESTIONS)]
    model: str | None = None
    min_confidence: Annotated[float, Field(ge=0, le=1)] | None = None


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    base64: Annotated[str, StringConstraints(min_length=4, max_length=MAX_IMAGE_B64_CHARS)]
    media_type: Literal["image/jpeg", "image/png"] | None = None


Label = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class ImageClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image: ImageInput
    candidate_labels: Annotated[list[Label], Field(min_length=1, max_length=MAX_LABELS)]
    hypothesis_template: Annotated[str, StringConstraints(pattern=r"\{\}", max_length=512)] | None = None
    top_k: Annotated[int, Field(ge=1, le=64)] | None = None
    pad_to_square: bool | None = None

    @field_validator("candidate_labels")
    @classmethod
    def _unique(cls, labels: list[str]) -> list[str]:
        if len(set(labels)) != len(labels):
            raise ValueError("candidate_labels must be unique")
        return labels
