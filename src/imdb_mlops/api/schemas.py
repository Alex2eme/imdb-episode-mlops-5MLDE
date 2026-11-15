"""Contrats d'entrée/sortie de l'API (validation stricte des entrées)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SERIES_ID_PATTERN = r"^tt\d{7,9}$"
MAX_BATCH = 1000


class EpisodeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(pattern=SERIES_ID_PATTERN, examples=["tt0903747"])
    season: int = Field(ge=0, le=500, examples=[5])
    episode: int = Field(ge=0, le=20000, examples=[14])


class ModelInfo(BaseModel):
    name: str
    version: str
    run_id: str
    data_sha: str
    rmse: float | None
    loaded_at: str


class Prediction(BaseModel):
    prediction_id: str
    series_id: str
    season: int
    episode: int
    predicted_rating: float
    model_version: str
    data_sha: str
    cached: bool
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EpisodeRef] = Field(min_length=1, max_length=MAX_BATCH)


class BatchItemError(BaseModel):
    index: int
    series_id: str
    season: int
    episode: int
    error: str
    message: str


class BatchResponse(BaseModel):
    model_version: str
    n_ok: int
    n_errors: int
    latency_ms: float
    predictions: list[Prediction]
    errors: list[BatchItemError]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(min_length=8, max_length=64)
    rating: float = Field(ge=1, le=10, description="Note observée ou attribuée (1-10)")
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    prediction_id: str
    received: bool = True


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: list | dict | None = None
