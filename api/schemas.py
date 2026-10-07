from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"            # "dense", "sparse" or "hybrid"
    session_id: str | None = None   # same id = same conversation (memory); no id = a fresh chat


class Source(BaseModel):
    file: str
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    abstained: bool
    blocked: bool = False           # true if a guard stopped the question
    guard_results: list[dict] = []  # what each guard did (allow, block, drop or redact)
    run_id: str
    trace_url: str | None = None    # link to the trace in LangSmith, if it could be built
    mode: str


class DocumentInfo(BaseModel):
    file: str
    chunks: int


class UploadResponse(DocumentInfo):
    message: str


class GuardSetting(BaseModel):
    enabled: bool


class Settings(BaseModel):
    guards: dict[str, GuardSetting]   # for example {"length": {"enabled": true}}
    thresholds: dict[str, float]      # for example {"dense_min_score": 0.55}


class FeedbackRequest(BaseModel):
    run_id: UUID                          # the trace id from /ask. FastAPI rejects anything that is not a valid UUID.
    score: int = Field(ge=0, le=1)        # 1 = thumbs up, 0 = thumbs down. Anything else is rejected.
    comment: str = ""


class DatasetInfo(BaseModel):
    name: str
    examples: int


class SourceRef(BaseModel):
    file: str
    page: int


class GoldenItem(BaseModel):
    """One golden question. extra="forbid" rejects fields we do not know (for example a typo in a field name)."""
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["answerable", "unanswerable", "multi_document"]
    question: str
    answer: str | None              # null for a question without an answer
    sources: list[SourceRef] = []   # the pages that hold the answer
    notes: str | None = None


class AttackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    attack: str


class CompareRequest(BaseModel):
    question: str


class CompareResponse(BaseModel):
    dense: AskResponse
    sparse: AskResponse
    hybrid: AskResponse


class EvalRunRequest(BaseModel):
    dataset: Literal["golden", "attacks_prompt", "attacks_code"]
    mode: Literal["dense", "sparse", "hybrid", "all"]
    evaluators: list[str]   # names from ALL_EVALUATORS, checked in the route
