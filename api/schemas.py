from pydantic import BaseModel


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
    run_id: str
    trace_url: str | None = None    # link to the trace in LangSmith, if it could be built
    mode: str


class DocumentInfo(BaseModel):
    file: str
    chunks: int


class UploadResponse(DocumentInfo):
    message: str


class CompareRequest(BaseModel):
    question: str


class CompareResponse(BaseModel):
    dense: AskResponse
    sparse: AskResponse
    hybrid: AskResponse
