import os
import time
from fastapi import APIRouter, HTTPException
from langsmith import Client
from api.schemas import AskRequest, AskResponse, CompareRequest, CompareResponse
from app.graph import ask

router = APIRouter()

MODES = ["dense", "sparse", "hybrid"]


def get_trace_url(run_id: str) -> str | None:
    """Builds the LangSmith link for a trace. Returns None if it cannot, so the answer is never lost."""
    client = Client()
    for _ in range(2):
        try:
            run = client.read_run(run_id)
            return client.get_run_url(run=run, project_name=os.environ["LANGSMITH_PROJECT"])
        except Exception:
            # The trace is sent in the background, so it may not exist yet. Wait a second and try once more.
            time.sleep(1)
    return None


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")

    result = ask(request.question, request.mode, session_id=request.session_id, tags=["source:ui"])
    trace_url = get_trace_url(result["run_id"])

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        abstained=result["abstained"],
        run_id=result["run_id"],
        trace_url=trace_url,
        mode=request.mode,
    )


@router.post("/compare", response_model=CompareResponse)
def compare_modes(request: CompareRequest):
    """Asks the same question in all three modes, one after the other (Groq has token limits)."""
    answers = {}
    for mode in MODES:
        answers[mode] = ask_question(AskRequest(question=request.question, mode=mode))
    return CompareResponse(dense=answers["dense"], sparse=answers["sparse"], hybrid=answers["hybrid"])
