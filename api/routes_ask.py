import os
import time
from fastapi import APIRouter, HTTPException
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from api.schemas import AskRequest, AskResponse, CompareRequest, CompareResponse
from app.graph import ask

router = APIRouter()

MODES = ["dense", "sparse", "hybrid"]

# How long we wait for LangSmith to store the trace before we give up on the link.
TRACE_LINK_TRIES = 5            # how many times we ask LangSmith for the trace
TRACE_LINK_WAIT_SECONDS = 2     # how long we wait between two tries (so at most about 10 seconds in total)


def get_trace_url(run_id: str) -> str | None:
    """Builds the LangSmith link for a trace. Returns None if it cannot, so the answer is never lost."""
    # Traces are sent to LangSmith in the background. Wait until everything that is waiting has been sent.
    # (A trace with many steps, like ours with the guards, takes longer to send.)
    wait_for_all_tracers()

    client = Client()
    for _ in range(TRACE_LINK_TRIES):
        try:
            run = client.read_run(run_id)
            return client.get_run_url(run=run, project_name=os.environ["LANGSMITH_PROJECT"])
        except Exception:
            # LangSmith may need a moment to store the trace. Wait a little and try again.
            time.sleep(TRACE_LINK_WAIT_SECONDS)
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
        blocked=result["blocked"],
        guard_results=result["guard_results"],
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
