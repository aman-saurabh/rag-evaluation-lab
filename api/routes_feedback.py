from fastapi import APIRouter, HTTPException
from api.schemas import FeedbackRequest
from app.tracing import find_project_id, langsmith_client

router = APIRouter()


@router.post("/feedback")
def save_feedback(request: FeedbackRequest):
    """Saves a thumbs up (1) or thumbs down (0) on the trace of an answer, in LangSmith."""
    try:
        langsmith_client.create_feedback(
            request.run_id,
            key="user_score",
            score=request.score,
            comment=request.comment or None,
            session_id=find_project_id(),  # LangSmith needs the project that holds the trace
        )
    except Exception:
        # Do not send the real error back: it could contain details about our keys.
        raise HTTPException(status_code=502, detail="LangSmith could not save the feedback. Please try again.")

    return {"ok": True}
