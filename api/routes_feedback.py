from fastapi import APIRouter, HTTPException
from langsmith import Client
from api.schemas import FeedbackRequest

router = APIRouter()

langsmith_client = Client()


@router.post("/feedback")
def save_feedback(request: FeedbackRequest):
    """Saves a thumbs up (1) or thumbs down (0) on the trace of an answer, in LangSmith."""
    try:
        langsmith_client.create_feedback(
            request.run_id,
            key="user_score",
            score=request.score,
            comment=request.comment or None,
        )
    except Exception:
        # Do not send the real error back: it could contain details about our keys.
        raise HTTPException(status_code=502, detail="LangSmith could not save the feedback. Please try again.")

    return {"ok": True}
