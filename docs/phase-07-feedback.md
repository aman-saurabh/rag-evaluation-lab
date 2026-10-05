# Phase 7: Feedback buttons

**Goal:** a thumbs up and thumbs down under every answer. Your click is saved on that answer's trace in LangSmith.
**This phase teaches:** LangSmith feedback, the human side of monitoring.

## Why

Automatic evaluators (phase 8) are only guesses by code or by another AI. Your thumbs are the ground truth. In LangSmith you can filter for traces you marked 👎, see what went wrong, and later compare your opinion with the automatic scores.

## Step 1. The endpoint (`api/routes_feedback.py`)

`POST /feedback` takes:

```python
class FeedbackRequest(BaseModel):
    run_id: str
    score: int          # 1 = thumbs up, 0 = thumbs down
    comment: str = ""
```

Check that `score` is 0 or 1 and that `run_id` is a valid UUID. Then:

```python
from langsmith import Client
Client().create_feedback(run_id, key="user_score", score=score, comment=comment or None)
```

Add `routes_feedback.router` in `api/main.py`. Return `{"ok": true}`. If LangSmith fails, return a clear error and do not crash.

## Step 2. The buttons in Chat (`ui/Home.py`)

Under each answer show two buttons (👍, 👎) and an optional comment box that appears after a 👎 ("What was wrong?"). On click call `client.feedback(run_id, score, comment)` and replace the buttons with "Thanks, saved". Remember in `st.session_state` which answers already got feedback, so a rerun does not show the buttons again.

Add a `feedback(...)` function in `ui/client.py`.

## Step 3. Try it and see it in LangSmith

1. Ask three questions. Give 👍 to a good answer and 👎 to a bad one, with a comment.
2. In LangSmith, open the project and look for the column **user_score**.
3. Filter traces for `user_score = 0` and open one. Use the trace to decide whether the search or the AI caused the problem.

## You are done when

- [ ] A click on 👍 or 👎 creates a `user_score` value on the right trace.
- [ ] The comment is visible in LangSmith.
- [ ] You can filter traces by `user_score` in LangSmith.
