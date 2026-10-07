# Phase 7: Feedback buttons

**Goal:** a thumbs up and a thumbs down under every answer. Your click is saved on that answer's trace in LangSmith.
**What you learn:** LangSmith feedback, the human side of monitoring.

## Why

The automatic evaluators in phase 8 are only guesses made by code or by another AI. Your own opinion is the most reliable check. In LangSmith you can filter for the answers you marked 👎, see what went wrong, and later compare your opinion with the automatic scores.

## Step 1. The URL that saves feedback (`api/routes_feedback.py`)

`POST /feedback` takes:

```python
class FeedbackRequest(BaseModel):
    run_id: UUID                    # the trace id that /ask returned
    score: int = Field(ge=0, le=1)  # 1 = thumbs up, 0 = thumbs down
    comment: str = ""
```

You do not write the checks by hand. In `api/schemas.py`, `run_id` has the type `UUID` and `score` is `Field(ge=0, le=1)`, so FastAPI rejects a `run_id` that is not a valid UUID (the long id with dashes) and a `score` that is not 0 or 1. Then save it in LangSmith:

```python
from app.tracing import find_project_id, langsmith_client
langsmith_client.create_feedback(run_id, key="user_score", score=score, comment=comment or None,
                                 session_id=find_project_id())
```

LangSmith now needs the `session_id` (the id of the **project** that holds the trace) when you save feedback. Without it you get a deprecation warning, and in a future version it stops working. `find_project_id()` in `app/tracing.py` returns it: the project from `LANGSMITH_PROJECT`, or, during an evaluation (when traces go into the experiment's own project), the project of the run that is running. It asks LangSmith once and remembers the answer.

Add `routes_feedback.router` in `api/main.py`. Return `{"ok": true}`. If LangSmith fails, catch the error and return a short message (status 502: "LangSmith could not save the feedback"), not the real error, because it could contain details about the keys.

## Step 2. The buttons in the chat (`ui/Home.py`)

Under each answer, show two buttons (👍 and 👎). After a 👎, show a small box for a comment ("What was wrong?"). When the user clicks, call `client.feedback(run_id, score, comment)` and replace the buttons with "Thanks, saved". Remember in `st.session_state.feedback_given` (a list of run ids) which answers already got feedback, so the buttons do not appear again when the page reloads. Each button needs its own `key`, made from the run id (for example `key=f"up_{run_id}"`), because there is one set of buttons for every answer on the page. A thumbs down sets a flag in `st.session_state`, so the comment box stays open after the page reloads.

Add a `feedback(...)` function in `ui/client.py`.

## Step 3. Try it and look at it in LangSmith

1. Ask three questions. Give 👍 to a good answer and 👎 to a bad one, with a comment.
2. In LangSmith, open the project and look for the column **user_score**.
3. Filter the traces for `user_score = 0` and open one. Use the trace to decide whether the search or the AI caused the problem.

## Step 4. Collect the 👎 answers in a review list (in the LangSmith website)

LangSmith has an **annotation queue**. It is a list of traces waiting for a person to review them. You can make LangSmith add every 👎 answer to such a list automatically. The menu names in the website may differ a little. Look for "Annotation queues", and for "Rules" or "Automations" inside your project.

1. Create a queue, for example `needs-review`.
2. In your project, add an automation rule. Filter: feedback `user_score = 0`. Action: "add to annotation queue" `needs-review`.
3. Click 👎 on an answer in the chat. After a moment the trace appears in the queue. Open it, read what went wrong, and add your own note or score.

This is how real teams review the answers of a live AI app. The traces you review are good candidates for the test questions in phase 8.

Phase 6 also saves a `guard_blocked` feedback on every request. You can use it in the same way, for example to review the requests that the guards blocked.

## You are done when

- [ ] A click on 👍 or 👎 creates a `user_score` value on the right trace.
- [ ] A 👎 trace shows up in your annotation queue.
- [ ] The comment is visible in LangSmith.
- [ ] You can filter traces by `user_score` in LangSmith.
