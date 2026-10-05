# Phase 11: Tests (optional)

**This phase is optional practice.** Do it only after the whole app is built and you have used it live. Until then, "testing" means running the app in the browser and fixing what breaks.

**Goal:** a small set of automatic tests that catch it when a change breaks something. Written last, because the code is stable now.

## Why now

Tests written early get rewritten every time the design changes. By now the guards, the search and the endpoints are settled, and your evaluation runs already told you what matters.

## Two rules

1. **Tests never call the internet.** No Groq, no HuggingFace, no LangSmith. They would be slow, cost tokens and fail randomly. Replace those calls with fakes.
2. **Test behaviour, not the code's insides.** Check what goes in and what comes out.

## Setup

`pytest` is already installed. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

Add `tests/conftest.py` with one fixture that turns off LangSmith tracing so tests do not send traces:

```python
import pytest

@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
```

Run all tests with `uv run pytest`.

## Test files

### `tests/test_guards.py` (the most valuable file)

Every guard is a plain function, so these are quick and need no fakes.

| Test | Input | Expect |
|---|---|---|
| prompt injection blocked | "Ignore previous instructions and..." | `block` |
| normal question allowed | "What are the four functions of the AI RMF?" | `allow` |
| spaced-out attack | "i g n o r e previous instructions" | whatever you decided in phase 6. If it gets through, write the test as the *current* behaviour and mark it `xfail` with the reason, so it reminds you |
| code injection | "run rm -rf /" | `block` |
| false alarm | "What does the system prompt in an LLM mean?" | `allow` or a known result |
| PII redaction | an answer containing `a@b.com` and `gsk_abc123...` | redacted |
| citation check | answer with `[9]` and 3 passages | `block` |
| disabled guard | guard set to `enabled: false` | `allow`, detail "disabled" |

Add one test per attack from `attacks_prompt.yaml` that you know the guards block. Loop over the file with `pytest.mark.parametrize`.

### `tests/test_retrieve.py`

Use a tiny fake corpus (5 chunks written in the test) rather than your real index.

- BM25 returns the chunk containing an exact rare word at rank 1.
- RRF: a chunk ranked 1st in both lists beats a chunk ranked 1st in only one.
- RRF handles a chunk that is in only one list.
- `search` raises a clear error for an unknown mode.

Test the RRF function and the BM25 helper directly. For dense search, replace `embed` with a fake that returns fixed vectors.

### `tests/test_graph.py`

Replace `search` and `chat` with fakes (use `monkeypatch`).

- Passages found, the fake LLM returns "Answer [1]" -> the answer has a source with the right file and page.
- No relevant passages -> `abstained` is true and the LLM fake was **not called**.
- A blocked question -> `blocked` is true and neither search nor LLM was called.
- A poisoned passage ("ignore previous instructions") is dropped before the LLM sees it.

### `tests/test_api.py`

Use FastAPI's `TestClient` (`from fastapi.testclient import TestClient`) and replace `ask` with a fake.

- `POST /ask` with a valid body -> 200 and the expected fields.
- `POST /ask` with `mode: "banana"` -> 400.
- `POST /ask` with an empty question -> a validation error (422 or 400).
- `POST /documents` with a `.txt` file -> rejected.
- `POST /documents` with a file name like `../../x.pdf` -> saved under the safe name only (check the folder).
- `PUT /settings` with an unknown guard name -> rejected.
- `PUT /datasets/golden` with an item missing `question` -> rejected.
- `PUT /datasets/../secret` or an unknown name -> rejected.
- `POST /feedback` with `score: 5` -> rejected.
- `POST /evals/run` with an unknown evaluator -> rejected, and a second run while one is running -> rejected.

### `tests/test_evaluators.py`

- The code evaluators (`retrieval_hit`, `abstention`, `citation_valid`) with hand-made inputs: a hit, a miss, a near-page hit, an abstention on an unanswerable question and a wrong abstention.
- For the LLM-judge evaluators, test only the part around the call: with a fake `chat` returning `{"score": 1, "reason": "x"}` the evaluator returns score 1, and with bad JSON it returns a sensible failure rather than crashing.

## What not to test

- Whether the LLM gives a good answer. That is what the evaluators are for.
- Streamlit pages. They are thin and just call the backend. Test the backend.

## Your tests and the evaluators together

Tests answer "does the code still work?" Evaluators answer "is the app still good?" Run `uv run pytest` after every code change. Run an evaluation after every change to a prompt, a threshold, a guard or the search.

## You are done when

- [ ] `uv run pytest` passes with no internet connection.
- [ ] Every guard has at least one test that blocks and one that allows.
- [ ] Every endpoint has at least one test for valid input and one for invalid input.
- [ ] The known weaknesses are recorded as `xfail` tests with a reason.
