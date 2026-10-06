# Phase 11: Tests (optional)

**This phase is optional practice.** Do it only after the whole app is built and you have used it live. Until then, "testing" means running the app in the browser and fixing what breaks.

**Goal:** a small set of automatic tests that catch it when a change breaks something. Written last, because the code is stable now.

## Why now

Tests written early get rewritten every time the design changes. By now the guards, the search and the endpoints are settled, and your evaluation runs already told you what matters.

## Two rules

1. **Tests never call the internet.** No Groq, no HuggingFace, no LangSmith. They would be slow, cost tokens and fail now and then for reasons that have nothing to do with your code. Use "fakes" instead: stand-in objects that answer instantly and cost nothing.
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

The guards that use an AI model would call Groq, so give them a fake model (LangChain's `GenericFakeChatModel`, or temporarily replace `check_prompt_injection` and `check_safety` using pytest's `monkeypatch`). Then test what your code does with the fake model's verdict. The plain guards (`length`, `citation_output`) need no fakes. Test `PIIMiddleware` through the answer step, with a fake model.

| Test | Input | Expect |
|---|---|---|
| prompt injection blocked | fake Prompt Guard score 0.99 | `block` |
| normal question allowed | "What are the four functions of the AI RMF?" | `allow` |
| fake safeguard verdict `violation=True` | any text | `block` |
| fake safeguard verdict `violation=False` | "What does the system prompt in an LLM mean?" | `allow` |
| PII redaction | a fake model that answers with `a@b.com` and `gsk_abc123...` | both redacted in the answer |
| citation check | answer with `[9]` and 3 retrieved documents | `block` |
| disabled guard | guard set to `enabled: false` | `allow`, detail "disabled" |

The real attack sets are for the evaluators (phase 8), not for these tests: real model-based guards need the internet.

### `tests/test_retrieve.py`

Use a tiny fake corpus (5 chunks written in the test) rather than your real index.

- BM25 returns the chunk containing an exact rare word at rank 1.
- Hybrid (`EnsembleRetriever`): a chunk ranked 1st in both lists beats a chunk ranked 1st in only one, and a chunk in only one list still appears.
- `search` raises a clear error for an unknown mode.

Build a `BM25Retriever` and an `EnsembleRetriever` on the fake corpus. For dense search, use LangChain's `DeterministicFakeEmbedding` as the `embeddings`, with an in-memory `QdrantVectorStore`.

### `tests/test_graph.py`

Replace `search` and the answer step's model with fakes (`monkeypatch` swaps a function for a fake during one test; LangChain's `GenericFakeChatModel` works as a fake model).

- Retrieved documents found, the fake LLM returns "Answer [1]" -> the answer has a source with the right file and page.
- No relevant retrieved documents -> `abstained` is true and the LLM fake was **not called**.
- A blocked question -> `blocked` is true and neither search nor LLM was called.
- A retrieved document that the fake Prompt Guard flags is dropped before the LLM sees it.

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
- `PUT /settings` with a wrong type for a threshold -> rejected.

### `tests/test_evaluators.py`

- The custom code evaluators (`retrieval_hit`, `abstention`, `citation_valid`, `blocked_by_guard`) with hand-made inputs: a hit, a miss, a near-page hit, an abstention on an unanswerable question and a wrong abstention.
- For the `openevals` evaluators, test only your wrappers: with a fake judge result the wrapper returns 0 or 1 in the direction "1 is good" (including the flipped ones).

## What not to test

- Whether the LLM gives a good answer. That is what the evaluators are for.
- Streamlit pages. They are thin and just call the backend. Test the backend.

## Your tests and the evaluators together

Tests answer "does the code still work?" Evaluators answer "is the app still good?" Run `uv run pytest` after every code change. Run an evaluation after every change to a prompt, a threshold, a guard or the search.

## You are done when

- [ ] `uv run pytest` passes with no internet connection.
- [ ] Every guard has at least one test that blocks and one that allows.
- [ ] Every endpoint has at least one test for valid input and one for invalid input.
- [ ] The known weaknesses are recorded as `xfail` tests with a reason (`xfail` is pytest's way of saying "this is a known problem").
