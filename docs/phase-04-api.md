# Phase 4: The backend (FastAPI)

**Goal:** a running server where `curl` (or the built-in docs page) can ask a question, upload a document and check health.
**This phase teaches:** FastAPI basics: endpoints, request and reply shapes, file upload.

## What FastAPI gives you

You write ordinary Python functions. FastAPI turns them into URLs. It reads the shape of the data from **Pydantic models** (small classes that list the fields), rejects bad requests for you, and builds a test page automatically at `http://localhost:8000/docs`.

## Step 1. Describe the data (`api/schemas.py`)

```python
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"        # "dense", "sparse" or "hybrid"
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
    trace_url: str | None = None
    mode: str
```

Step 2 of the build order adds `CompareRequest` (question only) and `CompareResponse` (one `AskResponse` for each of `dense`, `sparse`, `hybrid`). Step 3 adds `DocumentInfo` (file, chunks) and `UploadResponse` (the same plus a `message`). Phase 6 adds `guard_results` to `AskResponse`; do not add it now.

## Step 2. Start the app (`api/main.py`)

```python
from fastapi import FastAPI
from api import routes_ask, routes_documents

app = FastAPI(title="RAG Evaluation Lab")
app.include_router(routes_ask.router)
app.include_router(routes_documents.router)

@app.get("/health")
def health(): ...
```

A **router** is a group of endpoints kept in its own file. Each later step adds one router file and one `include_router` line (for example `routes_documents` when you build the document endpoints in step 3).

## Step 3. The endpoints

Each file holds the endpoints of one job.

### `api/main.py`

**`GET /health`**: try one cheap call to each of Groq, HuggingFace, Qdrant and LangSmith. Return `{"groq": true, "huggingface": true, ...}`. Never return an error message that could include a key. Use a call that costs nothing for Groq (`Groq().models.list()` lists models, it uses no tokens), because the web page may call `/health` often and the free token limit is small.

### `api/routes_ask.py`

**`POST /ask`**: takes `AskRequest`, calls `ask(question, mode, session_id=..., tags=["source:ui"])`, returns `AskResponse`. Reject a `mode` that is not one of the three with a clear error (`HTTPException(400, ...)`). The `session_id` is what gives the chat its memory: the web page sends the same id for the whole conversation.

**`POST /compare`**: takes a question and runs all three modes. Returns `dense`, `sparse` and `hybrid` results (`CompareResponse`). Run them one after the other (Groq has token limits). It simply calls the `/ask` function once per mode, with no session, so each mode starts a fresh chat. One call uses up to 3 answers' worth of Groq tokens.

### `api/routes_documents.py`

**`POST /documents`**: takes an uploaded PDF (`file: UploadFile`).
1. Reject a name that does not end in `.pdf`, or a file that does not start with `%PDF`. (The 20 MB size limit is checked in the web page in phase 5, before the file is sent. The API does not check the size.)
2. Save to `data/pdfs/` under a safe name (`Path(name).name`, never trust the path).
3. Re-run ingest. For the first version, re-index everything. Say so in the reply.
4. Return the file name and number of chunks.

**`GET /documents`**: list the PDFs in `data/pdfs/` with their chunk counts (count from `chunks.jsonl`).

**`DELETE /documents/{name}`**: delete the PDF, then re-index.

**Refreshing the search after an upload or delete.** The in-memory BM25 index and the Qdrant handle must be rebuilt. This is done by `reload_index()` in `app/retrieve.py`:
1. It closes the Qdrant client first, because on Windows the Qdrant folder cannot be deleted while it is open.
2. It runs ingest (`ingest_all()`), which rebuilds `chunks.jsonl` and the Qdrant folder.
3. It runs `load_index()`, which creates the search objects again (`store`, `bm25`, `hybrid`).

Other files must not keep their own copy of those objects, because the copy would be old after a reload. So instead of `from app.retrieve import bm25`, they write `from app import retrieve as retrieval` and use `retrieval.bm25` or `retrieval.store` each time. That always gives the current object. (`graph.py` uses it for the sparse check, `main.py` for `/health`.)

Other rules in `routes_documents.py`:
- The file must start with `%PDF`, so a renamed text file is rejected.
- If indexing fails, the new PDF is deleted and the reply says to run `uv run python -m app.ingest` and restart the server (the old index may be half-deleted).
- The last remaining document cannot be deleted, because there would be nothing to index.

Ingest re-embeds every document, so it takes minutes and uses HuggingFace credit each time. For the first version just let the request wait. (The UI will show a spinner.)

## Build order

1. `schemas.py`, `main.py` with `/health`, and `routes_ask.py` with `/ask` (including the trace URL from step 6). Test them on the `/docs` page.
2. `/compare`.
3. `routes_documents.py` and `reload_index()` (the hardest part, so it comes last).

## Step 4. Run it

Only one program can open the Qdrant folder at a time. **Stop the server before running `try_search.py`, `try_ask.py` or ingest**, and stop those before starting the server.

```powershell
uv run uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`. You can try each endpoint right on that page.

`--reload` restarts the server when you edit code. It also restarts when files in the project folder change. If it restarts in a loop when `data/` changes, use `--reload-dir app --reload-dir api`.

## Step 5. Try it with `curl`

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" `
  -d '{\"question\": \"What are the four functions of the AI RMF?\", \"mode\": \"hybrid\"}'
```

(If PowerShell mangles the quotes, use the `/docs` page.)

## Step 6. Get the trace URL

In `routes_ask.py`, after `ask()`, build the link to the LangSmith trace. Checked against `langsmith` 0.14.4:

```python
client = Client()
run = client.read_run(run_id)
trace_url = client.get_run_url(run=run, project_name="rag-evaluation-lab")
```

The trace is sent in the background, so `read_run` may fail for a moment right after the answer. Wait a second and retry once. If it still fails, return `trace_url = None`. Never let this break the answer.

## You are done when

- [ ] `uvicorn` starts and `/docs` opens.
- [ ] `POST /ask` returns an answer, sources and a `run_id`, in all three modes.
- [ ] `POST /compare` returns three answers (dense, sparse, hybrid).
- [ ] `POST /ask` with the same `session_id` twice remembers the first question (try "What is the GOVERN function?" then "What is its purpose?").
- [ ] `GET /documents` lists the five NIST PDFs with chunk counts.
- [ ] `POST /documents` accepts a small PDF and `GET /documents` lists it. Then `DELETE /documents/{name}` removes it. (Each of these re-indexes everything, so it takes minutes and uses HuggingFace credit; do it once.)
- [ ] After the upload and delete, `POST /ask` still works (the search was refreshed).
- [ ] `GET /health` shows all four services true.
