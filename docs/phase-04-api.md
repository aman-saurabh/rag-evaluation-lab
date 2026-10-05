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
    guard_results: list[dict] = []   # filled in phase 6
    mode: str
```

Add `CompareRequest` (question only) and a `DocumentInfo` (file, chunks) when you need them.

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

A **router** is a group of endpoints kept in its own file. Each later phase adds one router file and one `include_router` line.

## Step 3. The endpoints

**`POST /ask`** (`routes_ask.py`): takes `AskRequest`, calls `ask()`, returns `AskResponse`. Reject a `mode` that is not one of the three with a clear error (`HTTPException(400, ...)`).

**`POST /compare`**: takes a question and runs all three modes. Returns a dict with `dense`, `sparse` and `hybrid` results. Run them one after the other (Groq has token limits).

**`POST /documents`** (`routes_documents.py`): takes an uploaded PDF (`file: UploadFile`).
1. Reject a name that does not end in `.pdf`, or a file larger than 20 MB.
2. Save to `data/pdfs/` under a safe name (`Path(name).name`, never trust the path).
3. Re-run ingest. For the first version, re-index everything. Say so in the reply.
4. Return the file name and number of chunks.

**`GET /documents`**: list the PDFs in `data/pdfs/` with their chunk counts (count from `chunks.jsonl`).

**`DELETE /documents/{name}`**: delete the PDF, then re-index.

**`GET /health`**: try a tiny call to each of Groq, HuggingFace, Qdrant (count the collection) and LangSmith. Return `{"groq": true, "huggingface": true, ...}`. Never return an error message that could include a key.

After an upload or delete, the in-memory BM25 index and the Qdrant handle must be refreshed. Add a `reload_index()` function in `retrieve.py` and call it after ingest.

Ingest takes minutes for big files. For the first version just let the request wait. (The UI will show a spinner.)

## Step 4. Run it

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
- [ ] `POST /documents` accepts a PDF and `GET /documents` lists it.
- [ ] `GET /health` shows all four services true.
