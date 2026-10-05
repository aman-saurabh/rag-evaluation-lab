# Phase 0: Setup

**Goal:** an empty project that runs Python, with your four keys working.
**Time:** about an hour.

## Step 1. Check your tools

In PowerShell:

```powershell
cd C:\Users\asaur\Projects\rag-evaluation-lab
git --version
uv --version
python --version
```

You need git, `uv` (a fast Python package tool) and Python 3.13. If `uv` is missing: `winget install astral-sh.uv`.

## Step 2. Start the project

```powershell
git init -b main
uv init --app --python 3.13 --name rag-evaluation-lab
```

This creates `pyproject.toml` and a sample `main.py`. Delete `main.py`. You will not use it.

## Step 3. Install the libraries

```powershell
uv add fastapi uvicorn python-multipart streamlit httpx langgraph langsmith groq rank-bm25 qdrant-client pypdf pyyaml python-dotenv
uv add --dev pytest
```

## Step 4. Check `.gitignore`

`.gitignore` already exists (it was created when this folder was prepared). Check that it contains:

```
.venv/
__pycache__/
.env
data/qdrant/
.pytest_cache/
```

## Step 5. Check your keys

`.env` and `.env.example` already exist. The three keys (Groq, HuggingFace, LangSmith) were copied from the old project, so you do not need to get them again. Open `.env` and check that all three have a value. `.env.example` looks like this (no values):

```
GROQ_API_KEY=
HF_TOKEN=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-evaluation-lab
```

You do not need a Qdrant key, because Qdrant runs locally in a folder.

## Step 6. Create the folders

```powershell
mkdir app, api, ui, ui\pages, tests
```

(`data\pdfs` and `data\datasets` already exist.) Create an empty `app\__init__.py` and `api\__init__.py`.

## Step 7. Write `app/config.py`

This file loads `.env` and holds constants used everywhere:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
CHUNKS_FILE = DATA / "chunks.jsonl"
QDRANT_DIR = DATA / "qdrant"
SETTINGS_FILE = DATA / "settings.json"
DATASET_DIR = DATA / "datasets"

COLLECTION = "chunks"
EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
EMBED_DIM = 384
FAST_MODEL = "openai/gpt-oss-20b"
STRONG_MODEL = "openai/gpt-oss-120b"
```

## Step 8. Check each key once

Create a temporary file `check_keys.py` at the project root:

```python
import os
import httpx
from groq import Groq
from langsmith import Client
from app.config import EMBED_URL

print("Groq:", Groq().chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say ok"}],
    max_tokens=50,
).choices[0].message.content)

r = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
               json={"inputs": ["hello"]}, timeout=60)
print("HuggingFace:", r.status_code, len(r.json()[0]))

print("LangSmith:", [p.name for p in Client().list_projects(limit=3)])
```

Run `uv run python check_keys.py`. Expect an answer from Groq, `200 384` from HuggingFace, and a list (possibly empty) from LangSmith. Delete the file afterwards.

## Step 9. Check the documents

`data/pdfs/` already holds five NIST PDFs (US government publications, free to use), with their sources in `data/pdfs/SOURCES.md`. They are text PDFs, so no OCR is needed. `data/datasets/golden.yaml` already holds 19 test questions for phase 8. Just check that the files are there.

## Step 10. First commit

```powershell
git add .
git commit -m "Phase 0: project setup"
```

## You are done when

- [ ] `uv run python check_keys.py` printed all three results.
- [ ] `data\pdfs\` holds five PDFs.
- [ ] `.env` is listed in `.gitignore`, and `git status` does not show it.
