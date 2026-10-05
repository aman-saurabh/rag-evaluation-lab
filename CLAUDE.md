# CLAUDE.md: read this first

## What this project is

A small **practice** app to learn four things, and nothing else:

1. Dense vs sparse (BM25) vs hybrid search.
2. Guards (runtime checks that block bad questions and answers).
3. LangSmith evaluators (hallucination, toxicity, prompt injection, code injection, retrieval hit and so on).
4. LangSmith monitoring (traces, tags, latency, tokens, feedback).

The user asks questions about a few PDFs in a web page and an AI answers with sources (RAG). Everything is operated from the browser. The only things done in a terminal are starting the two servers.

Stack: **FastAPI** (backend) + **Streamlit** (UI, talks to the backend over HTTP only) + **LangGraph** + **LangSmith** + **Groq** (LLM) + **HuggingFace API** (dense embeddings) + **rank-bm25** (sparse) + **local Qdrant** (a folder, no server) + **pypdf**.

## Why this project exists: the lesson from the last one

The user spent many days on an earlier, much larger project (`C:\Users\asaur\Projects\ragsqaud`, three Python packages, planning docs, licence checks, CI, pre-commit, OCR, Docling, multi-tenancy, packaging) and ended up with almost no working app. They scrapped it for being **overly complicated**. **Do not repeat that.** That old folder is not to be used or edited.

## Simplicity rules (these override everything else)

1. **Build the smallest thing that works, then stop.** Do not add features, options, configuration or layers that the current phase does not ask for.
2. **No abstractions until they are needed twice.** No base classes, plugin systems, registries, factories, dependency injection, or generic "manager" classes. Plain functions and dictionaries are fine.
3. **One file per job, as listed in `docs/00-start-here.md`.** Do not split a file into a package. Aim for each file to stay readable in one sitting (roughly under 200 lines). If a file grows past that, say so rather than silently restructuring.
4. **Only the libraries in `docs/00-start-here.md`.** Ask before adding any other library.
4b. **Use LangChain components instead of hand-written code** wherever one exists (loaders, splitters, embeddings, vector stores, retrievers, `ChatGroq`, prompt templates, structured output). Plain Python only where it is the lesson (the crude guards) or no component exists. This is not "adding abstractions"; it is the opposite.
5. **One phase at a time.** Do the steps of the current phase, tell the user what to run, and go through the "you are done when" list with them once they share the result. Do not start the next phase or build ahead.
6. **Testing means running the live app, and the user runs it.** Build the real code, then tell the user which command to run (servers, scripts, the browser) and wait for the result. Fix what breaks. Do not write unit tests, check scripts, CI, linters, pre-commit hooks, type-checking setup, Docker or packaging. Unit tests exist only in phase 11, which is optional practice at the very end, when the whole app is built and working, and only if the user asks for it.
7. **No new planning documents.** The docs already exist. If something in them is wrong or too complicated, say so and propose a simpler change, then edit the doc.
8. **Prefer deleting to adding.** If a step in the docs looks unnecessary, point it out and offer to skip it.
9. Keep guards and evaluators crude on purpose (phrase lists, regex, one-question LLM judges). The point is to learn what they catch and miss.
9b. **Write simple code a beginner can read line by line.** Use plain `for` loops and `if` statements, one step per line, and meaningful names. No clever one-liners: no long comprehensions, no nested conditional expressions, no `lambda`, no dense slicing or chained calls. If a line needs explaining, split it into several lines or add a short comment. Readable beats short.
10. Explain things in plain language. The user is learning. When you write code, say in a sentence or two what it does and why.

## How to work with the user

- Start by reading `README.md`, then `docs/00-start-here.md`, then the phase the user says they are on. If unsure which phase they are on, look at which files exist and ask.
- Use `uv` for everything (`uv add`, `uv run`). Python 3.13. Windows 11 with PowerShell.
- Commit only when the user asks. Never put a secret in any file except `.env` (which is gitignored).
- The user does not want to use Claude Code or MCP *inside* the app. There is no MCP server in this project.
- If something fails, show the real error, explain it simply, and fix the smallest thing. Do not rewrite working code.
- **Never run anything without the user's permission** (no `uv run`, scripts, servers, `uv add`, ingest, or any other command; each call also uses the free Groq, HuggingFace and LangSmith quotas). Edit files, then tell the user the exact command to run. Run it yourself only when the user asks you to.
- Do not write separate check or test scripts for library calls. Write the real code, ask the user to run it, and fix errors when they share them.

## Already prepared (do not redo)

- `README.md` and `docs/`: the full plan, phase 0 to 11, step by step.
- `.env` with `GROQ_API_KEY`, `HF_TOKEN` and `LANGSMITH_API_KEY` (copied from the old project; never print them), plus `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=rag-evaluation-lab`. `.env.example` and `.gitignore` exist.
- `data/pdfs/`: five NIST PDFs (text PDFs, free to reuse), sources in `data/pdfs/SOURCES.md`.
- `data/datasets/golden.yaml`: 19 checked questions (9 answerable, 6 unanswerable, 4 multi-document). Pages are 1-based PDF page numbers.
- **Not done yet:** no code, no `pyproject.toml`, no git repository, no virtual environment. Phase 0 creates those.

## Facts that were checked (2026-10-05)

- **Groq** works with this key. Chat models seen: `openai/gpt-oss-20b` (fast, used for answers) and `openai/gpt-oss-120b` (strong, used as judge). Both returned valid JSON with `response_format={"type": "json_object"}`. Limits seen in headers: **1000 requests per day, 8000 tokens per minute**. Pace evaluation runs.
- **HuggingFace embeddings** work: POST to `https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction` with `{"inputs": [...]}` and `Authorization: Bearer <HF_TOKEN>`. A batch of 100 texts worked. Returns 384 numbers per text. The free credit used was not measured, so embed once and save the results.
- **No free hosted reranker** exists on HuggingFace, so there is no reranker in this project.
- **LangSmith `langsmith` 0.14.4** (latest on 2026-10-05), checked in a scratch environment:
  - `Client.get_run_url(run=<Run>, project_name=...)` needs a run object: call `client.read_run(run_id)` first.
  - `Client.create_feedback(run_id, key=..., score=..., comment=...)` exists.
  - `evaluate(target, data=..., evaluators=[...], experiment_prefix=..., max_concurrency=1, metadata=...)` exists. An evaluator can be `def f(inputs, outputs, reference_outputs) -> {"key", "score", "comment"}`. This was run successfully with `upload_results=False`.
  - There is **no progress callback** in `evaluate()`, so count progress inside `target`.
  - Not checked: the website's online-evaluator (automation rules) menu. Look for it when you get to phase 8 step 7. It is configured in the LangSmith web app, not in code.

## Phase list (see `docs/`)

0 Setup · 1 Search · 2 Answers · 3 Tracing · 4 FastAPI · 5 Streamlit · 6 Guards + Settings · 7 Feedback · 8 Datasets + evaluators · 9 Eval runs from the UI · 10 Findings · 11 Tests (optional).

If the user feels the project is growing too large, the cut list, in this order, is: phase 9's history table, the Datasets page editor (edit the YAML by hand instead), the Compare page, the Documents delete button. The core that must stay: phases 1 to 3, 6 (guards) and 8 (evaluators).
