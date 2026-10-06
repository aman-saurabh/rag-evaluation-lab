# CLAUDE.md: read this first

## What this project is

A small **practice** app to learn four things, and nothing else:

1. Dense vs sparse (BM25) vs hybrid search.
2. Guards (runtime checks that block bad questions and answers), built with the standard tools: LangGraph nodes, LangChain middleware and Groq's safety models. Custom code only for what those do not cover.
3. LangSmith evaluators (hallucination, toxicity, prompt injection, code injection, PII leakage, correctness and so on) from LangSmith's templates and the `openevals` library. Custom evaluators only for checks that depend on our own data (retrieval hit, abstention, citation validity).
4. LangSmith monitoring (traces, tags, latency, tokens, feedback, annotation queues).

The user asks questions about a few PDFs in a web page and an AI answers with sources (RAG). Everything is operated from the browser. The only things done in a terminal are starting the two servers.

Stack: **FastAPI** (backend) + **Streamlit** (UI, talks to the backend over HTTP only) + **LangChain** + **LangGraph** + **LangSmith** (with `openevals`) + **Groq** (LLM and safety models) + **HuggingFace API** (dense embeddings) + **rank-bm25** (sparse) + **local Qdrant** (a folder, no server) + **pypdf**.

The purpose is learning the LangChain, LangGraph and LangSmith ecosystem and the standard industry approaches. Use those three as much as possible.

## Why simplicity matters here

This app must stay small, working and understandable. A large setup (many packages, planning documents, licence checks, CI, pre-commit, OCR, multi-tenancy, packaging) ends with almost no working app. **Do not build that.**

## Simplicity rules (these override everything else)

1. **Build the smallest thing that works, then stop.** Do not add features, options, configuration or layers that the current phase does not ask for.
2. **No abstractions until they are needed twice.** No base classes, plugin systems, registries, factories, dependency injection, or generic "manager" classes. Plain functions and dictionaries are fine.
3. **One file per job, as listed in `docs/00-start-here.md`.** Do not split a file into a package. Aim for each file to stay readable in one sitting (roughly under 200 lines). If a file grows past that, say so rather than silently restructuring.
4. **Only the libraries in `docs/00-start-here.md`.** Ask before adding any other library.
4b. **Use LangChain, LangGraph and LangSmith features instead of hand-written code** wherever one exists (loaders, splitters, embeddings, vector stores, retrievers, `ChatGroq`, prompt templates, structured output, middleware, graph nodes and conditional edges, rate limiters, LangSmith feedback, datasets, evaluators, annotation queues). Plain Python only where none of them covers the need, and then say which feature you checked. This is not "adding abstractions"; it is the opposite.
4c. **The docs are not the truth.** `docs/` was written from the author's own understanding. If a doc disagrees with the official LangChain/LangGraph/LangSmith documentation or with the installed library, say so and fix the doc. Ask before relying on a doc claim that looks outdated.
5. **One phase at a time.** Do the steps of the current phase, tell the user what to run, and go through the "you are done when" list with them once they share the result. Do not start the next phase or build ahead.
6. **Testing means running the live app, and the user runs it.** Build the real code, then tell the user which command to run (servers, scripts, the browser) and wait for the result. Fix what breaks. Do not write unit tests, check scripts, CI, linters, pre-commit hooks, type-checking setup, Docker or packaging. Unit tests exist only in phase 11, which is optional practice at the very end, when the whole app is built and working, and only if the user asks for it.
7. **No new planning documents.** The docs already exist. If something in them is wrong or too complicated, say so and propose a simpler change, then edit the doc.
8. **Prefer deleting to adding.** If a step in the docs looks unnecessary, point it out and offer to skip it.
9. Guards and evaluators use the standard tools first: LangChain middleware (`PIIMiddleware`), LangGraph guard nodes, Groq safety models (`meta-llama/llama-prompt-guard-2-86m`, `openai/gpt-oss-safeguard-20b`), LangSmith evaluator templates and `openevals` prompts. Write a custom guard or evaluator only for what they do not cover. The point is to learn the tools and to measure what they catch and miss.
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
- `.env` with `GROQ_API_KEY`, `HF_TOKEN` and `LANGSMITH_API_KEY` (never print them), plus `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=rag-evaluation-lab`. `.env.example` and `.gitignore` exist.
- `data/pdfs/`: five NIST PDFs (text PDFs, free to reuse), sources in `data/pdfs/SOURCES.md`.
- `data/datasets/golden.yaml`: 19 checked questions (9 answerable, 6 unanswerable, 4 multi-document). Pages are 1-based PDF page numbers.
- **Done:** phases 0 to 5 (setup, search, answers with memory, tracing, FastAPI, Streamlit). **Next:** phase 6. Do not change the phase 0 to 5 docs; changes to earlier code are made inside the later phases that need them.

## Facts that were checked (2026-10-05)

- **Groq** works with this key. Chat models seen: `openai/gpt-oss-20b` (fast, used for answers) and `openai/gpt-oss-120b` (strong, used as judge). Both returned valid JSON with `response_format={"type": "json_object"}`. Limits seen in headers: **1000 requests per day, 8000 tokens per minute**. Pace evaluation runs.
- **HuggingFace embeddings** work: POST to `https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction` with `{"inputs": [...]}` and `Authorization: Bearer <HF_TOKEN>`. A batch of 100 texts worked. Returns 384 numbers per text. The free credit used was not measured, so embed once and save the results.
- **No free hosted reranker** exists on HuggingFace, so there is no reranker in this project.
- **LangSmith `langsmith` 0.14.4** (latest on 2026-10-05), checked in a scratch environment:
  - `Client.get_run_url(run=<Run>, project_name=...)` needs a run object: call `client.read_run(run_id)` first.
  - `Client.create_feedback(run_id, key=..., score=..., comment=...)` exists.
  - `evaluate(target, data=..., evaluators=[...], experiment_prefix=..., max_concurrency=1, metadata=...)` exists. An evaluator can be `def f(inputs, outputs, reference_outputs) -> {"key", "score", "comment"}`. This was run successfully with `upload_results=False`.
  - There is **no progress callback** in `evaluate()`, so count progress inside `target`.
  - Online evaluators are configured in the LangSmith web app (Tracing, then the project, then the Evaluators tab, then "+ Evaluator", then a template). **Confirmed by the user in the website (2026-10-07):** the evaluator templates (checked on PII leakage) offer Groq as the model provider and accept a Groq API key. Second option if that changes: an "OpenAI Compatible Endpoint" with `https://api.groq.com/openai/v1`.

## Facts checked on 2026-10-06 (from official docs, not yet run in this project)

- **Groq models on this key:** `meta-llama/llama-prompt-guard-2-22m`, `meta-llama/llama-prompt-guard-2-86m`, `openai/gpt-oss-safeguard-20b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and others. The safety models have their own limits.
- **LangChain `PIIMiddleware`** (`langchain.agents.middleware`): built-in types email, credit_card, ip, mac_address, url; custom types by regex or function; strategies redact, mask, hash, block; works inside `create_agent()` only. There are no built-in prompt-injection or toxicity guards. Custom middleware hooks: `before_agent`, `before_model`, `after_model`, `after_agent`, and `jump_to: "end"` to stop early.
- **`openevals`** has prompts for `HALLUCINATION`, `RAG_GROUNDEDNESS`, `RAG_RETRIEVAL_RELEVANCE`, `CORRECTNESS`, `TOXICITY`, `PII_LEAKAGE`, `PROMPT_INJECTION`, `CODE_INJECTION` and more; `create_llm_as_judge(prompt=..., judge=<LangChain chat model>, feedback_key=...)`; evaluators take `inputs`, `outputs`, `reference_outputs` and return `{"key", "score", "comment"}`. Read each prompt before using it. Read on 2026-10-06: `PROMPT_INJECTION` judges the input only (true = attack found); `CODE_INJECTION` judges the text in `{inputs}` (true = malicious code); `PII_LEAKAGE` uses `{inputs}` and `{outputs}` (true = private info, bad); `TOXICITY` grades `{outputs}` (true presumably = toxic); `HALLUCINATION` uses `{context}`, `{inputs}`, `{outputs}`, `{reference_outputs}` and `RAG_GROUNDEDNESS` uses `{context}`, `{outputs}` (true = supported, good).
- **LangSmith does not block live requests.** Online evaluators score a sample of traces afterwards.

## Phase list (see `docs/`)

0 Setup · 1 Search · 2 Answers · 3 Tracing · 4 FastAPI · 5 Streamlit · 6 Guards + Settings · 7 Feedback · 8 Datasets + evaluators · 9 Eval runs from the UI · 10 Findings · 11 Tests (optional).

If the user feels the project is growing too large, the cut list, in this order, is: the online evaluators in the LangSmith web app, the Datasets page editor (edit the YAML by hand instead), the Compare page, the Documents delete button. The core that must stay: phases 1 to 3, 6 (guards) and 8 (evaluators).
