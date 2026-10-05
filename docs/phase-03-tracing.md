# Phase 3: Tracing and monitoring

**Goal:** every question appears in LangSmith as one trace with a step for each part (search, answer), tagged by search mode.
**This phase teaches:** LangSmith tracing and monitoring.

## What a trace is

A trace is the recorded history of one request. In LangSmith you click a trace and see a tree: the whole `ask` call at the top, and inside it `retrieve`, `generate`, and so on, each with its input, output, duration and (for LLM calls) token count.

## Step 1. Turn tracing on

Your `.env` already has:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rag-evaluation-lab
```

`app/config.py` calls `load_dotenv()`, so these are in the environment when the app starts. LangSmith reads them by itself. Make sure `app.config` is imported before any traced function runs.

## Step 2. Mark the functions with `@traceable`

```python
from langsmith import traceable

@traceable(name="retrieve", run_type="retriever")
def retrieve_node(state): ...

@traceable(name="llm_answer", run_type="llm")
def chat(messages, ...): ...

@traceable(name="ask", run_type="chain")
def ask(question, mode): ...
```

- `run_type` tells LangSmith how to draw the step. Use `retriever` for search, `llm` for model calls and `chain` for the rest.
- Nested calls are linked automatically. A traced function called from inside another traced function becomes its child.
- For the Groq call, LangSmith can also wrap the client directly (`from langsmith.wrappers import wrap_openai` works for OpenAI-compatible clients). Check the current LangSmith docs for the Groq-specific option. If the wrapper is awkward, `@traceable(run_type="llm")` plus the metadata below is enough.

## Step 3. Add tags and metadata

Tags and metadata let you filter traces later.

Pass them at call time, through `langsmith_extra`:

```python
ask(question, mode, langsmith_extra={"tags": [f"mode:{mode}"], "metadata": {"mode": mode}})
```

Give `ask()` an optional `tags` argument (default: none) so callers can add their own, for example `source:ui` from the API and `source:eval` from the evaluation runner (phase 9). Merge them with the `mode:` tag before the call. Later phases add `guard:blocked` and `abstained`.

## Step 4. Return the trace id

The UI will need a link to the trace and, in phase 7, an id to attach feedback to.

- Create a UUID before the call: `run_id = uuid.uuid4()`.
- Pass it in: `ask(..., langsmith_extra={"run_id": run_id, ...})`.
- Return it from `ask()` as `run_id`.

Traces are sent in the background. If your script ends instantly, call `from langsmith import Client; Client().flush()` or wait a second before exiting, so the last trace is not lost.

## Step 5. Look at the results

Open smith.langchain.com, the project `rag-evaluation-lab`, and run a few questions in each mode from `try_ask.py`.

Explore:
1. Click a trace. Open the `retrieve` step. Do you see the passages?
2. Open the `llm_answer` step. Do you see the exact prompt? How many tokens?
3. Filter by tag `mode:dense`. Compare average latency with `mode:sparse`.
4. Open the **Monitor** tab of the project. You should see request counts, latency and token charts.
5. Find an answer that was wrong. Use the trace to see whether the search or the LLM was at fault. This is the main reason traces exist.

## Step 6. Keep your data private

Traces include the full passages and prompts. This is fine for public NIST documents. If you later use private documents, do not trace them to a shared LangSmith workspace.

## You are done when

- [ ] A question in each mode produces a trace with `ask`, `retrieve` and `llm_answer` steps.
- [ ] You can filter traces by `mode:dense`, `mode:sparse` and `mode:hybrid`.
- [ ] The Monitor tab shows your requests.
- [ ] `ask()` returns a `run_id`.
