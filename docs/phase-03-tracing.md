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

## Step 2. Almost nothing to mark

Because the app is a LangGraph graph built from LangChain parts, LangSmith traces it by itself once the environment variables are set:

- The graph run is the top of the tree. Every node (`condense`, `retrieve`, `generate`, `cite`, `remember`, and later the guards) is a child step.
- Inside a node, LangChain objects (the BM25 retriever, the vector store search, the prompt, the `ChatGroq` call) appear as their own steps, with run types `retriever`, `llm` and so on already set. The LLM step shows the exact prompt and the token counts.
- Use `@traceable` only for a plain Python function you want to see as its own step, such as a single guard check inside a node:

```python
from langsmith import traceable

@traceable(name="check_prompt_injection")
def check_prompt_injection(text): ...
```

## Step 3. Add tags and metadata

Tags and metadata let you filter traces later. With LangGraph you pass them in the `config` of the call:

```python
app_graph.invoke(
    {"question": question, "mode": mode},
    config={"tags": [f"mode:{mode}"], "metadata": {"mode": mode}},
)
```

Give `ask()` an optional `tags` argument (default: none) so callers can add their own, for example `source:ui` from the API and `source:eval` from the evaluation runner (phase 9). Merge them with the `mode:` tag before the call. Later phases add `guard:blocked` and `abstained`.

## Step 4. Return the trace id

The UI will need a link to the trace and, in phase 7, an id to attach feedback to.

- Create a UUID before the call: `run_id = uuid.uuid4()`.
- Pass it in the same config: `config={"run_id": run_id, "tags": [...], "metadata": {...}}`.
- Return it from `ask()` as `run_id`.

Traces are sent in the background. If your script ends instantly, call `from langsmith import Client; Client().flush()` or wait a second before exiting, so the last trace is not lost.

## Step 5. Look at the results

Open smith.langchain.com, the project `rag-evaluation-lab`, and run a few questions in each mode from `try_ask.py`.

Explore:
1. Click a trace. Open the `retrieve` step. Do you see the passages?
2. Open the `ChatGroq` step (the LLM call inside `generate`). Do you see the exact prompt? How many tokens?
3. Filter by tag `mode:dense`. Compare average latency with `mode:sparse`.
4. Open the **Monitor** tab of the project. You should see request counts, latency and token charts.
5. Find an answer that was wrong. Use the trace to see whether the search or the LLM was at fault. This is the main reason traces exist.

## Step 6. Keep your data private

Traces include the full passages and prompts. This is fine for public NIST documents. If you later use private documents, do not trace them to a shared LangSmith workspace.

## You are done when

- [ ] A question in each mode produces a trace with `retrieve` and `generate` steps, and a `ChatGroq` step inside `generate`.
- [ ] You can filter traces by `mode:dense`, `mode:sparse` and `mode:hybrid`.
- [ ] The Monitor tab shows your requests.
- [ ] `ask()` returns a `run_id`.
