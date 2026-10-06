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
- Inside a node, the LangChain parts (the BM25 search, the vector store search, the prompt, the `ChatGroq` call) appear as their own steps, already labelled (`retriever`, `llm` and so on). The LLM step shows the exact prompt and the number of tokens.
- Use `@traceable` only for a plain Python function you want to see as its own step, such as a single guard check inside a node:

```python
from langsmith import traceable

@traceable(name="check_prompt_injection")
def check_prompt_injection(text): ...
```

## Step 3. Add tags

Tags let you filter traces later. (LangSmith also has "metadata", key and value pairs for the same purpose. We use tags only, so the mode is not stored twice.) With LangGraph you pass tags in the `config` of the call. **It is one config dictionary**: it already holds the `thread_id` for memory (phase 2), so tracing adds to it and does not replace it:

```python
config = {
    "configurable": {"thread_id": session_id or f"no-session-{run_id}"},  # memory (phase 2): new chat if no session
    "run_id": run_id,                                                       # tracing (step 4)
    "tags": [f"mode:{mode}"],
}
app_graph.invoke({"question": question, "mode": mode}, config)
```

Give `ask()` an optional `tags` argument (default: none) so callers can add their own, for example `source:ui` from the API and `source:eval` from the evaluation runner (phase 9). If the caller passed no tags (`tags is None`), use an empty list, then build the full list as `[f"mode:{mode}"] + tags`. Later phases add `guard:blocked` and `abstained`.

## Step 4. Return the trace id

The UI will need a link to the trace and, in phase 7, an id to attach feedback to.

- Create a unique id before the call: `run_id = uuid.uuid4()` (a UUID, a long random id).
- Pass it in the same config (see step 3).
- Return it from `ask()` as `run_id` (as text: `str(run_id)`).

Traces are sent in the background. If your script ends instantly, the last trace can be lost. In `try_ask.py`, add `print("    trace id:", result["run_id"])` to `show`, and at the end call `wait_for_all_tracers()` (`from langchain_core.tracers.langchain import wait_for_all_tracers`) so it waits until everything is sent. (The FastAPI server keeps running, so it does not need this.)

## Step 5. Look at the results

Open smith.langchain.com, the project `rag-evaluation-lab`, and run a few questions in each mode from `try_ask.py`.

Explore:
1. Click a trace. Open the `retrieve` step. Do you see the retrieved documents?
2. Open the `ChatGroq` step (the LLM call inside `generate`). Do you see the exact prompt? How many tokens?
3. Compare a first question with a follow-up from the "memory ON" part of `try_ask.py`. The follow-up has an extra `ChatGroq` step inside `condense` (the question rewrite). Its input shows the conversation, and its output is the standalone question the search used.
4. Filter by tag `mode:dense`. Compare average latency with `mode:sparse`.
5. Open the **Monitor** tab of the project. You should see request counts, latency and token charts.
6. Open the **Threads** view of the project. Because `ask()` passes a `thread_id`, the three "memory ON" questions are grouped as one conversation.
7. Find an answer that was wrong. Use the trace to see whether the search or the LLM was at fault. This is the main reason traces exist.

## Step 6. Keep your data private

Traces include the full retrieved documents and prompts. This is fine for public NIST documents. If you later use private documents, do not trace them to a shared LangSmith workspace.

## You are done when

- [ ] A question in each mode produces a trace with `retrieve` and `generate` steps, and a `ChatGroq` step inside `generate`.
- [ ] You can filter traces by `mode:dense`, `mode:sparse` and `mode:hybrid`.
- [ ] The Monitor tab shows your requests.
- [ ] `ask()` returns a `run_id`, and `try_ask.py` prints a `trace id` that you can find in LangSmith.
- [ ] Memory still works after adding tracing (the "memory ON" follow-up still searches for GOVERN), because the `thread_id` is still in the config.
