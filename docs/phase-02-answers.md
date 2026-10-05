# Phase 2: Answers

**Goal:** from a script, ask a question and get an answer that cites its sources (file and page), or "I don't know".
**This phase teaches:** the LLM call and LangGraph.

## The plan

```
question -> search -> enough relevant text? -- no --> "I don't know"
                              |
                             yes -> LLM answers using only the passages -> answer + sources
```

(Guards are added in phase 6. Leave room for them but do not write them yet.)

## Step 1. The LLM call (`app/llm.py`)

One function: `chat(messages, model=FAST_MODEL, json_mode=False) -> str`.

- Use `Groq().chat.completions.create(model=..., messages=..., temperature=0)`.
- If `json_mode`, add `response_format={"type": "json_object"}`. You will need this for evaluators in phase 8.
- Handle a rate limit error (Groq returns 429): wait a few seconds and retry up to 3 times.
- Return `choices[0].message.content`.

Use the fast model for answers at first. The strong model is for judging in phase 8.

## Step 2. The prompt

In `app/graph.py`, write a function that builds the messages:

- **System message:** "You answer questions using only the numbered passages below. After each claim, cite the passage like [2]. If the passages do not contain the answer, reply exactly: I don't know."
- **User message:** the passages, each as `[1] (file, page N) text`, then the question.

Number the passages from 1. Later, citations like `[2]` are mapped back to the file and page.

## Step 3. Decide "enough relevant text"

Dense and BM25 scores are on different scales, so keep this rule simple:

- For `dense`: if the best score (cosine similarity) is below a threshold (start with 0.55), treat it as nothing relevant.
- For `sparse`: if the best BM25 score is 0 (no query word appears anywhere), treat it as nothing relevant.
- For `hybrid`: use the dense rule on the dense result.

Put the thresholds in `app/config.py`. You will tune them in phase 10 using the unanswerable questions. Do not worry about getting them perfect now.

## Step 4. Build the graph (`app/graph.py`)

LangGraph models the app as a **state** (a dictionary passed along), **nodes** (functions that update the state) and **edges** (arrows).

State fields: `question`, `mode`, `passages`, `answer`, `sources`, `abstained`.

Nodes:
1. `retrieve`: calls `search(question, mode)`, sets `passages`.
2. `abstain`: sets `answer = "I don't know."`, `abstained = True`.
3. `generate`: builds the prompt, calls `chat`, sets `answer`.
4. `cite`: reads `[n]` markers from the answer and sets `sources` to the matching passages (file, page, text).

Edges:
- start -> `retrieve`
- `retrieve` -> conditional: if not enough relevant text, go to `abstain`; otherwise go to `generate`
- `generate` -> `cite` -> end
- `abstain` -> end

LangGraph basics (`from langgraph.graph import StateGraph, START, END`):
- `g = StateGraph(MyStateTypedDict)`
- `g.add_node("retrieve", retrieve_fn)`
- `g.add_edge(START, "retrieve")`
- `g.add_conditional_edges("retrieve", router_fn, {"abstain": "abstain", "generate": "generate"})`
- `app_graph = g.compile()`, then `app_graph.invoke({"question": ..., "mode": ...})`

Expose a plain function `ask(question, mode) -> dict` that runs the graph and returns `answer`, `sources`, `abstained`, `passages`.

## Step 5. Try it

Temporary `try_ask.py`:

```python
from app.graph import ask
for q in ["What are the four functions of the NIST AI RMF?",
          "What was Acme's budget for Q4?"]:
    for mode in ["dense", "sparse", "hybrid"]:
        r = ask(q, mode)
        print(mode, "|", r["answer"][:200], "|", [(s["file"], s["page"]) for s in r["sources"]])
```

The first question should be answered with a correct source. The second has no answer in the documents, so it should abstain in at least some modes. If it makes something up, that is a hallucination. Note which mode did it. This is exactly what the evaluators will catch.

## You are done when

- [ ] A question from the documents returns a correct, cited answer in all three modes.
- [ ] A question that is not in the documents returns "I don't know" in at least one mode, and you know why the others did not.
- [ ] `ask()` returns `answer`, `sources`, `abstained` and `passages`.
