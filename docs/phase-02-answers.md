# Phase 2: Answers

**Goal:** from a script, ask a question and get an answer that cites its sources (file and page), or "I don't know".
**This phase teaches:** the LLM call and LangGraph.

## The plan

```
question -> search -> enough relevant text? -- no --> "I don't know"
                              |
                             yes -> LLM answers using only the retrieved documents -> answer + sources
```

(Guards are added in phase 6. Leave room for them but do not write them yet.)

## Step 1. The LLM call (`app/llm.py`)

Two LangChain chat models from `langchain-groq` (this package replaces calling `groq` directly; it installs `groq` itself):

```python
from langchain_groq import ChatGroq
fast_llm = ChatGroq(model=FAST_MODEL, temperature=0, max_retries=3, max_tokens=2000)
strong_llm = ChatGroq(model=STRONG_MODEL, temperature=0, max_retries=3, max_tokens=2000)
```

- `max_retries=3` already retries on a rate limit error (Groq returns 429). No retry code to write.
- Keep `max_tokens` generous. The `gpt-oss` models spend part of it on hidden reasoning, and a small value gives an empty answer (seen in `check_keys.py`).
- Call with `fast_llm.invoke(messages)`. The answer text is `.content`.
- For judges in phase 8, use `strong_llm.with_structured_output(...)` (see phase 8).

Use the fast model for answers at first. The strong model is for judging in phase 8.

## Step 2. The prompt

In `app/graph.py`, use a `ChatPromptTemplate` and a small chain:

```python
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Earlier conversation:\n{history}\n\nRetrieved documents:\n{retrieved_docs_text}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | fast_llm | StrOutputParser()
```

`StrOutputParser()` turns the model's reply into plain text, so `answer_chain.invoke(...)` returns a string. A second prompt and chain, `condense_prompt` and `condense_chain`, rewrite follow-up questions (see step 4b).

- **System message:** "You answer questions using only the numbered documents below. After each claim, cite the document like [2], with plain square brackets. If the documents do not contain the answer, reply exactly: I don't know. The earlier conversation is only for understanding follow-up questions; never use it as a source."
- **Retrieved documents text:** each as `[1] (file, page N) text`, joined with blank lines.

"Retrieved documents" means the top search results (chunks of the PDFs). Number them from 1. The `[2]` in the answer is the position of the document in this one search result (not a page number or a file number). It is mapped back to the file and page in the `cite` node. No regex is needed: for each document number, check whether `f"[{doc_number}]"` appears in the answer.

## Step 3. Decide "enough relevant text"

Dense and BM25 scores are on different scales, so keep this rule simple:

- For `dense`: if the best score (cosine similarity) is below a threshold (start with 0.55), treat it as nothing relevant.
- For `sparse`: if the best BM25 score (`retriever.vectorizer.get_scores(...)`, see phase 1) is 0 (no query word appears anywhere), treat it as nothing relevant.
- For `hybrid`: use the dense rule on the dense result.

Put the threshold as a constant (`DENSE_MIN_SCORE = 0.55`) at the top of `app/graph.py`, because only the graph uses it (phase 6 moves it into the settings file). You will tune it in phase 10 using the unanswerable questions. Do not worry about getting them perfect now.

## Step 4. Build the graph (`app/graph.py`)

LangGraph models the app as a **state** (a dictionary passed along), **nodes** (functions that update the state) and **edges** (arrows).

State fields: `question`, `mode`, `history`, `search_query`, `retrieved_docs`, `relevant`, `answer`, `sources`.

Nodes:
1. `condense`: rewrites a follow-up question so it makes sense alone (see step 4b).
2. `retrieve`: calls `search(search_query, mode)`, sets `retrieved_docs` and `relevant`.
3. `abstain`: sets `answer = "I don't know."` and empty `sources`. It takes the state as `_state` (LangGraph always passes it; the `_` shows it is unused).
4. `generate`: runs `answer_chain.invoke(...)`, sets `answer`.
5. `cite`: sets `sources` to the retrieved documents whose number (like `[2]`) appears in the answer (file, page, text), using a plain `for` loop and `if`.
6. `remember`: saves the turn in `history` (see step 4b).

Edges:
- start -> `condense` -> `retrieve`
- `retrieve` -> conditional: if not enough relevant text, go to `abstain`; otherwise go to `generate`
- `generate` -> `cite` -> `remember` -> end
- `abstain` -> `remember` -> end

LangGraph basics (`from langgraph.graph import StateGraph, START, END`):
- `graph_builder = StateGraph(MyStateTypedDict)`
- `graph_builder.add_node("retrieve", retrieve_function)`
- `graph_builder.add_edge(START, "retrieve")`
- `graph_builder.add_conditional_edges("retrieve", route_after_retrieve)`, where `route_after_retrieve(state)` is a normal function with an `if` that returns the name of the next node (`"generate"` or `"abstain"`)
- `app_graph = graph_builder.compile(checkpointer=InMemorySaver())`, then `app_graph.invoke({"question": ..., "mode": ...})`

Expose a plain function `ask(question, mode) -> dict` that runs the graph and returns `answer`, `sources`, `abstained`, `retrieved_docs`, `relevant` and `search_query` (the last two help you check why it answered or abstained). `abstained` is true when the answer starts with "I don't know" (it covers both the `abstain` node and the model abstaining itself).

## Step 4b. Memory (sessions)

The chat remembers earlier turns so follow-ups like "What is its purpose?" work.

- `ask(question, mode, session_id=None)`: the `session_id` is the LangGraph `thread_id`. Same id = same conversation. No id = a fresh chat every time (use this for evaluation runs).
- State gets a `history` field (the last 3 question/answer pairs). `remember` adds the new turn to the list, then `history[-3:]` keeps only the last 3 items (a negative number in a slice counts from the end), so old turns are dropped. `InMemorySaver` keeps it per session, so a backend restart forgets everything (fine for practice).
- New node `condense` (first): if there is history, `condense_chain` (the fast LLM) rewrites the current question to stand alone ("What is its purpose?" becomes "What is the purpose of the GOVERN function?") so search works. The first question skips this call.
- New node `remember` (last): appends the turn to `history`.
- History goes into the answer prompt only to understand the question, never as a source.
- Phase 4 will take `session_id` in `/ask`; phase 5 will make Streamlit send one id per browser session.

## Step 5. Try it

Create `try_ask.py` at the project root. **Keep it**; you will re-run it after every change to the graph, the prompt or the thresholds:

```python
from app.graph import ask
from app.retrieve import store

for question in ["What are the four functions of the NIST AI RMF?",
                 "What was Acme's budget for Q4?"]:
    for mode in ["dense", "sparse", "hybrid"]:
        result = ask(question, mode)
        sources = [(source["file"], source["page"]) for source in result["sources"]]
        print(question)
        print(mode, "|", result["answer"][:200], "|", sources)

# Memory: the second question only makes sense with the first ("its" means the GOVERN function).
print("\n-- follow-up in one session (memory ON) --")
for question in ["What is the GOVERN function in the NIST AI RMF?",
                 "What is its purpose?",
                 "And how does the MAP function differ from it?"]:
    result = ask(question, "hybrid", session_id="demo")
    sources = [(source["file"], source["page"]) for source in result["sources"]]
    print(question)
    print("  searched for:", result["search_query"])
    print("  relevant text found:", result["relevant"])
    print("  hybrid", "|", result["answer"][:200], "|", sources)

# Control: the same follow-up without a session. With no memory it cannot know what "its" means.
print("\n-- same follow-up, no session (memory OFF) --")
result = ask("What is its purpose?", "hybrid")
print("  searched for:", result["search_query"])
print("  relevant text found:", result["relevant"])
print("  hybrid", "|", result["answer"][:200])

# Release the Qdrant folder so Windows does not print an error when the script exits.
store.client.close()
```

Run it from the project root (stop the API first if it is running):

```powershell
uv run python try_ask.py
```

It prints 6 lines (2 questions x 3 modes), then the follow-up section. For each follow-up it prints the question, what the app actually searched for (`searched for`), whether relevant text was found, and the answer.

The first question should be answered with a correct source. The second has no answer in the documents, so it should abstain in at least some modes. If it makes something up, that is a hallucination. Note which mode did it. This is exactly what the evaluators will catch.

### If a follow-up says "I don't know", which part failed?

1. `searched for` still looks like the original vague question: the rewrite (`condense`) did not use the history. Check the session id is the same in both calls.
2. `searched for` is a good standalone question but `relevant text found` is `False`: the search score is below `DENSE_MIN_SCORE` at the top of `graph.py` (step 3). Lower the threshold or try another mode.
3. `relevant text found` is `True` but the answer is still "I don't know": the model decided the retrieved text does not contain the answer. Read the retrieved documents. If the answer really is not there, the model is right.

## You are done when

Run `uv run python try_ask.py` from the project root.

- [ ] The script runs without an error and prints 6 lines.
- [ ] The AI RMF question returns a non-empty answer with a citation like `[1]` in all three modes, and the printed sources include `nist-ai-rmf-1.0.pdf` page 8 (or a page next to it).
- [ ] The Acme question returns "I don't know" in at least one mode, and you know why the others did not (look at the threshold rule in step 3).
- [ ] Memory works: in the "memory ON" part, `searched for` for "What is its purpose?" mentions GOVERN (the app rewrote the question using the first one). In the "memory OFF" part, `searched for` is just "What is its purpose?" and the answer is "I don't know" or off-topic.
- [ ] `ask()` returns the keys `answer`, `sources`, `abstained`, `retrieved_docs`, `relevant` and `search_query`.
- [ ] Each source's page matches the passage the answer actually used: open the PDF at that page and check one answer by eye.
