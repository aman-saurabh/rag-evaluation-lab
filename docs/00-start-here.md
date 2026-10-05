# Start here

Read this once. Every later phase uses these words.

## The idea in one minute

You give the app some PDFs. When you ask a question:

1. The app **searches** the PDFs for the few passages most likely to hold the answer.
2. It gives those passages to an AI model and says: "answer using only these".
3. The AI writes an answer and names its sources (file and page).

This is called **RAG** (retrieval-augmented generation). "Retrieval" is step 1. "Generation" is step 2 and 3. RAG is used so the AI answers from your documents and does not make things up.

## Words you will meet

| Word | Plain meaning |
|---|---|
| **Chunk** | A small piece of a document, about one paragraph. We search chunks, not whole PDFs. |
| **Dense search** | Each chunk is turned into a list of numbers (an **embedding**) that captures its *meaning*. A question is turned into numbers the same way. Chunks whose numbers are close to the question's numbers are returned. It finds "car" when you ask about "automobile". |
| **Sparse search** | Matches the *exact words*. We use **BM25**, the classic keyword-ranking method. It finds the chunk containing "CSF 2.0" when you type exactly that. It misses synonyms. |
| **Hybrid search** | Run both and merge the two result lists (we use **RRF**, reciprocal rank fusion: a chunk ranked high in both lists wins). |
| **Embedding** | The list of numbers that stands for a text. We get them from a HuggingFace model over the internet. |
| **Vector store** | A database for embeddings. We use **Qdrant**, running locally in a folder (no account needed). |
| **LLM** | The AI model that writes the answer. We use Groq, which has a free tier. |
| **LangGraph** | A library to describe the app as boxes and arrows (a **graph**): check the question, search, answer, check the answer. |
| **Trace** | A recorded history of one request: every step, its input, its output and how long it took. |
| **LangSmith** | A website that stores traces, runs evaluations, and shows charts. This is the **monitoring** part. |
| **Guard** | A check that runs on every live request and can **block** it. Example: block a question that tries to trick the AI. |
| **Evaluator** | A scorer that runs *after* an answer exists and gives it a number. Example: "was this answer made up? 0 or 1". |
| **Dataset** | A list of test questions with the right answers, kept in LangSmith. |
| **Experiment** | One run of the dataset through the app, with all the evaluator scores. You compare experiments to see what is better. |
| **Hallucination** | When the AI states something the documents do not support. |
| **Prompt injection** | A question or a document containing text like "ignore your instructions and...". It tries to take control of the AI. |
| **Code injection** | Text that tries to make the app produce or run harmful code or commands. |
| **Toxicity** | Rude, hateful or abusive language. |
| **Abstain** | Saying "I don't know" when the documents do not contain the answer. This is a good behaviour. |

## Guards vs evaluators (the most important difference)

| | Guard | Evaluator |
|---|---|---|
| When | On every live request, *before* the answer is shown | *After*, on saved test questions or sampled live traces |
| Job | **Stop** bad things | **Measure** how good or safe the app is |
| Output | Allow or block | A score |
| Example | Block "ignore previous instructions" | Score how many of 20 attack prompts the app resisted |

You need both. Guards protect users today. Evaluators tell you whether the guards and the search are actually working.

## How the app is built

```
Streamlit page  --HTTP-->  FastAPI backend  -->  app/ code  -->  Groq, HuggingFace, Qdrant
 (what you see)           (the doorway)        (the real work)           |
                                                    +---------------> LangSmith (traces, scores)
```

- **Streamlit** is the web page. It never touches the search or the AI. It only calls the backend.
- **FastAPI** is the backend. It has a few URLs (endpoints) such as `/ask`.
- **`app/`** holds the real work: reading PDFs, searching, guards, the graph.
- Keeping them apart means you can change the page without breaking the logic.

## The question's journey

```
question -> guard_in -> search -> guard_passages -> enough relevant text?
                                                      |no -> "I don't know"
                                                      |yes -> LLM answer -> guard_out -> shown to user
```

Every box above is recorded as a step in the LangSmith trace.

## Folder layout (final)

```
rag-evaluation-lab/
  README.md
  docs/                      these files
  .env                       your keys (never share or commit)
  .env.example               the same without the keys
  pyproject.toml             the list of libraries
  data/
    pdfs/                    the documents (five NIST PDFs to start with)
    chunks.jsonl             the chunks (written by ingest)
    qdrant/                  the vector database (written by ingest)
    datasets/                golden.yaml, attacks_prompt.yaml, attacks_code.yaml
    settings.json            which guards are on
  app/
    config.py                reads .env, holds constants
    ingest.py                PDF -> chunks -> embeddings
    embeddings.py            calls the HuggingFace API
    retrieve.py              dense, sparse, hybrid
    llm.py                   calls Groq
    guards.py                all the guards
    graph.py                 the LangGraph flow
    evaluators.py            all the evaluators
    eval_runner.py           runs a dataset through the app
  api/
    main.py                  starts FastAPI
    schemas.py               the shape of requests and replies
    routes_ask.py  routes_documents.py  routes_settings.py
    routes_datasets.py  routes_evals.py  routes_feedback.py
  ui/
    Home.py                  the Chat page
    client.py                the only file that knows the backend URL
    pages/
      1_Compare.py  2_Documents.py  3_Settings.py  4_Datasets.py  5_Evaluations.py
  tests/                     written in phase 11
```

You will create these files phase by phase. Do not create them all upfront.

## Libraries (only what is needed)

| Library | Used for |
|---|---|
| `fastapi`, `uvicorn`, `python-multipart` | The backend and file uploads |
| `streamlit` | The web page |
| `httpx` | The page calls the backend; the backend calls HuggingFace |
| `langgraph` | The graph |
| `langsmith` | Traces, datasets, evaluators, feedback |
| `groq` | The AI model |
| `rank-bm25` | Sparse search |
| `qdrant-client` | Dense search store |
| `pypdf` | Read text from PDFs |
| `pyyaml` | Read and write the dataset files |
| `python-dotenv` | Read `.env` |
| `pytest` | Tests (phase 11) |

## Free-tier limits to remember

- **Groq:** about 8,000 tokens per minute and 1,000 requests per day (measured on 2026-10-05 for `openai/gpt-oss-20b` and `openai/gpt-oss-120b`). An evaluation run uses many calls, so the runner will pause between them.
- **LangSmith:** the free plan limits traces per month. Evaluate on the test set freely, but do not score every live trace.
- **HuggingFace:** the free credit is small. Embed each document once (we save the results) and do not re-embed on every run.

## Rules we follow

1. Small steps. Every phase ends with something you can run.
2. Change one thing, run it, then move on.
3. No tests until phase 11. Instead, each phase has a "you are done when" check you do by hand.
4. Never put a key in a file that is not `.env`.
