# Phase 1: Search (dense, sparse, hybrid)

**Goal:** from a script, type a question and see the top 5 passages found by dense, sparse and hybrid search, with file and page.
**This phase teaches:** dense vs sparse search.

## The plan

```
PDFs -> pages -> chunks -> save to chunks.jsonl
                        -> embeddings (HuggingFace) -> save in Qdrant     (dense)
                        -> split into words -> BM25 in memory             (sparse)
```

## Step 1. Cut the PDFs into chunks (`app/ingest.py`)

For each PDF:

1. Read each page's text with `pypdf` (`PdfReader(path).pages[i].extract_text()`).
2. Split the page into chunks of about 800 characters, with a 100 character overlap, so a sentence cut at a boundary still appears whole in one chunk. Try to cut at a paragraph or sentence end.
3. Give each chunk: `id` (a number), `file`, `page` (1-based), `text`.
4. Write one JSON object per line to `data/chunks.jsonl`.

Keep it simple: no tables, no layout. If a page returns no text, skip it.

**Check:** print the number of chunks per file. Open `chunks.jsonl` and read three chunks. Do they look like normal sentences?

## Step 2. Get embeddings (`app/embeddings.py`)

One function: `embed(texts: list[str]) -> list[list[float]]`.

- Use `httpx.post` to `EMBED_URL` with `{"inputs": texts}` and the `Authorization: Bearer <HF_TOKEN>` header (the same call as your `check_keys.py`).
- Send **32 texts per request** and join the results.
- The reply is one list of 384 numbers per text.
- Retry once or twice on a 5xx error or a timeout, waiting a few seconds.

## Step 3. Store the vectors (end of `app/ingest.py`)

Use Qdrant in local mode, which needs no server or account:

```python
from qdrant_client import QdrantClient, models
client = QdrantClient(path=str(QDRANT_DIR))
client.recreate_collection(
    COLLECTION,
    vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
)
client.upsert(COLLECTION, points=[
    models.PointStruct(id=c["id"], vector=v, payload={"file": c["file"], "page": c["page"], "text": c["text"]})
    for c, v in zip(chunks, vectors)
])
```


Add `if __name__ == "__main__": ingest_all()` so you can run `uv run python -m app.ingest`. It reads every PDF in `data/pdfs`, writes `chunks.jsonl` and fills Qdrant.

**Check:** run it once. It takes a few minutes. Run `client.count(COLLECTION)`: the number should equal the number of lines in `chunks.jsonl`.

## Step 4. The three searches (`app/retrieve.py`)

Every function takes `(query, k=5)` and returns a list of dicts: `{id, file, page, text, score, rank}`.

**Dense:**
1. `embed([query])[0]` gives the question's numbers.
2. `client.query_points(COLLECTION, query=vector, limit=k)`. Qdrant returns the nearest chunks.

**Sparse (BM25):**
1. At startup, load `chunks.jsonl` and make a word list per chunk: lowercase, then `re.findall(r"\w+", text)`.
2. Build `BM25Okapi(list_of_word_lists)`.
3. For a query, split it the same way, then `bm25.get_scores(words)`. Take the top `k` by score.

**Hybrid (RRF):**
1. Get the top 20 from dense and the top 20 from sparse.
2. For each chunk, `score = 1/(60 + rank_in_dense) + 1/(60 + rank_in_sparse)` (use 0 for a list the chunk is not in). 60 is the usual constant.
3. Sort by score, return the top `k`.

Finally add one entry point: `search(query, mode, k=5)` where `mode` is `"dense"`, `"sparse"` or `"hybrid"`.

Qdrant's local mode lets only one program open the folder at a time. Create the client once at module level and reuse it. Do not run the ingest script while the API is running.

## Step 5. Try it

Create a temporary `try_search.py`:

```python
from app.retrieve import search
for q in ["What are the four functions of the AI RMF?",
          "SP 800-218 PS.1.1",
          "how do I keep my software from being tampered with"]:
    for mode in ["dense", "sparse", "hybrid"]:
        print("\n", q, "|", mode)
        for r in search(q, mode):
            print(f"  {r['file']} p{r['page']}  {r['text'][:80]!r}")
```

What you should notice:
- The exact code `PS.1.1` is found well by **sparse** (exact word match) and weakly by dense.
- The loose wording ("keep my software from being tampered with") is found better by **dense** (meaning).
- **Hybrid** is usually the safest.

This is the core lesson of the project. Write your observations in a notes file. In phase 10 you will measure it properly.

## You are done when

- [ ] `uv run python -m app.ingest` fills `chunks.jsonl` and Qdrant.
- [ ] `search(q, "dense")`, `search(q, "sparse")` and `search(q, "hybrid")` all return 5 results with file and page.
- [ ] You found at least one question where dense and sparse gave clearly different top results.
