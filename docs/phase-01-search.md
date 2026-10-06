# Phase 1: Search (dense, sparse, hybrid)

**Goal:** from a script, type a question and see the top 5 retrieved documents found by dense, sparse and hybrid search, with file and page.
**This phase teaches:** dense vs sparse search.

## The plan

```
PDFs -> pages -> chunks -> save to chunks.jsonl
                        -> embeddings (HuggingFace) -> save in Qdrant     (dense)
                        -> lowercase, split on spaces -> BM25 in memory             (sparse)
```

## Step 1. Cut the PDFs into chunks (`app/ingest.py`)

For each PDF:

1. Load the pages with LangChain's `PyPDFLoader(path).load()` (one Document per page; `metadata["page"]` is 0-based).
2. Split them with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(pages)`, so a sentence cut at a boundary still appears whole in one chunk. It tries paragraph breaks first, then lines, then spaces.
3. Give each chunk: `id` (a number), `file`, `page` (1-based), `text`.
4. Write one JSON object per line to `data/chunks.jsonl`.

Keep it simple: no tables, no layout. If a page returns no text, skip it.

**Check:** print the number of chunks per file. Open `chunks.jsonl` and read three chunks. Do they look like normal sentences?

## Step 2. Get embeddings (`app/embeddings.py`)

Use LangChain's `HuggingFaceEndpointEmbeddings(model=EMBED_MODEL, huggingfacehub_api_token=HF_TOKEN)` from `langchain-huggingface`. It calls the HuggingFace API (the local `HuggingFaceEmbeddings` would download PyTorch and run the model on your PC).

- `app/embeddings.py` is just that one object, called `embeddings`.
- The vector store calls `embeddings.embed_documents(texts)` for chunks; `embeddings.embed_query(text)` is used for a question (step 4).
- Each text becomes a list of 384 numbers.

## Step 3. Store the vectors (end of `app/ingest.py`)

Use LangChain's `QdrantVectorStore` (package `langchain-qdrant`) in local mode, which needs no server or account. One call creates the collection, embeds the chunks in batches and uploads them:

```python
shutil.rmtree(QDRANT_DIR, ignore_errors=True)  # on Windows Qdrant cannot delete its own open file
store = QdrantVectorStore.from_documents(
    chunks, embeddings, path=str(QDRANT_DIR), collection_name=COLLECTION
)
```

Here `chunks` are the LangChain Documents from step 1, with `id`, `file` and `page` in their metadata.

Add `if __name__ == "__main__": ingest_all()` so you can run `uv run python -m app.ingest`. It reads every PDF in `data/pdfs`, writes `chunks.jsonl` and fills Qdrant.

**Check:** run it once. It takes a few minutes. The script prints `store.client.count(COLLECTION).count`: the number should equal the number of lines in `chunks.jsonl`.

## Step 4. The three searches (`app/retrieve.py`)

All three searches are ready-made LangChain objects. Build them once and reuse them. You do not write any ranking code yourself.

**Dense:** open the saved Qdrant data with `QdrantVectorStore.from_existing_collection(embedding=embeddings, collection_name=COLLECTION, path=str(QDRANT_DIR))`. Then `store.similarity_search_with_score(query, k=k)` returns the best `k` chunks, each with a score (a cosine score: how close the meaning of the chunk is to the question). The store turns the question into numbers by itself, which is why `embeddings` is imported here too.

**Sparse (BM25):** `BM25Retriever.from_documents(docs, k=k, preprocess_func=words)` from `langchain-community` (it uses `rank-bm25` underneath). `docs` are the chunks rebuilt from `chunks.jsonl` as `Document(page_content=text, metadata={id, file, page})`. By default BM25 does not turn words into lowercase, so pass `preprocess_func=lambda t: t.lower().split()` (lowercase the text and cut it into words at the spaces). Punctuation stays attached to words (`functions,` is not the same as `functions`). But a code like `PS.1.1` stays one word, which is what makes exact-code search work. Write this weakness down for phase 10. Search with `retriever.invoke(query)`. It returns the documents only, with no scores. For the "is anything relevant" check in phase 2, `retriever.vectorizer.get_scores(query.lower().split())` gives the raw BM25 scores.

**Hybrid (RRF):** `EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.5, 0.5], c=60)` from `langchain_classic.retrievers`. It combines the two result lists by position: a chunk gets the points `1/(c + position)` from each list, and `c=60` is the usual constant. This method is called reciprocal rank fusion (RRF). Make the dense retriever with `store.as_retriever(search_kwargs={"k": 20})`, and the BM25 one with `k=20`. Keep the top `k` chunks of the combined list.

Finally add one function the rest of the app will use: `search(query, mode, k=5)`, where `mode` is `"dense"`, `"sparse"` or `"hybrid"`. It returns each result as a dictionary `{id, file, page, text, score, rank}` (`score` is only filled for dense). That way the rest of the app does not need to know which search ran.

Qdrant's local mode lets only one program open the folder at a time. Create the store once and reuse it. Do not run the ingest script while the API is running.

(Phase 4 moves the three build steps above into one function, `load_index()`, so the app can rebuild them after a PDF is uploaded or deleted. The searches work the same way. `try_search.py` is not affected.)

## Step 5. Try it

Create `try_search.py` at the project root. **Keep it** (it is not temporary). You will re-run it whenever you change the chunking, the embeddings or the search code, and in phase 10 it is the quick way to see what each search finds.

```python
from app.retrieve import search, store

for question in ["What are the four functions of the AI RMF?",
                 "SP 800-218 PS.1.1",
                 "how do I keep my software from being tampered with"]:
    for mode in ["dense", "sparse", "hybrid"]:
        print("\n", question, "|", mode)
        for result in search(question, mode):
            print(f"  {result['file']} p{result['page']}  {result['text'][:80]!r}")

# Release the Qdrant folder so Windows does not print an error when the script exits.
store.client.close()
```

Run it from the project root:

```powershell
uv run python try_search.py
```

It prints 9 blocks (3 questions x 3 modes), each with 5 lines of `file pN 'first 80 characters'`. Stop the API first if it is running (Qdrant's folder can be open in only one program).

What you should notice:
- The exact code `PS.1.1` is found well by **sparse** (exact word match) and weakly by dense.
- The loose wording ("keep my software from being tampered with") is found better by **dense** (meaning).
- **Hybrid** is usually the safest.

This is the core lesson of the project. Write your observations in a notes file. In phase 10 you will measure it properly.

## You are done when

Run each command from the project root (`C:\Users\asaur\Projects\rag-evaluation-lab`).

**Ingest** (`uv run python -m app.ingest`, a few minutes, uses HuggingFace credit, so run it only when needed):
- [ ] It prints a chunk count per PDF (5 files) and ends with `Qdrant count: N`.
- [ ] `N` equals the number of lines in `data\chunks.jsonl`. Check with `(Get-Content data\chunks.jsonl | Measure-Object -Line).Lines`.
- [ ] Read 3 lines of `chunks.jsonl` (for example `Get-Content data\chunks.jsonl -TotalCount 3`). The `text` looks like normal sentences, and `file` and `page` (1-based) are correct.
- [ ] Running ingest a second time gives the same `N` (no duplicates).

**Search** (`uv run python try_search.py`):
- [ ] The script runs without an error and prints 9 blocks (3 questions x 3 modes).
- [ ] Every block has exactly 5 results, each with a file and a page.
- [ ] `SP 800-218 PS.1.1`: **sparse** puts a chunk from `nist-sp-800-218-ssdf.pdf` that contains `PS.1.1` at or near the top, and dense does so less reliably.
- [ ] The loose-wording question (software tampering): **dense** finds sensible retrieved documents (SSDF or CSF), and sparse is weaker.
- [ ] At least one question where dense and sparse gave clearly different top results. Write down which question and what differed.
- [ ] **Hybrid** results include items from both the dense and the sparse lists.
- [ ] Calling `search(question, "banana")` (an unknown mode) fails with a clear error, not a silent empty list.
