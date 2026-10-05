import json
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from app.config import CHUNKS_FILE, QDRANT_DIR, COLLECTION
from app.embeddings import embeddings
from app.ingest import ingest_all

POOL = 20  # how many results dense and sparse each give to the hybrid merge

# The three search objects. load_index() fills them in, and again after documents change.
store = None    # dense: Qdrant (local folder). Only one program can open it at a time.
bm25 = None     # sparse: BM25 over the same chunks, built in memory from chunks.jsonl
hybrid = None   # hybrid: both lists merged with reciprocal rank fusion (1 / (60 + rank))


def load_index():
    # "global" is needed because we assign values to these variables in this function. Without it, Python would create a new local
    # variables in this function and the module-level variables(store, bm25, hybrid) above would never be updated whenever this function is called.
    global store, bm25, hybrid

    store = QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION, embedding=embeddings, path=str(QDRANT_DIR)
    )

    with open(CHUNKS_FILE, encoding="utf-8") as chunks_file:
        chunks = [json.loads(line) for line in chunks_file]

    documents = [
        Document(page_content=chunk["text"],
                 metadata={"id": chunk["id"], "file": chunk["file"], "page": chunk["page"]})
        for chunk in chunks
    ]

    # Lowercase and split on spaces. "PS.1.1" stays one word, so exact codes match well.
    bm25 = BM25Retriever.from_documents(
        documents, k=POOL, preprocess_func=lambda text: text.lower().split()
    )

    hybrid = EnsembleRetriever(
        retrievers=[store.as_retriever(search_kwargs={"k": POOL}), bm25],
        weights=[0.5, 0.5],
        c=60,
    )


def reload_index():
    """Re-reads all PDFs (ingest) and rebuilds the search objects. Call it after a PDF is added or deleted."""
    store.client.close()  # Windows cannot delete the Qdrant folder while it is open
    ingest_all()
    load_index()


def to_dict(document: Document, rank: int, score: float | None = None) -> dict:
    metadata = document.metadata
    return {"id": metadata["id"], "file": metadata["file"], "page": metadata["page"],
            "text": document.page_content, "score": score, "rank": rank}


def search(query: str, mode: str, k: int = 5) -> list[dict]:
    """mode is "dense", "sparse" or "hybrid". Only dense results have a score."""
    if mode == "dense":
        scored_documents = store.similarity_search_with_score(query, k=k)
        return [to_dict(document, rank, score)
                for rank, (document, score) in enumerate(scored_documents, start=1)]
    if mode == "sparse":
        found = bm25.invoke(query)[:k]
        return [to_dict(document, rank) for rank, document in enumerate(found, start=1)]
    if mode == "hybrid":
        found = hybrid.invoke(query)[:k]
        return [to_dict(document, rank) for rank, document in enumerate(found, start=1)]
    raise ValueError(f"Unknown mode {mode!r}: use 'dense', 'sparse' or 'hybrid'")


load_index()
