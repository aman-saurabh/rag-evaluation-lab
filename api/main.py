from fastapi import FastAPI
from groq import Groq
from langsmith import Client
from api import routes_ask, routes_datasets, routes_documents, routes_evals, routes_feedback, routes_settings
from app import retrieve as retrieval  # used as retrieval.store, so we always get the current object after a reload
from app.config import COLLECTION
from app.embeddings import embeddings

app = FastAPI(title="RAG Evaluation Lab")
app.include_router(routes_ask.router)
app.include_router(routes_documents.router)
app.include_router(routes_settings.router)
app.include_router(routes_feedback.router)
app.include_router(routes_datasets.router)
app.include_router(routes_evals.router)


@app.get("/health")
def health():
    """Tries one cheap call to each service. Only true or false is returned, never an error text (it could contain a key)."""
    status = {}

    try:
        Groq().models.list()  # lists models, so it uses no tokens
        status["groq"] = True
    except Exception:
        status["groq"] = False

    try:
        embeddings.embed_query("hello")  # embeds one word
        status["huggingface"] = True
    except Exception:
        status["huggingface"] = False

    try:
        retrieval.store.client.count(COLLECTION)
        status["qdrant"] = True
    except Exception:
        status["qdrant"] = False

    try:
        list(Client().list_projects(limit=1))
        status["langsmith"] = True
    except Exception:
        status["langsmith"] = False

    return status
