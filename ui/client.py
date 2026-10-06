import httpx

# The only place that knows the backend address.
BASE = "http://localhost:8000"

BACKEND_DOWN_MESSAGE = "Backend is not running. Start it with `uv run uvicorn api.main:app`."


def ask(question, mode, session_id):
    response = httpx.post(
        f"{BASE}/ask",
        json={"question": question, "mode": mode, "session_id": session_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def compare(question):
    response = httpx.post(f"{BASE}/compare", json={"question": question}, timeout=300)
    response.raise_for_status()
    return response.json()


def list_documents():
    response = httpx.get(f"{BASE}/documents", timeout=30)
    response.raise_for_status()
    return response.json()


def upload_document(file_name, content):
    # Indexing re-embeds every document, so this can take minutes.
    response = httpx.post(
        f"{BASE}/documents",
        files={"file": (file_name, content, "application/pdf")},
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def delete_document(name):
    response = httpx.delete(f"{BASE}/documents/{name}", timeout=1800)
    response.raise_for_status()
    return response.json()


def health():
    response = httpx.get(f"{BASE}/health", timeout=60)
    response.raise_for_status()
    return response.json()
