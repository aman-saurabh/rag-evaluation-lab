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


def feedback(run_id, score, comment):
    """score: 1 = thumbs up, 0 = thumbs down."""
    response = httpx.post(
        f"{BASE}/feedback",
        json={"run_id": run_id, "score": score, "comment": comment},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def list_datasets():
    response = httpx.get(f"{BASE}/datasets", timeout=30)
    response.raise_for_status()
    return response.json()


def get_dataset(name):
    response = httpx.get(f"{BASE}/datasets/{name}", timeout=30)
    response.raise_for_status()
    return response.json()


def save_dataset(name, items):
    response = httpx.put(f"{BASE}/datasets/{name}", json=items, timeout=30)
    response.raise_for_status()
    return response.json()


def eval_options():
    response = httpx.get(f"{BASE}/evals/options", timeout=30)
    response.raise_for_status()
    return response.json()


def start_eval(dataset, mode, evaluators):
    response = httpx.post(
        f"{BASE}/evals/run",
        json={"dataset": dataset, "mode": mode, "evaluators": evaluators},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def eval_status(job_id):
    response = httpx.get(f"{BASE}/evals/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_settings():
    response = httpx.get(f"{BASE}/settings", timeout=30)
    response.raise_for_status()
    return response.json()


def save_settings(settings):
    response = httpx.put(f"{BASE}/settings", json=settings, timeout=30)
    response.raise_for_status()
    return response.json()


def health():
    response = httpx.get(f"{BASE}/health", timeout=60)
    response.raise_for_status()
    return response.json()
