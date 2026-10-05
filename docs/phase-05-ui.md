# Phase 5: The web page (Streamlit)

**Goal:** a browser page where you chat with your documents, compare the three search modes and manage documents.
From now on you stop using scripts.

## What Streamlit gives you

You write a Python file and Streamlit turns it into a web page: `st.title`, `st.text_input`, `st.button`, `st.selectbox`. The whole file reruns from the top every time you click something. Anything you want to remember between clicks goes into `st.session_state`.

## The rule

**The UI never imports from `app/`.** It only talks to the backend with HTTP. This keeps the page replaceable.

## Step 1. The client (`ui/client.py`)

This is the only file that knows the backend address.

```python
import httpx

BASE = "http://localhost:8000"

def ask(question, mode):
    response = httpx.post(f"{BASE}/ask", json={"question": question, "mode": mode}, timeout=120)
    response.raise_for_status()
    return response.json()

def compare(question): ...
def list_documents(): ...
def upload_document(name, data): ...
def delete_document(name): ...
def health(): ...
```

One small function per endpoint. If the backend is down, catch `httpx.ConnectError` and show: "Backend is not running. Start it with `uv run uvicorn api.main:app`."

## Step 2. Chat page (`ui/Home.py`)

Layout:
- **Sidebar:** health lights for the four services (green/red), a mode selector (`dense`, `sparse`, `hybrid`).
- **Main area:** a chat. Use `st.chat_input` for the question and `st.chat_message` for each message. Keep the history in `st.session_state.messages`.
- For each answer, show:
  - the answer text,
  - an "Abstained" badge if `abstained` is true,
  - an expander **Sources**: file, page and the passage text for each source,
  - a link "Open trace in LangSmith" if `trace_url` is set,
  - the mode used.
- While waiting: `with st.spinner("Thinking..."):`.

## Step 3. Compare page (`ui/pages/1_Compare.py`)

- One text box, one button.
- On click: call `compare(question)`.
- Show three columns (`st.columns(3)`): dense, sparse, hybrid. Each column has the answer, the sources (file and page) and whether it abstained.
- This page is where you see the lesson from phase 1 live: a keyword question versus a meaning question.

## Step 4. Documents page (`ui/pages/2_Documents.py`)

- **Upload:** `st.file_uploader("PDF", type="pdf")` then a button "Add to index". Before sending, check `uploaded_file.size`: if it is larger than 20 MB (`20 * 1024 * 1024` bytes), show an error and do not call the API (the API does not check the size). Show a spinner and then the chunk count.
- **List:** a table of file name and chunk count from `list_documents()`.
- **Delete:** a button per row. Ask for confirmation first (a checkbox "I am sure").
- Say in the page text that re-indexing re-embeds everything and uses HuggingFace credit.

## Step 5. Run both servers

Two terminals:

```powershell
uv run uvicorn api.main:app --reload
uv run streamlit run ui/Home.py
```

Streamlit opens at `http://localhost:8501`. The pages in `ui/pages/` appear in the sidebar by themselves, ordered by the number in the file name.

These two commands are the only thing you do outside the browser for the rest of the project.

## Step 6. Use it

1. Upload one more PDF on the Documents page. Ask a question about it.
2. Ask a question in each mode on the Chat page.
3. Use Compare on "PS.1.1" (a code) and on a loosely worded question. Which mode wins?
4. Click "Open trace in LangSmith" and confirm the same answer is there.

## You are done when

- [ ] You can ask a question in the browser and get a cited answer.
- [ ] Compare shows three columns.
- [ ] You can upload and delete a PDF from the browser.
- [ ] The sidebar shows the health of the four services.
- [ ] Stopping the backend shows a clear message, not a Python error.
