import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile
from api.schemas import DocumentInfo, UploadResponse
from app.config import PDF_DIR, CHUNKS_FILE
from app.retrieve import reload_index

router = APIRouter()


def count_chunks_per_file() -> dict:
    """Reads chunks.jsonl and counts the chunks of each PDF, for example {"nist-ai-rmf-1.0.pdf": 412}."""
    file_counts = {}
    with open(CHUNKS_FILE, encoding="utf-8") as chunks_file:
        for line in chunks_file:
            file_name = json.loads(line)["file"]
            
            if file_name not in file_counts:
                file_counts[file_name] = 0          # If not already exists in file_counts dict, initialize the count for this PDF to 0
            
            file_counts[file_name] = file_counts[file_name] + 1     # Increase the count for this PDF by 1 for each chunk found
    return file_counts


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    counts = count_chunks_per_file()
    documents = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        documents.append(DocumentInfo(file=pdf_path.name, chunks=counts.get(pdf_path.name, 0)))
    return documents


@router.post("/documents", response_model=UploadResponse)
def upload_document(file: UploadFile):
    # Never trust the path from the user: keep only the file name.
    file_name = Path(file.filename or "").name
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    content = file.file.read()  # the 20 MB limit is checked in the web page (phase 5), before the upload
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file is not a real PDF.")

    pdf_path = PDF_DIR / file_name
    pdf_path.write_bytes(content)

    try:
        reload_index()  # first version: re-index every document (slow, uses HuggingFace credit)
    except Exception:
        pdf_path.unlink()  # do not keep a PDF that could not be indexed
        raise HTTPException(
            status_code=500,
            detail="Indexing failed and the PDF was removed. Run `uv run python -m app.ingest`, then restart the server.",
        )

    chunks = count_chunks_per_file().get(file_name, 0)
    return UploadResponse(file=file_name, chunks=chunks,
                          message="Saved. All documents were re-indexed.")


@router.delete("/documents/{name}")
def delete_document(name: str):
    file_name = Path(name).name
    pdf_path = PDF_DIR / file_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="No such document.")
    if len(list(PDF_DIR.glob("*.pdf"))) == 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last document.")

    pdf_path.unlink()
    reload_index()  # re-index the remaining documents
    return {"deleted": file_name, "message": "Deleted. All remaining documents were re-indexed."}
