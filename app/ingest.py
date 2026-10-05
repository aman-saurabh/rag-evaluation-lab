import json
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import PDF_DIR, CHUNKS_FILE, QDRANT_DIR, COLLECTION
from app.embeddings import embeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def make_chunks():
    """PDFs -> pages -> chunks, as LangChain Documents with id, file and page in the metadata."""
    chunks = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf)).load()  # one Document per page, metadata["page"] is 0-based
        pdf_chunks = splitter.split_documents(pages)
        for chunk in pdf_chunks:
            chunk.metadata = {"id": len(chunks), "file": pdf.name, "page": chunk.metadata["page"] + 1}
            chunks.append(chunk)
        print(f"{pdf.name}: {len(pdf_chunks)} chunks")
    return chunks


def ingest_all():
    chunks = make_chunks()
    with open(CHUNKS_FILE, "w", encoding="utf-8") as chunks_file:
        for chunk in chunks:
            line = json.dumps({**chunk.metadata, "text": chunk.page_content}, ensure_ascii=False)
            chunks_file.write(line + "\n")
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_FILE}")

    # Qdrant cannot delete its own open file on Windows, so wipe the folder first
    shutil.rmtree(QDRANT_DIR, ignore_errors=True)
    # One call: creates the collection, embeds the chunks in batches and uploads them
    store = QdrantVectorStore.from_documents(
        chunks, embeddings, path=str(QDRANT_DIR), collection_name=COLLECTION
    )
    print("Qdrant count:", store.client.count(COLLECTION).count)
    store.client.close()


if __name__ == "__main__":
    ingest_all()
