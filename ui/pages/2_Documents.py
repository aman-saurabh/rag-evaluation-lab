import httpx
import streamlit as st
import client

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB. The backend does not check the size, so we check it here.

st.title("Documents")
st.write(
    "Adding or deleting a document re-indexes **all** documents. "
    "That re-embeds everything, takes minutes and uses HuggingFace credit."
)

# ---------- Upload ----------
st.subheader("Add a document")
uploaded_file = st.file_uploader("PDF", type="pdf")

if uploaded_file and st.button("Add to index"):
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        st.error("This file is larger than 20 MB.")
    else:
        with st.spinner("Indexing. This can take a few minutes..."):
            try:
                result = client.upload_document(uploaded_file.name, uploaded_file.getvalue())
                st.success(f"{result['message']} {result['file']} has {result['chunks']} chunks.")
            except httpx.ConnectError:
                st.error(client.BACKEND_DOWN_MESSAGE)
            except httpx.HTTPStatusError as error:
                st.error(f"The backend returned an error: {error.response.text}")

# ---------- List and delete ----------
st.subheader("Indexed documents")

try:
    documents = client.list_documents()
except httpx.ConnectError:
    st.error(client.BACKEND_DOWN_MESSAGE)
    st.stop()

is_sure = st.checkbox("I am sure I want to delete a document")

for document in documents:
    name_column, chunks_column, button_column = st.columns([3, 1, 1])
    name_column.write(document["file"])
    chunks_column.write(f"{document['chunks']} chunks")

    if button_column.button("Delete", key=f"delete_{document['file']}", disabled=not is_sure):
        with st.spinner("Deleting and re-indexing..."):
            try:
                client.delete_document(document["file"])
            except httpx.HTTPStatusError as error:
                st.error(f"The backend returned an error: {error.response.text}")
                st.stop()
        st.rerun()  # reload the page so the list is up to date
