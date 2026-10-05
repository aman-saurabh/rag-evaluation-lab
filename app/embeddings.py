import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.config import EMBED_MODEL

# Calls the HuggingFace API (nothing runs on this computer).
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL,
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)
