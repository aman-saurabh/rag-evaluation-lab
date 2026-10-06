from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
CHUNKS_FILE = DATA / "chunks.jsonl"
QDRANT_DIR = DATA / "qdrant"
SETTINGS_FILE = DATA / "settings.json"
DATASET_DIR = DATA / "datasets"

COLLECTION = "chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_URL ="https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
EMBED_DIM = 384
FAST_MODEL = "openai/gpt-oss-20b"
STRONG_MODEL = "openai/gpt-oss-120b"

# Safety models on Groq (used by the guards in phase 6)
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"   # catches trick questions ("prompt injection")
SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"             # checks a text against safety rules we give it
