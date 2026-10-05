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
EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
EMBED_DIM = 384
FAST_MODEL = "openai/gpt-oss-20b"
STRONG_MODEL = "openai/gpt-oss-120b"
