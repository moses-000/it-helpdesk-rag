"""Central settings. Everything can be overridden in a .env file (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_DIR = ROOT / "data" / "kb"            # markdown help articles
INDEX_DIR = ROOT / ".index"              # vector DB + chunk store (created by ingest.py)
CHUNKS_FILE = INDEX_DIR / "chunks.json"
COLLECTION = "helpdesk_kb"

# Embedding model (runs locally, free). all-MiniLM-L6-v2 is small (~90 MB) and fast on CPU.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM: any OpenAI-compatible API. Defaults to Groq (free tier available).
# For a fully local setup use Ollama: LLM_BASE_URL=http://localhost:11434/v1, LLM_MODEL=llama3.2
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

TOP_K = int(os.getenv("TOP_K", "4"))
