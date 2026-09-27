"""Step 1 of RAG: turn the knowledge base into searchable chunks.

    python -m src.ingest

1. Load every markdown article in data/kb/
2. Split each article into chunks, one per "## " section, so each chunk covers one topic
3. Embed each chunk with a sentence-transformer model
4. Store vectors in ChromaDB (for semantic search) and the raw chunks in JSON (for BM25)
"""
import json
import re
import shutil
from pathlib import Path

from . import config


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_article(path: Path) -> list[dict]:
    """Split one markdown article into section-level chunks.

    The article title is prepended to every chunk. Without it, a section like
    "If self-service does not work" loses the context that it is about passwords.
    """
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem

    chunks = []
    # Split on level-2 headings, keeping the heading with its section
    for part in re.split(r"^(?=## )", text, flags=re.M):
        part = part.strip()
        if not part.startswith("## "):
            continue  # skip the title / intro block before the first section
        heading, _, body = part.partition("\n")
        heading = heading.removeprefix("## ").strip()
        body = body.strip()
        chunks.append({
            "id": f"{path.stem}#{slugify(heading)}",
            "doc": path.stem,
            "title": title,
            "section": heading,
            "text": f"{title} - {heading}\n{body}",
        })
    return chunks


def load_chunks(kb_dir: Path = config.KB_DIR) -> list[dict]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_article(path))
    return chunks


def build_index(chunks: list[dict], embed_fn) -> None:
    import chromadb

    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)  # rebuild from scratch so deleted articles disappear
    config.INDEX_DIR.mkdir(parents=True)

    client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
    # We pass our own embeddings, so no embedding_function is needed.
    # Cosine distance suits normalised sentence embeddings.
    col = client.create_collection(config.COLLECTION, embedding_function=None,
                                   metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed_fn([c["text"] for c in chunks]),
        metadatas=[{"doc": c["doc"], "title": c["title"], "section": c["section"]} for c in chunks],
    )
    config.CHUNKS_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def main():
    from .retriever import load_embedder

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(list(config.KB_DIR.glob('*.md')))} articles")
    print(f"Embedding with {config.EMBED_MODEL} (first run downloads the model)...")
    build_index(chunks, load_embedder())
    print(f"Index saved to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
