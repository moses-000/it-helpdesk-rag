"""Step 2 of RAG: find the chunks most relevant to a question.

Three retrieval methods, so you can compare them in the evaluation:
  - bm25:   classic keyword search. Great for exact terms ("BitLocker", "4400").
  - dense:  semantic search with embeddings. Handles paraphrases
            ("my laptop crawls" -> "Slow Laptop Performance").
  - hybrid: combines both rankings with Reciprocal Rank Fusion (RRF).
"""
import json
import re
from typing import Callable

from rank_bm25 import BM25Okapi

from . import config

EmbedFn = Callable[[list[str]], list[list[float]]]


def load_embedder(model_name: str = config.EMBED_MODEL) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, embed_fn: EmbedFn | None = None):
        import chromadb

        if not config.CHUNKS_FILE.exists():
            raise FileNotFoundError("No index found. Run `python -m src.ingest` first.")
        self.chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embed = embed_fn or load_embedder()
        client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
        self.collection = client.get_collection(config.COLLECTION, embedding_function=None)

    def search_bm25(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked]

    def search_dense(self, query: str, k: int) -> list[dict]:
        res = self.collection.query(query_embeddings=self.embed([query]), n_results=k)
        return [{**self.by_id[cid], "score": 1 - dist}  # cosine similarity
                for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def search_hybrid(self, query: str, k: int, rrf_k: int = 60, pool: int = 10) -> list[dict]:
        """Reciprocal Rank Fusion: score = sum over methods of 1 / (rrf_k + rank).

        RRF uses ranks rather than raw scores, so BM25 scores (unbounded) and
        cosine similarities (0 to 1) can be combined without any tuning.
        """
        fused: dict[str, float] = {}
        for results in (self.search_bm25(query, pool), self.search_dense(query, pool)):
            for rank, chunk in enumerate(results, start=1):
                fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1 / (rrf_k + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.by_id[cid], "score": fused[cid]} for cid in best]

    def search(self, query: str, k: int = config.TOP_K, method: str = "hybrid") -> list[dict]:
        return {"bm25": self.search_bm25, "dense": self.search_dense,
                "hybrid": self.search_hybrid}[method](query, k)
