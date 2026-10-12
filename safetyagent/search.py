"""Recall-summary index: Sentence Transformers embeddings in a FAISS inner-product index with vehicle filters."""

from __future__ import annotations

import hashlib
from pathlib import Path

import faiss
import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE v1.5 retrieval instruction
MAX_SEQ_LENGTH = 256


def recall_text(components: str, defect: str, consequence: str) -> str:
    """Text embedded for one campaign: its component names, defect and consequence summaries."""
    return f"{components}. {(defect or '').strip()} {(consequence or '').strip()}"


def texts_key(ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha1()
    for i, t in zip(ids, texts):
        h.update(i.encode())
        h.update(t.encode())
    return h.hexdigest()[:12]


class Encoder:
    """Sentence Transformers bi-encoder (fp32), loaded lazily."""

    def __init__(self, model_id: str = EMBED_MODEL, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id, device=self.device)
        self.model.max_seq_length = MAX_SEQ_LENGTH

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        return emb.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([QUERY_PREFIX + text])[0]


def encode_cached(encoder: Encoder, ids: list[str], texts: list[str], cache_dir: Path) -> np.ndarray:
    """Encode recall texts, or load the cached matrix for exactly these ids and texts."""
    name = f"emb__{encoder.model_id.replace('/', '--')}__{MAX_SEQ_LENGTH}__fp32__{texts_key(ids, texts)}.npy"
    path = cache_dir / name
    if path.exists():
        return np.load(path)
    emb = encoder.encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


class RecallIndex:
    """Exact cosine search (``IndexFlatIP`` on normalised vectors); a filter restricts the search to given campaigns."""

    def __init__(self, campaigns: list[str], embeddings: np.ndarray, query_encoder) -> None:
        self.campaigns = list(campaigns)
        self.pos = {c: i for i, c in enumerate(self.campaigns)}
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.query_encoder = query_encoder

    def search(self, query: str, k: int = 5, allowed: list[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (campaign, cosine) pairs; ``allowed`` (possibly empty) limits the candidates."""
        q = np.asarray(self.query_encoder(query), dtype=np.float32).reshape(1, -1)
        return self.search_vector(q, k, allowed)

    def search_vector(self, q: np.ndarray, k: int, allowed: list[str] | None = None) -> list[tuple[str, float]]:
        if allowed is not None:
            ids = np.array([self.pos[c] for c in allowed if c in self.pos], dtype=np.int64)
            if len(ids) == 0:
                return []
            k = min(k, len(ids))
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
            scores, idx = self.index.search(q, k, params=params)
        else:
            scores, idx = self.index.search(q, min(k, self.index.ntotal))
        return [(self.campaigns[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
