"""Embedding backends.

We expose one small interface (`Embedder`) with three implementations so
the rest of the system is agnostic to where vectors come from:

  * SentenceTransformerEmbedder - semantic, local model (default if available)
  * OpenAIEmbedder              - semantic, hosted (needs OPENAI_API_KEY)
  * TfidfEmbedder               - lexical, dependency-free OFFLINE fallback

All embedders return L2-normalised float32 vectors, so an inner-product
FAISS index yields cosine similarity directly. `is_semantic` lets the
evidence gate pick the appropriate similarity threshold.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.config import get_settings


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class Embedder(ABC):
    name: str = "base"
    is_semantic: bool = False

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit any state needed (e.g. TF-IDF vocabulary). No-op for hosted models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class TfidfEmbedder(Embedder):
    """Offline lexical embedder. Deterministic, no network, no model download.

    Combines word n-grams (captures phrases) with character n-grams (captures
    acronyms like "HFpEF" and morphological variants such as symptom/symptoms).
    Not as powerful as a sentence transformer for paraphrase matching, but it
    makes the whole system runnable anywhere and keeps results reproducible.
    """

    name = "tfidf"
    is_semantic = False

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._word = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True,
        )
        self._char = TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
            min_df=1, sublinear_tf=True,
        )
        self._dim = 0

    def fit(self, corpus: List[str]) -> None:
        self._word.fit(corpus)
        self._char.fit(corpus)
        self._dim = (
            len(self._word.get_feature_names_out())
            + len(self._char.get_feature_names_out())
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        import scipy.sparse as sp

        combined = sp.hstack([self._word.transform(texts), self._char.transform(texts)])
        return _l2_normalize(combined.toarray().astype("float32"))

    @property
    def dim(self) -> int:
        return self._dim


class SentenceTransformerEmbedder(Embedder):
    """Local semantic embeddings via sentence-transformers (all-MiniLM-L6-v2)."""

    name = "sentence_transformers"
    is_semantic = True

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: List[str]) -> None:  # no fitting needed
        return None

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.astype("float32")

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedder(Embedder):
    """Hosted semantic embeddings via the OpenAI embeddings API."""

    name = "openai"
    is_semantic = True

    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dim = 1536  # text-embedding-3-small

    def fit(self, corpus: List[str]) -> None:
        return None

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        mat = np.array([d.embedding for d in resp.data], dtype="float32")
        return _l2_normalize(mat)

    @property
    def dim(self) -> int:
        return self._dim


def build_embedder() -> Embedder:
    """Resolve the configured backend, with graceful auto-fallback.

    Order for `auto`: OpenAI (if key) -> sentence-transformers (if usable)
    -> TF-IDF. Any failure (missing package, no network for model download)
    silently falls back so the service always starts.
    """
    settings = get_settings()
    choice = settings.embedding_backend.lower()

    def try_openai() -> Embedder | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            return OpenAIEmbedder(settings.openai_embedding_model)
        except Exception:
            return None

    def try_st() -> Embedder | None:
        try:
            return SentenceTransformerEmbedder(settings.st_model_name)
        except Exception:
            return None

    if choice == "openai":
        return try_openai() or TfidfEmbedder()
    if choice == "sentence_transformers":
        return try_st() or TfidfEmbedder()
    if choice == "tfidf":
        return TfidfEmbedder()

    # auto
    return try_openai() or try_st() or TfidfEmbedder()
