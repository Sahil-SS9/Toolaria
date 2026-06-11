"""Ranking for the `search` fetch mode.

Two scorers, picked at runtime:

  - lexical (BM25): always available, no dependencies, deterministic.
  - embedding (cosine over sentence-transformers vectors): used when the
    optional `sentence-transformers` package is installed; numpy is used for
    the maths when present, with a pure-Python cosine fallback otherwise.

Both rank the same line-aligned chunks, so every hit maps back to a
`rescuer_fetch(mode="range", ...)` call.
"""
from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

try:
    import numpy as _np
    _HAVE_NUMPY = True
except ImportError:
    _np = None
    _HAVE_NUMPY = False

# Embedding model cache, keyed by name so a config change takes effect.
_model = None
_model_name: str | None = None
_available: bool | None = None


def embeddings_available() -> bool:
    # Resolved once: the import is retried on every search otherwise, which
    # is a per-call filesystem scan on the common dep-free path.
    global _available
    if _available is None:
        try:
            import sentence_transformers  # noqa: F401
            _available = True
        except Exception:
            _available = False
    return _available


def _get_model(model_name: str):
    global _model, _model_name
    if _model is not None and _model_name in (None, model_name):
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        # trust_remote_code stays off: a model name is operator config, but
        # never let it pull executable code.
        _model = SentenceTransformer(model_name, trust_remote_code=False)
        _model_name = model_name
        return _model
    except Exception as exc:
        logger.info("toolaria: embedding model unavailable (%s); using lexical", exc)
        return None


def embed(texts: list[str], model_name: str) -> list[list[float]] | None:
    """Embed *texts*, or None if no model is available."""
    model = _get_model(model_name)
    if model is None:
        return None
    try:
        vecs = model.encode(texts)
        return [list(map(float, v)) for v in vecs]
    except Exception as exc:
        logger.warning("toolaria: embedding failed: %s", exc)
        return None


# ── scoring ─────────────────────────────────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    if _HAVE_NUMPY:
        va, vb = _np.asarray(a), _np.asarray(b)
        na, nb = _np.linalg.norm(va), _np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(chunk_texts: list[str], query: str,
                k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Standard BM25 score per chunk for *query*. Deterministic, dep-free."""
    docs = [_tokenize(t) for t in chunk_texts]
    n = len(docs)
    if n == 0:
        return []
    lengths = [len(d) for d in docs]
    avg_len = sum(lengths) / n if n else 0.0

    df: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1

    q_terms = set(_tokenize(query))
    idf = {
        t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for t in q_terms if t in df
    }

    scores = []
    for d, length in zip(docs, lengths):
        if not d:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for term in d:
            if term in idf:
                tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            denom = freq + k1 * (1 - b + b * length / avg_len) if avg_len else freq
            score += idf[term] * (freq * (k1 + 1)) / denom if denom else 0.0
        scores.append(score)
    return scores


def rank(chunk_texts: list[str], chunk_vectors: list[list[float]] | None,
         query: str, model_name: str, top_k: int) -> tuple[str, list[tuple[int, float]]]:
    """Return (method, [(chunk_index, score), ...]) for the top_k chunks.

    Uses embeddings when *chunk_vectors* is provided, else BM25.
    """
    if chunk_vectors is not None:
        qv = embed([query], model_name)
        if qv:
            scores = [cosine(qv[0], v) for v in chunk_vectors]
            method = "semantic"
        else:
            scores = bm25_scores(chunk_texts, query)
            method = "lexical"
    else:
        scores = bm25_scores(chunk_texts, query)
        method = "lexical"

    ranked = sorted(enumerate(scores), key=lambda kv: -kv[1])
    ranked = [(i, s) for i, s in ranked if s > 0][:top_k]
    return method, ranked
