"""Tests for Toolaria v2: BM25, cosine, and search ranking (no model loads)."""
from semantic import bm25_scores, cosine, rank


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        "the cat sat on the mat",
        "database connection pooling and retries",
        "weather today is sunny and warm",
    ]
    scores = bm25_scores(chunks, "database connection retry")
    assert scores[1] == max(scores)
    assert scores[1] > 0


def test_bm25_empty_inputs():
    assert bm25_scores([], "x") == []
    assert bm25_scores(["only doc"], "nomatch") == [0.0]


def test_cosine_identical_and_orthogonal():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_rank_uses_vectors_when_present():
    texts = ["apple stuff", "banana stuff", "cherry stuff"]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # query vector built by the stub embed below
    import semantic
    orig = semantic.embed
    semantic.embed = lambda texts, model: [[0.0, 1.0, 0.0]]  # 'banana'
    try:
        method, ranked = rank(texts, vectors, "banana", "stub", 2)
    finally:
        semantic.embed = orig
    assert method == "semantic"
    assert ranked[0][0] == 1  # banana chunk ranked first


def test_rank_falls_back_to_lexical_without_vectors():
    texts = ["alpha terms here", "beta terms here"]
    method, ranked = rank(texts, None, "beta", "stub", 2)
    assert method == "lexical"
    assert ranked[0][0] == 1
