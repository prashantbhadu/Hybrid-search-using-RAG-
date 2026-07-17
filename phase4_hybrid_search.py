"""
phase4_hybrid_search.py
-----------------------
Core hybrid search module combining dense (semantic) and sparse (BM25)
vectors using a convex combination controlled by alpha.

  alpha = 1.0  →  pure dense (semantic only)
  alpha = 0.0  →  pure sparse (BM25 / keyword only)
  alpha = 0.5  →  balanced hybrid
"""

import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# Load environment variables from .env file
load_dotenv()


# ─────────────────────────────────────────────
# Module-level singletons (lazy-loaded once)
# ─────────────────────────────────────────────
_bm25: BM25Encoder | None = None
_model: SentenceTransformer | None = None
_index = None


def _load_resources(encoder_path: str = "bm25_encoder.json",
                    model_name: str = "all-MiniLM-L6-v2",
                    index_name: str = "ecommerce-search") -> None:
    """Initialise BM25 encoder, dense model, and Pinecone index (once)."""
    global _bm25, _model, _index

    if _bm25 is None:
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"BM25 encoder not found at '{encoder_path}'. "
                "Run phase3_upsert_sparse.py first."
            )
        print(f"Loading BM25 encoder from '{encoder_path}'...")
        _bm25 = BM25Encoder()
        _bm25.load(encoder_path)
        print("BM25 encoder loaded.")

    if _model is None:
        print(f"Loading SentenceTransformer model '{model_name}'...")
        _model = SentenceTransformer(model_name)
        print("Dense model loaded.")

    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY environment variable is not set."
            )
        print(f"Connecting to Pinecone index '{index_name}'...")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)
        print("Connected to Pinecone index.\n")


# ─────────────────────────────────────────────
# Core helper: convex combination scaling
# ─────────────────────────────────────────────
def hybrid_scale(dense_vec: list[float],
                 sparse_vec: dict,
                 alpha: float) -> tuple[list[float], dict]:
    """
    Scale dense and sparse vectors by alpha for hybrid search.

    Parameters
    ----------
    dense_vec   : list of floats  – raw dense embedding (length 384)
    sparse_vec  : dict            – {'indices': [...], 'values': [...]}
    alpha       : float in [0, 1] – 1.0 = pure dense, 0.0 = pure sparse

    Returns
    -------
    scaled_dense  : list[float]
    scaled_sparse : dict with same indices, values scaled by (1 - alpha)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values":  [v * (1.0 - alpha) for v in sparse_vec["values"]],
    }
    return scaled_dense, scaled_sparse


# ─────────────────────────────────────────────
# Main search function
# ─────────────────────────────────────────────
def hybrid_search(query_text: str,
                  alpha: float = 0.5,
                  top_k: int = 5,
                  price_max: float | None = None,
                  price_min: float | None = None) -> list[dict]:
    """
    Run a hybrid search query.

    Parameters
    ----------
    query_text : str   – natural language query
    alpha      : float – blend weight (1.0 = dense only, 0.0 = sparse only)
    top_k      : int   – number of results to return
    price_max  : float – optional upper price filter (inclusive)
    price_min  : float – optional lower price filter (inclusive)

    Returns
    -------
    list of dicts with keys: id, score, title, brand, price, price_known
    """
    _load_resources()

    # 1. Dense embedding
    dense_vec = _model.encode(query_text, normalize_embeddings=True).tolist()

    # 2. Sparse BM25 encoding
    raw_sparse = _bm25.encode_queries(query_text)
    sparse_vec = {
        "indices": [int(i) for i in raw_sparse["indices"]],
        "values":  [float(v) for v in raw_sparse["values"]],
    }

    # 3. Convex combination scaling
    scaled_dense, scaled_sparse = hybrid_scale(dense_vec, sparse_vec, alpha)

    # 4. Optional metadata price filter
    query_filter = None
    if price_max is not None or price_min is not None:
        query_filter = {}
        if price_max is not None:
            query_filter["price"] = query_filter.get("price", {})
            query_filter["price"]["$lte"] = float(price_max)
        if price_min is not None:
            query_filter["price"] = query_filter.get("price", {})
            query_filter["price"]["$gte"] = float(price_min)

    # 5. Query Pinecone
    kwargs = dict(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    if query_filter:
        kwargs["filter"] = query_filter

    response = _index.query(**kwargs)

    # 6. Parse results
    results = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        price_known = bool(meta.get("price_known", False))
        price = float(meta.get("price", 0.0))
        results.append({
            "id":          match["id"],
            "score":       round(float(match.get("score", 0.0)), 4),
            "title":       meta.get("title", "N/A"),
            "brand":       meta.get("brand", ""),
            "price":       price,
            "price_known": price_known,
        })

    return results


# ─────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────
def print_results(query: str, results: list[dict], alpha: float) -> None:
    print(f"\nHYBRID QUERY (alpha={alpha:.2f}): '{query}'")
    print("-" * 60)
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        price_str = f"Rs. {r['price']:.2f}" if r["price_known"] else "N/A"
        print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}")
        print(f"       Brand: {r['brand'] or '—'} | Price: {price_str} | ID: {r['id']}")


# ─────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        ("running shoes",               0.5),
        ("waterproof jacket",           0.5),
        ("wireless bluetooth headphones", 0.5),
        ("kids toys under 500",         0.5),
        ("formal shirt for men",        0.5),
    ]
    print("=" * 60)
    print("PHASE 4: HYBRID SEARCH SMOKE TEST (alpha=0.5)")
    print("=" * 60)
    for q, a in queries:
        results = hybrid_search(q, alpha=a, top_k=5)
        print_results(q, results, a)
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
