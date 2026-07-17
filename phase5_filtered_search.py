"""
phase5_filtered_search.py
-------------------------
Phase 5: Metadata filtering combined with hybrid search.
Reuses the hybrid search logic from phase4, extends it to support price (min/max),
brand (exact match), and category (exact match) filters.

Includes side-by-side comparison testing and evaluation re-runs.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import phase4_hybrid_search as p4


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extended Hybrid Search with Metadata Filters
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_search_filtered(
    query_text: str,
    alpha: float = 0.5,
    top_k: int = 5,
    max_price: float | None = None,
    min_price: float | None = None,
    category: str | None = None,
    brand: str | None = None
) -> list[dict]:
    """
    Run a hybrid search query with metadata filtering.

    Parameters
    ----------
    query_text : str   - Natural language query
    alpha      : float - Blend weight (1.0 = dense only, 0.0 = sparse only)
    top_k      : int   - Number of results to return
    max_price  : float - Optional maximum price
    min_price  : float - Optional minimum price
    category   - Optional category string (exact match)
    brand      - Optional brand string (exact match)

    Returns
    -------
    list of dicts with parsed metadata fields.
    """
    # Lazily load models and index using phase4 loader
    p4._load_resources()

    # 1. Generate dense embedding
    dense_vec = p4._model.encode(query_text, normalize_embeddings=True).tolist()

    # 2. Generate sparse BM25 encoding
    raw_sparse = p4._bm25.encode_queries(query_text)
    sparse_vec = {
        "indices": [int(i) for i in raw_sparse["indices"]],
        "values":  [float(v) for v in raw_sparse["values"]],
    }

    # 3. Scale dense & sparse vectors
    scaled_dense, scaled_sparse = p4.hybrid_scale(dense_vec, sparse_vec, alpha)

    # 4. Construct Pinecone query filter dictionary
    query_filter = {}

    # Price constraints
    if max_price is not None or min_price is not None:
        # Since missing/unknown prices are stored as 0 with price_known=False,
        # we enforce price_known=True to prevent 0-price products from incorrectly
        # matching cheap price bounds.
        query_filter["price_known"] = {"$eq": True}
        
        price_cond = {}
        if max_price is not None:
            price_cond["$lte"] = float(max_price)
        if min_price is not None:
            price_cond["$gte"] = float(min_price)
        query_filter["price"] = price_cond

    # Brand constraint (exact match)
    if brand is not None:
        query_filter["brand"] = {"$eq": brand}

    # Category constraint (exact match)
    if category is not None:
        query_filter["category"] = {"$eq": category}

    # 5. Query Pinecone index
    kwargs = dict(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    if query_filter:
        kwargs["filter"] = query_filter

    response = p4._index.query(**kwargs)

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
            "category":    meta.get("category", "")
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Find brands containing substring in products_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
def find_matching_brands_in_dataset(substring: str) -> list[str]:
    csv_path = "data/products_clean.csv"
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        if "brand" in df.columns:
            matching_brands = df[df["brand"].astype(str).str.contains(substring, case=False, na=False)]["brand"].unique()
            return sorted(list(matching_brands))
    except Exception as e:
        print(f"[Warning] Error reading local dataset for brand suggestion: {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Printing Side-by-Side Results
# ─────────────────────────────────────────────────────────────────────────────
def print_side_by_side(query: str, results_no_filter: list[dict], results_filtered: list[dict], filter_desc: str):
    print("=" * 115)
    print(f" QUERY: \"{query}\"")
    print(f" Filter: {filter_desc}")
    print("=" * 115)
    
    col_w = 54
    header_no_filter = "WITHOUT FILTER"
    header_filtered = "WITH FILTER"
    print(f"  {header_no_filter:<{col_w}} |  {header_filtered:<{col_w}}")
    print("-" * (col_w * 2 + 6))
    
    max_len = max(len(results_no_filter), len(results_filtered))
    for i in range(max_len):
        # Format left side (no filter)
        if i < len(results_no_filter):
            r = results_no_filter[i]
            p_str = f"Rs. {r['price']:.2f}" if r["price_known"] else "N/A"
            brand_trunc = r["brand"][:10]
            brand_str = f" [{brand_trunc}]" if r['brand'] else ""
            title_trunc = r["title"][:22]
            left_str = f"{i+1}. [{r['score']:.4f}] {title_trunc}{brand_str} ({p_str})"
            left_str = f"  {left_str:<{col_w}}"
        else:
            left_str = " " * (col_w + 2)
            
        # Format right side (filtered)
        if i < len(results_filtered):
            r = results_filtered[i]
            p_str = f"Rs. {r['price']:.2f}" if r["price_known"] else "N/A"
            brand_trunc = r["brand"][:10]
            brand_str = f" [{brand_trunc}]" if r['brand'] else ""
            title_trunc = r["title"][:22]
            right_str = f"{i+1}. [{r['score']:.4f}] {title_trunc}{brand_str} ({p_str})"
            right_str = f" {right_str:<{col_w}}"
        else:
            right_str = " " * col_w
            
        print(f"{left_str} | {right_str}")
    print("=" * 115 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Running the 3 Comparisons
# ─────────────────────────────────────────────────────────────────────────────
def run_comparisons():
    print("=" * 80)
    print("PHASE 5: RUNNING 3 COMPARISON TESTS (alpha = 0.5)")
    print("=" * 80)

    # Comparison 1: Query: "kids toys", max_price=500
    q1 = "kids toys"
    res1_unfiltered = hybrid_search_filtered(q1, alpha=0.5, top_k=5)
    res1_filtered = hybrid_search_filtered(q1, alpha=0.5, top_k=5, max_price=500)
    print_side_by_side(q1, res1_unfiltered, res1_filtered, "max_price=500")

    # Comparison 2: Query: "formal shirt", max_price=1000, brand contains "Slim"
    # Note: Pinecone does not support substring/partial match filters on metadata.
    # Therefore, we filter by brand exact match = "Slim" and explain the results.
    q2 = "formal shirt"
    print("Note: Pinecone metadata filtering only supports exact matches. We will pass brand='Slim'.")
    matching_brands = find_matching_brands_in_dataset("Slim")
    print(f"Local CSV search shows matching brands in dataset containing 'Slim': {matching_brands}")
    
    res2_unfiltered = hybrid_search_filtered(q2, alpha=0.5, top_k=5)
    res2_filtered = hybrid_search_filtered(q2, alpha=0.5, top_k=5, max_price=1000, brand="Slim")
    print_side_by_side(q2, res2_unfiltered, res2_filtered, "max_price=1000, brand='Slim' (Exact Match)")

    # Comparison 3: Query: "running shoes", min_price=1000, max_price=3000
    q3 = "running shoes"
    res3_unfiltered = hybrid_search_filtered(q3, alpha=0.5, top_k=5)
    res3_filtered = hybrid_search_filtered(q3, alpha=0.5, top_k=5, min_price=1000, max_price=3000)
    print_side_by_side(q3, res3_unfiltered, res3_filtered, "min_price=1000, max_price=3000")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision@5 Evaluation (alpha = 0.75) Before vs After
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(results: list[dict], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    top_ids = {r["id"] for r in results[:k]}
    hits = len(top_ids & relevant)
    return hits / k


def run_evaluation():
    print("=" * 80)
    print("PHASE 5: RE-RUNNING PHASE 4 EVALUATION AT alpha = 0.75")
    print("We apply max_price=500 filter to the 'kids toys under 500' query.")
    print("=" * 80)

    # Use ground truth definitions from phase4_eval.py
    GROUND_TRUTH = {
        "running shoes":                  {"prod_7290","prod_786","prod_409","prod_1936","prod_11122"},
        "waterproof jacket":              set(),
        "wireless bluetooth headphones":  {"prod_5464","prod_6361","prod_3735","prod_5525"},
        "kids toys under 500":            {"prod_10684","prod_4523","prod_8062"},
        "formal shirt for men":           {"prod_10775","prod_6827","prod_10726","prod_6824","prod_10744"},
    }

    queries = list(GROUND_TRUTH.keys())
    alpha = 0.75
    top_k = 5

    # 1. Baseline Evaluation (No filter on any query)
    baseline_scores = {}
    for q in queries:
        results = hybrid_search_filtered(q, alpha=alpha, top_k=top_k)
        baseline_scores[q] = precision_at_k(results, GROUND_TRUTH[q], k=top_k)

    # 2. Filtered Evaluation (max_price=500 on "kids toys under 500" only)
    filtered_scores = baseline_scores.copy()
    results_filtered = hybrid_search_filtered(
        "kids toys under 500",
        alpha=alpha,
        top_k=top_k,
        max_price=500
    )
    filtered_scores["kids toys under 500"] = precision_at_k(results_filtered, GROUND_TRUTH["kids toys under 500"], k=top_k)

    # Display evaluation comparison table
    q_w = 34
    col_w = 18
    print(f"\n{'Query':<{q_w}} | {'Baseline P@5':^{col_w}} | {'Filtered P@5':^{col_w}} | {'Change':^{col_w}}")
    print("-" * (q_w + col_w * 3 + 10))

    for q in queries:
        b_score = baseline_scores[q]
        f_score = filtered_scores[q]
        diff = f_score - b_score
        diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
        
        marker = " (Filtered)" if q == "kids toys under 500" else ""
        print(f"{q + marker:<{q_w}} | {b_score:^{col_w}.2f} | {f_score:^{col_w}.2f} | {diff_str:^{col_w}}")

    print("-" * (q_w + col_w * 3 + 10))
    avg_baseline = sum(baseline_scores.values()) / len(queries)
    avg_filtered = sum(filtered_scores.values()) / len(queries)
    avg_diff = avg_filtered - avg_baseline
    avg_diff_str = f"{avg_diff:+.4f}" if avg_diff != 0 else "0.0000"
    print(f"{'AVERAGE':<{q_w}} | {avg_baseline:^{col_w}.4f} | {avg_filtered:^{col_w}.4f} | {avg_diff_str:^{col_w}}")
    print("=" * 80 + "\n")

    # Detailed view for kids toys before/after
    results_kids_baseline = hybrid_search_filtered("kids toys under 500", alpha=alpha, top_k=top_k)
    print("Detailed result listing for query: 'kids toys under 500'")
    print_side_by_side("kids toys under 500", results_kids_baseline, results_filtered, "max_price=500")


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_comparisons()
    run_evaluation()
