"""
phase4_eval.py
--------------
Step 1 (2a): Print top-10 hybrid candidates at alpha=0.5 for each query
             so the user can identify ground-truth relevant product IDs.

Step 2 (2b-2d): Once ground_truth dict is populated, compute Precision@5
                across alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and print a
                formatted results table with the best alpha highlighted.

Usage
-----
  python phase4_eval.py          # runs Step 1 (candidates only)
  python phase4_eval.py --eval   # runs Step 2 (full precision table)
"""

import sys
import os
from phase4_hybrid_search import hybrid_search, print_results

# ═══════════════════════════════════════════════════════════
# ▶  STEP 1 – candidate queries (fill ground truth below after seeing output)
# ═══════════════════════════════════════════════════════════
QUERIES = [
    "running shoes",
    "waterproof jacket",
    "wireless bluetooth headphones",
    "kids toys under 500",
    "formal shirt for men",
]

# ═══════════════════════════════════════════════════════════
# ▶  STEP 2 – Ground truth: fill in after running Step 1
#    Format: query_string -> set of product IDs you consider relevant
#    Example: "running shoes" -> {"prod_251", "prod_786", "prod_254"}
# ═══════════════════════════════════════════════════════════
GROUND_TRUTH: dict[str, set[str]] = {
    "running shoes":                  set({"prod_7290","prod_786","prod_409","prod_1936","prod_11122"}),   # ← fill after Step 1
    "waterproof jacket":              set(),   # ← fill after Step 1
    "wireless bluetooth headphones":  set({"prod_5464","prod_6361","prod_3735","prod_5525"}),   # ← fill after Step 1
    "kids toys under 500":            set({"prod_10684","prod_4523","prod_8062"}),   # ← fill after Step 1
    "formal shirt for men":           set({"prod_10775","prod_6827","prod_10726","prod_6824","prod_10744"}),   # ← fill after Step 1
}

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K = 5
CANDIDATE_TOP_K = 10


# ─────────────────────────────────────────────
# Precision@K helper
# ─────────────────────────────────────────────
def precision_at_k(results: list[dict], relevant: set[str], k: int = 5) -> float:
    """Fraction of top-k results that are relevant."""
    if not relevant:
        return 0.0
    top_ids = {r["id"] for r in results[:k]}
    hits = len(top_ids & relevant)
    return hits / k


# ─────────────────────────────────────────────
# Step 1: Print top-10 candidates at alpha=0.5
# ─────────────────────────────────────────────
def run_candidates():
    print("=" * 70)
    print("PHASE 4 EVAL – STEP 1: Top-10 Candidates at alpha=0.5")
    print("Review the results and note which product IDs are truly relevant.")
    print("Then fill in GROUND_TRUTH in this script and re-run with --eval")
    print("=" * 70)

    for query in QUERIES:
        results = hybrid_search(query, alpha=0.5, top_k=CANDIDATE_TOP_K)

        print(f"\n{'-'*70}")
        print(f"QUERY: \"{query}\"  (top {CANDIDATE_TOP_K})")
        print(f"{'-'*70}")
        for i, r in enumerate(results, 1):
            price_str = f"Rs. {r['price']:.2f}" if r["price_known"] else "N/A"
            relevant_flag = ""  # will fill manually
            print(f"  {i:>2}. [{r['score']:.4f}] {r['title']}")
            print(f"       Brand: {r['brand'] or '—'} | Price: {price_str}")
            print(f"       ID: {r['id']}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("Copy the IDs you consider relevant into GROUND_TRUTH above,")
    print("then run:  python phase4_eval.py --eval")
    print("=" * 70)


# ─────────────────────────────────────────────
# Step 2: Precision@5 table across alpha values
# ─────────────────────────────────────────────
def run_evaluation():
    # Validate ground truth is populated
    empty_queries = [q for q, ids in GROUND_TRUTH.items() if not ids]
    if empty_queries:
        print("[WARNING] The following queries have empty ground truth sets:")
        for q in empty_queries:
            print(f"  - \"{q}\"")
        print("Precision will be 0.0 for those queries.\n")

    print("=" * 70)
    print("PHASE 4 EVAL – STEP 2: Precision@5 across alpha values")
    print("=" * 70)

    # Collect results: {query: {alpha: [results]}}
    scores: dict[str, dict[float, float]] = {}

    for query in QUERIES:
        scores[query] = {}
        relevant = GROUND_TRUTH[query]
        for alpha in ALPHA_VALUES:
            results = hybrid_search(query, alpha=alpha, top_k=TOP_K)
            p_at_5 = precision_at_k(results, relevant, k=TOP_K)
            scores[query][alpha] = p_at_5

    # ── Print table ──────────────────────────────────────────────────────
    col_w = 10
    q_w   = 36

    header_alphas = "".join(f"  α={a:.2f}  " for a in ALPHA_VALUES)
    print(f"\n{'Query':<{q_w}}{header_alphas}")
    print("-" * (q_w + col_w * len(ALPHA_VALUES) + 2))

    avg_by_alpha: dict[float, float] = {a: 0.0 for a in ALPHA_VALUES}

    for query in QUERIES:
        row = f"{query[:q_w-1]:<{q_w}}"
        for alpha in ALPHA_VALUES:
            p = scores[query][alpha]
            avg_by_alpha[alpha] += p
            cell = f"{p:.2f}"
            row += f"{cell:^{col_w}}"
        print(row)

    # Average row
    n = len(QUERIES)
    print("-" * (q_w + col_w * len(ALPHA_VALUES) + 2))
    avg_row = f"{'AVERAGE':<{q_w}}"
    best_alpha = max(avg_by_alpha, key=avg_by_alpha.get)
    for alpha in ALPHA_VALUES:
        avg = avg_by_alpha[alpha] / n
        cell = f"{avg:.2f}"
        # Mark best alpha with *
        if alpha == best_alpha:
            cell = f"{avg:.2f}*"
        avg_row += f"{cell:^{col_w}}"
    print(avg_row)

    print("\n" + "=" * 70)
    best_avg = avg_by_alpha[best_alpha] / n
    print(f"  ✅ Best alpha = {best_alpha:.2f}  (avg Precision@5 = {best_avg:.4f})")
    print(f"     α=1.0  → pure dense (semantic)  | avg P@5 = {avg_by_alpha[1.0]/n:.4f}")
    print(f"     α=0.0  → pure sparse (BM25)      | avg P@5 = {avg_by_alpha[0.0]/n:.4f}")
    print(f"     α=0.5  → balanced hybrid         | avg P@5 = {avg_by_alpha[0.5]/n:.4f}")
    print("=" * 70)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if "--eval" in sys.argv:
        run_evaluation()
    else:
        run_candidates()
