"""
app.py — Streamlit UI for Hybrid Product Search
"""

import os
import time
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Typography ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- Hide Streamlit defaults ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---------- Page header gradient ---------- */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.8rem;
}

/* ---------- Result card ---------- */
.result-card {
    background: linear-gradient(135deg, #1e1b4b08, #312e8112);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.10);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.45rem;
}
.card-meta {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 0.3rem;
}
.card-pill {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}
.pill-brand  { background: #ede9fe; color: #6d28d9; }
.pill-price  { background: #d1fae5; color: #065f46; }
.pill-score  { background: #e0e7ff; color: #3730a3; }
.pill-cat    { background: #fef3c7; color: #92400e; }
.pill-na     { background: #f1f5f9; color: #64748b; }

/* ---------- Metrics row ---------- */
.metrics-row {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.2rem;
    align-items: center;
}
.metric-badge {
    font-size: 0.88rem;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 8px;
    background: #f1f5f9;
    color: #475569;
}

/* ---------- Example query buttons ---------- */
.example-heading {
    font-size: 0.95rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* ---------- About section ---------- */
.about-text { color: #64748b; font-size: 0.92rem; line-height: 1.65; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc, #eef2ff);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders (run only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BM25 encoder…")
def load_bm25(encoder_path: str = "bm25_encoder.json") -> BM25Encoder:
    bm25 = BM25Encoder()
    bm25.load(encoder_path)
    return bm25


@st.cache_resource(show_spinner="Loading dense embedding model…")
def load_dense_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def load_pinecone_index(index_name: str = "ecommerce-search"):
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        st.error("❌ PINECONE_API_KEY is not set. Please add it to your .env file.")
        st.stop()
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


@st.cache_data(show_spinner="Loading product catalog…")
def load_dataset(csv_path: str = "data/products_clean.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for categories and hybrid scaling
# ─────────────────────────────────────────────────────────────────────────────
def clean_category_display(category_str: str) -> str:
    """
    Parse a category raw string like '["Footwear >> Women\'s Footwear >> Sports Shoes >> Running"]'
    and display as a clean breadcrumb.
    """
    if not category_str or not category_str.strip():
        return ""
    try:
        if category_str.startswith("["):
            cats = json.loads(category_str)
            if cats and isinstance(cats, list):
                category_str = cats[0]
        return " ➔ ".join(x.strip() for x in category_str.split(">>"))
    except Exception:
        return category_str


def extract_top_level_categories(dataframe: pd.DataFrame) -> list[str]:
    """Extract all unique top-level categories from the dataset."""
    unique_cats = set()
    for cat in dataframe['category'].dropna():
        try:
            if cat.startswith("["):
                cats_list = json.loads(cat)
                if cats_list and isinstance(cats_list, list):
                    path = cats_list[0]
                    parts = path.split(" >> ")
                    unique_cats.add(parts[0].strip())
        except Exception:
            pass
    return sorted(list(unique_cats))


def hybrid_scale(dense_vec, sparse_vec, alpha):
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values": [v * (1.0 - alpha) for v in sparse_vec["values"]],
    }
    return scaled_dense, scaled_sparse


def run_search(
    query_text: str,
    alpha: float,
    top_k: int,
    max_price: float | None,
    min_price: float | None,
    brand: str | None,
    category: str | None,
    bm25: BM25Encoder,
    model: SentenceTransformer,
    index,
    dataframe: pd.DataFrame,
) -> list[dict]:
    # Dense embedding
    dense_vec = model.encode(query_text, normalize_embeddings=True).tolist()

    # Sparse BM25 encoding
    raw_sparse = bm25.encode_queries(query_text)
    sparse_vec = {
        "indices": [int(i) for i in raw_sparse["indices"]],
        "values": [float(v) for v in raw_sparse["values"]],
    }

    # Convex combination scaling
    scaled_dense, scaled_sparse = hybrid_scale(dense_vec, sparse_vec, alpha)

    # Metadata filter
    query_filter = {}
    if max_price is not None or min_price is not None:
        query_filter["price_known"] = {"$eq": True}
        price_cond = {}
        if max_price is not None:
            price_cond["$lte"] = float(max_price)
        if min_price is not None:
            price_cond["$gte"] = float(min_price)
        query_filter["price"] = price_cond

    if brand:
        query_filter["brand"] = {"$eq": brand}

    if category:
        matching_cats = []
        for cat in dataframe['category'].dropna().unique():
            try:
                if cat.startswith("["):
                    cats_list = json.loads(cat)
                    if cats_list and isinstance(cats_list, list):
                        if cats_list[0].split(" >> ")[0].strip() == category:
                            matching_cats.append(cat)
            except Exception:
                pass
        if matching_cats:
            query_filter["category"] = {"$in": matching_cats}

    kwargs = dict(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    if query_filter:
        kwargs["filter"] = query_filter

    response = index.query(**kwargs)

    results = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        price_known = bool(meta.get("price_known", False))
        price = float(meta.get("price", 0.0))
        results.append({
            "id": match["id"],
            "score": round(float(match.get("score", 0.0)), 4),
            "title": meta.get("title", "N/A"),
            "brand": meta.get("brand", ""),
            "price": price,
            "price_known": price_known,
            "category": meta.get("category", ""),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
bm25 = load_bm25()
dense_model = load_dense_model()
pc_index = load_pinecone_index()
df = load_dataset()
categories_list = extract_top_level_categories(df)

dataset_max_price = float(df["price"].max())

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    st.markdown("---")

    alpha = st.slider(
        "Semantic vs Keyword Balance",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="**0.0** = pure keyword (BM25)  ·  **1.0** = pure semantic (dense).  "
             "The sweet spot for this dataset is **0.75**.",
    )

    st.markdown("")
    price_range = st.slider(
        "💰 Price Range (₹)",
        min_value=0,
        max_value=int(dataset_max_price),
        value=(0, int(dataset_max_price)),
        step=100,
    )

    st.markdown("")
    brand_input = st.text_input(
        "🏷️ Brand (exact match)",
        placeholder="e.g. Nike, Puma, Samsung",
    )

    st.markdown("")
    category_options = ["All Categories"] + categories_list
    category_input = st.selectbox(
        "📁 Category",
        options=category_options,
        index=0,
    )

    st.markdown("")
    top_k = st.slider("Results to return", min_value=1, max_value=20, value=10, step=1)

    st.markdown("---")
    search_clicked = st.button("🔍  Search", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 Hybrid Product Search</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Search powered by dense (semantic) + sparse (BM25) hybrid '
    'retrieval with metadata filtering</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Search bar
# ─────────────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Search",
    value=st.session_state.query,
    placeholder="Search for products… e.g. running shoes, formal shirt",
    label_visibility="collapsed",
)
# Sync back to session state
st.session_state.query = query

# ─────────────────────────────────────────────────────────────────────────────
# Example queries (shown when search box is empty)
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "running shoes",
    "wireless bluetooth headphones",
    "formal shirt for men",
    "kids toys under 500",
]

should_search = search_clicked or bool(query)

if not query:
    st.markdown('<p class="example-heading">✨ Try an example query:</p>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, eq in zip(cols, EXAMPLE_QUERIES):
        with col:
            if st.button(eq, key=f"ex_{eq}", use_container_width=True):
                st.session_state.query = eq
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Execute search & render results
# ─────────────────────────────────────────────────────────────────────────────
if query:
    # Resolve filter values
    min_price_val = price_range[0] if price_range[0] > 0 else None
    max_price_val = price_range[1] if price_range[1] < int(dataset_max_price) else None
    brand_val = brand_input.strip() if brand_input.strip() else None
    category_val = category_input if category_input != "All Categories" else None

    t0 = time.perf_counter()
    results = run_search(
        query_text=query,
        alpha=alpha,
        top_k=top_k,
        min_price=min_price_val,
        max_price=max_price_val,
        brand=brand_val,
        category=category_val,
        bm25=bm25,
        model=dense_model,
        index=pc_index,
        dataframe=df,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Metrics row
    st.markdown(
        f'<div class="metrics-row">'
        f'  <span class="metric-badge">📦 Found <b>{len(results)}</b> results</span>'
        f'  <span class="metric-badge">⏱️ {elapsed_ms:.0f} ms</span>'
        f'  <span class="metric-badge">⚖️ α = {alpha:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not results:
        st.info("No products matched your query and filters. Try broadening your search or adjusting the filters.")
    else:
        for r in results:
            price_html = (
                f'<span class="card-pill pill-price">₹ {r["price"]:,.0f}</span>'
                if r["price_known"]
                else '<span class="card-pill pill-na">Price N/A</span>'
            )
            brand_html = (
                f'<span class="card-pill pill-brand">{r["brand"]}</span>'
                if r["brand"]
                else ""
            )
            clean_cat = clean_category_display(r["category"])
            cat_html = (
                f'<span class="card-pill pill-cat">{clean_cat}</span>'
                if clean_cat
                else ""
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="card-title">{r["title"]}</div>
                    <div class="card-meta">
                        <span class="card-pill pill-score">Score: {r["score"]:.4f}</span>
                        {price_html}
                        {brand_html}
                        {cat_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# About section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️  About this project"):
    st.markdown(
        """
        <div class="about-text">
        <b>Hybrid Search</b> combines two complementary retrieval strategies:<br><br>
        <b>• Dense (Semantic)</b> — uses a sentence-transformer model (<code>all-MiniLM-L6-v2</code>)
        to encode queries and products as 384-dim vectors, capturing <em>meaning</em> beyond exact words.<br>
        <b>• Sparse (BM25)</b> — classic keyword matching via a fitted BM25 encoder, excelling at
        exact term recall.<br><br>
        The <b>alpha</b> parameter controls the blend: <code>α = 1</code> is pure semantic,
        <code>α = 0</code> is pure keyword. After evaluation with Precision@5 across a ground-truth
        test set, <b>α = 0.75</b> was found optimal — <b>44 % better</b> than pure-dense search.<br><br>
        <b>Tech stack:</b> Pinecone (serverless vector DB) · SentenceTransformers · BM25Encoder ·
        Streamlit · python-dotenv
        </div>
        """,
        unsafe_allow_html=True,
    )
