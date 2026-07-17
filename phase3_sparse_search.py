import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# Load environment variables from .env file
load_dotenv()

def sparse_search(index, bm25, query_text, top_k=5):
    """
    Encodes query_text using the fitted BM25 encoder to get sparse values,
    and queries Pinecone using ONLY the sparse_vector parameter.
    """
    # 1. Encode query to sparse format
    sparse_val = bm25.encode_queries(query_text)
    
    # Ensure types are correct for Pinecone
    sparse_vector = {
        "indices": [int(ind) for ind in sparse_val["indices"]],
        "values": [float(val) for val in sparse_val["values"]]
    }

    # 2. Query Pinecone using sparse_vector and dummy dense vector (required for dense-primary indexes)
    results = index.query(
        vector=[0.0] * 384,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def main():
    print("=" * 60)
    print("PHASE 3: SPARSE SEARCH TEST")
    print("=" * 60)

    # Verify Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key.strip() == "":
        print("\n[ERROR] PINECONE_API_KEY environment variable is not set!")
        sys.exit(1)

    # Load fitted BM25 encoder
    encoder_path = "bm25_encoder.json"
    if not os.path.exists(encoder_path):
        print(f"[ERROR] Encoder file not found at {encoder_path}!")
        print("Please run phase3_upsert_sparse.py first.")
        sys.exit(1)
        
    print(f"Loading fitted BM25 encoder from {encoder_path}...")
    bm25 = BM25Encoder()
    bm25.load(encoder_path)
    print("BM25 encoder loaded successfully.\n")

    # Connect to index
    index_name = "ecommerce-search"
    print(f"Connecting to Pinecone client and index '{index_name}'...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print("Connected to index.\n")

    # Define test queries
    test_queries = [
        "running shoes",
        "waterproof jacket",
        "wireless bluetooth headphones",
        "kids toys under 500",
        "formal shirt for men"
    ]

    print("=" * 60)
    print("RUNNING SPARSE SEARCH QUERY TESTS (BM25)")
    print("=" * 60)

    for q in test_queries:
        print(f"\nSPARSE QUERY: '{q}'")
        print("-" * 50)
        try:
            results = sparse_search(index, bm25, q, top_k=5)
            
            matches = results.get("matches", [])
            if not matches:
                print("No matches found.")
            else:
                for idx, match in enumerate(matches):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    title = meta.get("title", "N/A")
                    brand = meta.get("brand", "N/A")
                    price = meta.get("price", 0.0)
                    price_known = meta.get("price_known", False)
                    
                    price_str = f"Rs. {price:.2f}" if price_known else "N/A (Missing)"
                    
                    print(f" {idx + 1}. [Score: {score:.4f}] {title}")
                    print(f"    Brand: {brand} | Price: {price_str}")
                    print(f"    Vector ID: {match.get('id')}")
                    print("-" * 40)
        except Exception as e:
            print(f"Error running sparse search: {e}")
            
    print("=" * 60)
    print("PHASE 3: SPARSE SEARCH COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
