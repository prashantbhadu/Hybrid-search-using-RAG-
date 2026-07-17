import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

def search(pc_index, model, query_text, top_k=5):
    """
    Embeds query_text using the model, queries the Pinecone index,
    and returns top_k matches.
    """
    # 1. Embed query
    query_vector = model.encode(query_text).tolist()
    query_vector = [float(val) for val in query_vector]

    # 2. Query Pinecone
    results = pc_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def main():
    print("=" * 60)
    print("PHASE 2: QUERY TEST")
    print("=" * 60)

    # Verify Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key.strip() == "":
        print("\n[ERROR] PINECONE_API_KEY environment variable is not set!")
        print("Please set your API key in the environment before running. Example:")
        print('  Windows PowerShell:   $env:PINECONE_API_KEY="your_api_key_here"')
        print('  Command Prompt (cmd):  set PINECONE_API_KEY=your_api_key_here')
        sys.exit(1)

    # 3.a. Load the same embedding model
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.\n")

    # 3.b. Connect to the same Pinecone index
    index_name = "ecommerce-search"
    print(f"Connecting to Pinecone client and index '{index_name}'...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists before attempting connection
    try:
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
    except Exception:
        try:
            index_names = pc.list_indexes().names()
        except Exception:
            index_names = [index_name] # fallback to check directly
            
    if index_name not in index_names:
        print(f"[ERROR] Index '{index_name}' does not exist! Please run phase2_embed_and_upsert.py first to create and populate it.")
        sys.exit(1)

    index = pc.Index(index_name)
    print("Connected to index.\n")

    # 3.d. Runs this test function on the 5 example queries and prints results nicely formatted
    test_queries = [
        "running shoes",
        "waterproof jacket",
        "wireless bluetooth headphones",
        "kids toys under 500",
        "formal shirt for men"
    ]

    print("=" * 60)
    print("RUNNING SEARCH QUERY TESTS")
    print("=" * 60)

    for q in test_queries:
        print(f"\nQUERY: '{q}'")
        print("-" * 50)
        try:
            results = search(index, model, q, top_k=5)
            
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
            print(f"Error running search query: {e}")
            
    print("=" * 60)
    print("PHASE 2: QUERY TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
