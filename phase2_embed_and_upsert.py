import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

def main():
    print("=" * 60)
    print("PHASE 2: EMBED AND UPSERT")
    print("=" * 60)

    # Verify Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key.strip() == "":
        print("\n[ERROR] PINECONE_API_KEY environment variable is not set!")
        print("Please set your API key in the environment before running. Example:")
        print('  Windows PowerShell:   $env:PINECONE_API_KEY="your_api_key_here"')
        print('  Command Prompt (cmd):  set PINECONE_API_KEY=your_api_key_here')
        sys.exit(1)

    # 2.a. Load data/products_clean.csv
    cleaned_csv_path = "data/products_clean.csv"
    if not os.path.exists(cleaned_csv_path):
        print(f"[ERROR] Cleaned product dataset not found at {cleaned_csv_path}!")
        print("Please run phase1_data_prep.py first.")
        sys.exit(1)
        
    print(f"Loading cleaned dataset from {cleaned_csv_path}...")
    df = pd.read_csv(cleaned_csv_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} products.\n")

    # 2.b. Load the sentence-transformers model "all-MiniLM-L6-v2"
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.\n")

    # 2.c. Generate embeddings in batches of 128
    print("-" * 50)
    print("Generating embeddings for 'search_text' in batches of 128...")
    print("-" * 50)
    
    embeddings = []
    batch_size = 128
    
    for i in range(0, total_rows, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_texts = batch_df['search_text'].astype(str).tolist()
        
        # Generate embeddings
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        
        current_batch = i // batch_size + 1
        total_batches = (total_rows + batch_size - 1) // batch_size
        print(f"  Batch {current_batch}/{total_batches} processed (Rows {i} to {min(i + batch_size, total_rows)})")
        
    print(f"Successfully generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.\n")

    # 2.d. Connect to Pinecone using API key
    print("Connecting to Pinecone client...")
    pc = Pinecone(api_key=api_key)
    print("Pinecone client initialized.\n")

    # 2.e. Create a serverless Pinecone index called "ecommerce-search"
    index_name = "ecommerce-search"
    dimension = 384
    metric = "cosine"
    
    print(f"Checking if Pinecone index '{index_name}' exists...")
    
    # Defensive check for list of indexes across different SDK versions
    index_exists = False
    try:
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        index_exists = index_name in index_names
    except Exception as e:
        print(f"Warning/Error fetching index names: {e}. Trying fallback methods...")
        try:
            index_names = pc.list_indexes().names()
            index_exists = index_name in index_names
        except Exception:
            try:
                index_names = [idx['name'] for idx in pc.list_indexes().to_dict().get('indexes', [])]
                index_exists = index_name in index_names
            except Exception as final_e:
                print(f"Fallback methods failed: {final_e}")
                # We will assume it does not exist and try to create, catching already exists error
                index_exists = False

    if not index_exists:
        print(f"Index '{index_name}' does not exist. Creating serverless index...")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Successfully created index '{index_name}'.")
        except Exception as e:
            # If creation fails because it actually exists (race condition/etc.)
            if "already exists" in str(e).lower():
                print(f"Index '{index_name}' already exists (verified on creation request).")
            else:
                print(f"Error creating index: {e}")
                sys.exit(1)
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

    # Connect to the index
    print(f"Connecting to index '{index_name}'...")
    index = pc.Index(index_name)
    print("Connected to index.\n")

    # 2.f. Upsert all products in batches of 100
    print("-" * 50)
    print("Preparing vector payload for upsert...")
    print("-" * 50)
    
    vectors = []
    for idx, row in df.iterrows():
        # Ensure data types are strictly compatible with Pinecone metadata (no NaNs, lists, or custom numpy types)
        meta = {
            "title": str(row['title']) if pd.notna(row['title']) else "",
            "brand": str(row['brand']) if pd.notna(row['brand']) else "",
            "price": float(row['price']) if pd.notna(row['price']) else 0.0,
            "category": str(row['category']) if pd.notna(row['category']) else "",
            "price_known": bool(row['price_known'])
        }
        
        # Ensure values is a list of Python floats
        values = [float(v) for v in embeddings[idx]]
        
        vectors.append({
            "id": str(row['product_id']),
            "values": values,
            "metadata": meta
        })

    upsert_batch_size = 100
    total_vectors = len(vectors)
    print(f"Upserting {total_vectors} vectors to index '{index_name}' in batches of {upsert_batch_size}...")
    
    for i in range(0, total_vectors, upsert_batch_size):
        batch = vectors[i : i + upsert_batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted vectors {i} to {min(i + upsert_batch_size, total_vectors)}")
        
    print("\nUpsert completed successfully.\n")

    # 2.g. Print total vectors upserted at the end and the index stats
    print("-" * 50)
    print("Index Statistics Summary:")
    print("-" * 50)
    try:
        stats = index.describe_index_stats()
        print(stats)
    except Exception as e:
        print(f"Could not retrieve index stats: {e}")
        
    print("=" * 60)
    print("PHASE 2: EMBED AND UPSERT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
