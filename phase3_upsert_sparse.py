import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# Load environment variables from .env file
load_dotenv()

def main():
    print("=" * 60)
    print("PHASE 3: UPSERT BOTH DENSE AND SPARSE VECTORS")
    print("=" * 60)

    # Verify Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key.strip() == "":
        print("\n[ERROR] PINECONE_API_KEY environment variable is not set!")
        sys.exit(1)

    # 1. Load data/products_clean.csv
    cleaned_csv_path = "data/products_clean.csv"
    if not os.path.exists(cleaned_csv_path):
        print(f"[ERROR] Cleaned product dataset not found at {cleaned_csv_path}!")
        print("Please run phase1_data_prep.py first.")
        sys.exit(1)
        
    print(f"Loading cleaned dataset from {cleaned_csv_path}...")
    df = pd.read_csv(cleaned_csv_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} products.\n")

    # 2. Fit the BM25 encoder on the full search_text column
    print("Initializing BM25 encoder...")
    bm25 = BM25Encoder()
    print("Fitting BM25 encoder on search_text corpus...")
    
    # Ensure search_text contains strings
    search_texts = df['search_text'].astype(str).tolist()
    bm25.fit(search_texts)
    
    encoder_path = "bm25_encoder.json"
    print(f"Saving fitted BM25 encoder to {encoder_path}...")
    bm25.dump(encoder_path)
    print("BM25 encoder saved successfully.\n")

    # 3. Load the sentence-transformers model to get dense embeddings
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model '{model_name}'...")
    dense_model = SentenceTransformer(model_name)
    print("Dense embedding model loaded successfully.\n")

    # Generate dense embeddings
    print("Generating dense embeddings in batches of 128...")
    dense_embeddings = []
    batch_size = 128
    for i in range(0, total_rows, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_texts = batch_df['search_text'].astype(str).tolist()
        batch_embeddings = dense_model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        dense_embeddings.extend(batch_embeddings)
    print("Dense embeddings generated.\n")

    # 4. Connect to Pinecone and prepare index
    print("Connecting to Pinecone client...")
    pc = Pinecone(api_key=api_key)
    index_name = "ecommerce-search"
    
    print(f"Checking configuration of index '{index_name}'...")
    index_exists = False
    recreate_needed = False
    
    try:
        desc = pc.describe_index(index_name)
        index_exists = True
        current_metric = desc.metric
        if current_metric != "dotproduct":
            print(f"Index exists with metric '{current_metric}' instead of 'dotproduct'. Re-creation required.")
            recreate_needed = True
        else:
            print(f"Index exists with correct metric ('dotproduct').")
    except Exception:
        # Index does not exist
        print(f"Index '{index_name}' does not exist yet.")
        index_exists = False

    if index_exists and recreate_needed:
        print(f"Deleting index '{index_name}' to change metric...")
        pc.delete_index(index_name)
        import time
        print("Waiting for index deletion to complete...")
        time.sleep(15)
        index_exists = False

    if not index_exists:
        from pinecone import ServerlessSpec
        print(f"Creating serverless index '{index_name}' with metric='dotproduct'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        import time
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status.ready:
            time.sleep(2)
        print("Index is ready.")

    index = pc.Index(index_name)
    print("Connected to Pinecone index.\n")

    # 5. Build combined vector payload
    print("Preparing vector payload with both dense and sparse vectors...")
    vectors = []
    
    for idx, row in df.iterrows():
        # Encode document text for BM25 sparse representation
        # It yields a dict: {"indices": [int, ...], "values": [float, ...]}
        text_content = str(row['search_text'])
        sparse_val = bm25.encode_documents(text_content)
        
        # Ensure indices and values are standard Python types
        sparse_val = {
            "indices": [int(ind) for ind in sparse_val["indices"]],
            "values": [float(val) for val in sparse_val["values"]]
        }
        
        dense_val = [float(v) for v in dense_embeddings[idx]]
        
        meta = {
            "title": str(row['title']) if pd.notna(row['title']) else "",
            "brand": str(row['brand']) if pd.notna(row['brand']) else "",
            "price": float(row['price']) if pd.notna(row['price']) else 0.0,
            "category": str(row['category']) if pd.notna(row['category']) else "",
            "price_known": bool(row['price_known'])
        }
        
        vectors.append({
            "id": str(row['product_id']),
            "values": dense_val,
            "sparse_values": sparse_val,
            "metadata": meta
        })

    # Upsert to Pinecone
    upsert_batch_size = 100
    total_vectors = len(vectors)
    print(f"Upserting {total_vectors} vectors (dense + sparse) to index '{index_name}' in batches of {upsert_batch_size}...")
    
    for i in range(0, total_vectors, upsert_batch_size):
        batch = vectors[i : i + upsert_batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted vectors {i} to {min(i + upsert_batch_size, total_vectors)}")
        
    print("\nUpsert of dense + sparse vectors completed successfully.\n")
    print("-" * 50)
    print("Index Statistics Summary:")
    print("-" * 50)
    try:
        stats = index.describe_index_stats()
        print(stats)
    except Exception as e:
        print(f"Could not retrieve index stats: {e}")
        
    print("=" * 60)
    print("PHASE 3: UPSERT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
