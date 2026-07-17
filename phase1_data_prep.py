import os
import pandas as pd

def main():
    print("=" * 60)
    print("PHASE 1: DATA PREPARATION START")
    print("=" * 60)

    # 1. Load the CSV with pandas (handling the case if dataset path is a directory or file)
    input_path = "data/flipkart_com-ecommerce_sample.csv"
    if os.path.isdir(input_path):
        # If it is a directory, find the CSV file within it
        files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        if files:
            csv_path = os.path.join(input_path, files[0])
        else:
            raise FileNotFoundError(f"No CSV file found in directory {input_path}")
    else:
        csv_path = input_path

    print(f"[Step 1] Loading CSV from: {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully.\n")

    # 2. Print total rows, column names, and missing value counts per column
    print("-" * 50)
    print("[Step 2] Initial Dataset Summary:")
    print(f"Total Rows: {len(df)}")
    print(f"Column Names: {list(df.columns)}")
    print("\nMissing Value Counts per Column:")
    print(df.isnull().sum())
    print("-" * 50 + "\n")

    # 3. Print count of duplicate product_name values
    print("-" * 50)
    print("[Step 3] Checking duplicates of 'product_name'...")
    num_duplicates = df['product_name'].duplicated().sum()
    print(f"Count of duplicate 'product_name' values: {num_duplicates}")
    print("-" * 50 + "\n")

    # 4. Rename columns: product_name->title, retail_price->price, product_category_tree->category
    print("[Step 4] Renaming columns...")
    rename_mapping = {
        'product_name': 'title',
        'retail_price': 'price',
        'product_category_tree': 'category'
    }
    df = df.rename(columns=rename_mapping)
    print(f"Columns renamed using mapping: {rename_mapping}\n")

    # 5. Keep only these columns: title, description, brand, price, category
    print("[Step 5] Filtering columns...")
    keep_columns = ['title', 'description', 'brand', 'price', 'category']
    df = df[keep_columns]
    print(f"Kept columns: {keep_columns}\n")

    # 6. Drop rows where title is missing
    print("[Step 6] Dropping rows where 'title' is missing...")
    initial_len = len(df)
    df = df.dropna(subset=['title'])
    dropped_title = initial_len - len(df)
    print(f"Dropped {dropped_title} rows with missing 'title'. Remaining rows: {len(df)}\n")

    # 7. Fill missing description and brand with empty string
    print("[Step 7] Filling missing 'description' and 'brand' with empty string...")
    missing_desc = df['description'].isnull().sum()
    missing_brand = df['brand'].isnull().sum()
    df['description'] = df['description'].fillna('')
    df['brand'] = df['brand'].fillna('')
    print(f"Filled {missing_desc} missing descriptions and {missing_brand} missing brands.\n")

    # 8. Fill missing price with 0, and add a boolean column price_known
    print("[Step 8] Handling missing 'price' values...")
    # Add a boolean column price_known (True if not missing, False if missing)
    df['price_known'] = df['price'].notna()
    missing_price = (~df['price_known']).sum()
    
    # Standardize to numeric and fill NaN with 0
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(0.0)
    print(f"Filled {missing_price} missing prices with 0.0. Added 'price_known' column.\n")

    # 9. Drop duplicate rows based on title
    print("[Step 9] Dropping duplicate rows based on 'title'...")
    len_before_dup = len(df)
    df = df.drop_duplicates(subset=['title'])
    dropped_dups = len_before_dup - len(df)
    print(f"Dropped {dropped_dups} duplicate title rows. Remaining rows: {len(df)}\n")

    # 10. Randomly sample down to 15000 rows if there are more
    print("[Step 10] Sampling down dataset if necessary...")
    max_rows = 15000
    if len(df) > max_rows:
        print(f"Dataset has {len(df)} rows, which is more than {max_rows}. Sampling down to {max_rows} rows...")
        df = df.sample(n=max_rows, random_state=42)
    else:
        print(f"Dataset has {len(df)} rows, which is <= {max_rows}. No sampling needed.")
    print(f"Current row count: {len(df)}\n")

    # 11. Create a new column search_text that combines title + "Brand: " + brand + first 300 characters of description
    print("[Step 11] Creating 'search_text' column...")
    title_part = df['title'].astype(str)
    brand_part = "Brand: " + df['brand'].astype(str)
    desc_part = df['description'].astype(str).str[:300]
    df['search_text'] = title_part + ". " + brand_part + ". " + desc_part
    print("Created 'search_text' column by joining title, brand, and truncated description.\n")

    # 12. Add a unique product_id column like "prod_0", "prod_1", etc.
    print("[Step 12] Assigning unique product IDs...")
    df = df.reset_index(drop=True)
    df['product_id'] = 'prod_' + df.index.astype(str)
    print(f"Assigned IDs from prod_0 to prod_{len(df) - 1}.\n")

    # 13. Save the final cleaned dataframe to data/products_clean.csv
    output_csv = "data/products_clean.csv"
    print(f"[Step 13] Saving cleaned dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Dataset saved successfully.\n")

    # 14. At the end, print the final row count, final columns, and 3 sample search_text values
    print("=" * 60)
    print("FINAL SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Final Row Count: {len(df)}")
    print(f"Final Columns: {list(df.columns)}")
    print("\n" + "-" * 50)
    print("Sample search_text values for validation:")
    print("-" * 50)
    for i in range(min(3, len(df))):
        print(f"Sample {i + 1} (ID: {df.loc[i, 'product_id']}):")
        print(df.loc[i, 'search_text'])
        print("-" * 50)

    print("\nPHASE 1: DATA PREPARATION COMPLETED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    main()
