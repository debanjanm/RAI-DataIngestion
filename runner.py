from t import EmbeddingGenerator, UMAPProjector

# Step 1: Generate embeddings
eg = EmbeddingGenerator(api_url="http://localhost:1234/v1/embeddings", model="all-MiniLM-L6-v2")
df = eg.load_dataset("mydata.csv")  # must have a "text" column
df = eg.create_embeddings(df, text_col="text")
eg.save(df, "with_embeddings.parquet")

# # Step 2: Create UMAP projection
# proj = UMAPProjector(n_components=2)
# df_proj = proj.project(df)
# proj.save(df_proj, "umap_projection.parquet")
