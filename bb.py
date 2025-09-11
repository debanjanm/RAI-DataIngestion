import os
import csv
import lms as lms
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def embed_texts(model_key, texts):
    """Worker function: embed a small list of texts."""
    model = lms.embedding_model(model_key)
    embeddings = [model.embed(t) for t in texts]
    return embeddings

class ArxivEmbeddingFastAdderWithProgress:
    def __init__(
        self,
        save_dir="arxiv_data",
        chunk_csv="all_chunks.csv",
        model_key="text-embedding-qwen3-embedding-4b",
        batch_size=32,
        num_workers=4
    ):
        self.save_dir = save_dir
        self.chunk_csv = os.path.join(save_dir, chunk_csv)
        self.output_parquet = os.path.join(save_dir, "all_chunks_with_embeddings.parquet")
        self.model_key = model_key
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run(self):
        if not os.path.exists(self.chunk_csv):
            print(f"Error: {self.chunk_csv} not found.")
            return

        # Read all chunk rows
        rows = []
        with open(self.chunk_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        total = len(rows)
        print(f"Total chunks to embed: {total}")

        writer = None

        # Use tqdm to track the batch progress
        for batch_start in tqdm(range(0, total, self.batch_size), desc="Embedding batches"):
            batch = rows[batch_start : batch_start + self.batch_size]
            texts = [r["chunk_text"] for r in batch]

            # Parallel embedding
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                splits = []
                size_per_worker = max(1, len(texts) // self.num_workers)
                for i in range(self.num_workers):
                    s = texts[i * size_per_worker : (i + 1) * size_per_worker]
                    if s:
                        splits.append(s)
                future_to_split = {executor.submit(embed_texts, self.model_key, s): s for s in splits}
                embeddings = []
                for future in as_completed(future_to_split):
                    embeddings.extend(future.result())

            # Combine rows with embeddings
            for i, r in enumerate(batch):
                r["embedding"] = embeddings[i]

            # Convert batch to PyArrow table
            table = pa.Table.from_pydict({k: [r[k] for r in batch] for k in batch[0].keys()})

            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(self.output_parquet, schema=schema, compression="SNAPPY")

            writer.write_table(table)

        if writer:
            writer.close()
        print(f"✅ Fast embeddings saved to {self.output_parquet}")

if __name__ == "__main__":

    # Instantiate your class with the command line arguments
    adder = ArxivEmbeddingFastAdderWithProgress()

    # Run embedding generation
    adder.run()
