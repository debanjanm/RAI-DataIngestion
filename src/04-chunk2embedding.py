import os
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import nbs.lms as lms

class ArxivEmbeddingStreamer:
    def __init__(self, save_dir="arxiv_data", chunk_csv="all_chunks.csv", model_key="text-embedding-qwen3-embedding-4b"):
        self.save_dir = save_dir
        self.chunk_csv = os.path.join(save_dir, chunk_csv)
        self.output_parquet = os.path.join(save_dir, "all_chunks_with_embeddings.parquet")
        self.model = lms.embedding_model(model_key)

    def run(self):
        if not os.path.exists(self.chunk_csv):
            print(f"Error: {self.chunk_csv} not found.")
            return

        # Initialize variables for streaming
        writer = None
        schema = None

        with open(self.chunk_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row_idx, row in enumerate(reader, start=1):
                chunk_text = row["chunk_text"]
                embedding = self.model.embed(chunk_text)

                # Add embedding as a list to row
                row["embedding"] = embedding

                # Create a PyArrow Table from this row if it's the first
                table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
                
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(self.output_parquet, schema=schema, compression='SNAPPY')
                
                writer.write_table(table)

                if row_idx % 100 == 0:
                    print(f"Processed {row_idx} chunks...")

        if writer:
            writer.close()
        
        print(f"Streaming embeddings saved to {self.output_parquet}")


if __name__ == "__main__":
    streamer = ArxivEmbeddingStreamer(save_dir="arxiv_data")
    streamer.run()

