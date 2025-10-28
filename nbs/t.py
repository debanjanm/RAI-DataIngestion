import pandas as pd
import numpy as np
import requests
import json
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from umap import UMAP


class EmbeddingGenerator:
    """
    Loads dataset, generates embeddings via LM Studio embedding API,
    and saves enriched dataset with embeddings.
    """

    def __init__(self, api_url="http://localhost:1234/v1/embeddings", model="embedding-model"):
        self.api_url = api_url
        self.model = model

    def load_dataset(self, path: str) -> pd.DataFrame:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".json"):
            df = pd.read_json(path, lines=True)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or Parquet.")
        return df

    def get_embedding(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def create_embeddings(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        embeddings = []
        for txt in df[text_col].tolist():
            emb = self.get_embedding(txt)
            embeddings.append(emb)
        df["embedding"] = embeddings
        return df

    def save(self, df: pd.DataFrame, path: str):
        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        elif path.endswith(".json"):
            df.to_json(path, orient="records", lines=True)
        elif path.endswith(".parquet"):
            # handle list<float> correctly
            arr = pa.array([np.array(e, dtype=np.float32).tolist() for e in df["embedding"]],
                           type=pa.list_(pa.float32()))
            table = pa.Table.from_pandas(df.drop(columns=["embedding"]))
            table = table.append_column("embedding", arr)
            pq.write_table(table, path)
        else:
            raise ValueError("Unsupported output format. Use CSV, JSON, or Parquet.")


class UMAPProjector:
    """
    Projects embeddings into 2D or 3D space with UMAP and saves results.
    """

    def __init__(self, n_components=2, random_state=42, metric="cosine"):
        self.umap = UMAP(n_components=n_components, random_state=random_state, metric=metric)

    def project(self, df: pd.DataFrame, embedding_col="embedding") -> pd.DataFrame:
        embeddings = np.array(df[embedding_col].tolist())
        projections = self.umap.fit_transform(embeddings)
        for i in range(projections.shape[1]):
            df[f"umap_{i+1}"] = projections[:, i]
        return df

    def save(self, df: pd.DataFrame, path: str):
        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        elif path.endswith(".json"):
            df.to_json(path, orient="records", lines=True)
        elif path.endswith(".parquet"):
            pq.write_table(pa.Table.from_pandas(df), path)
        else:
            raise ValueError("Unsupported output format. Use CSV, JSON, or Parquet.")
