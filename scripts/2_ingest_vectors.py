#!/usr/bin/env python3
"""Embed Destiny 2 lore CSV rows and upsert them into Pinecone."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LORE_CSV_PATH = DATA_DIR / "lore_raw.csv"
BATCH_SIZE = 100
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split long lore entries into overlapping chunks."""
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_lore_text(row: pd.Series) -> Tuple[str, str]:
    """Extract a title and body from a DestinyLoreDefinition row."""
    default_title = f"Lore {row.get('hash', row.name)}"
    if "json" not in row or pd.isna(row["json"]):
        return default_title, ""

    try:
        payload = json.loads(row["json"])
    except (TypeError, json.JSONDecodeError):
        return default_title, ""

    display = payload.get("displayProperties", {})
    title = display.get("name") or default_title
    description = display.get("description") or ""
    lore_body = payload.get("loreHash", "")
    # The lore text lives in displayProperties.description for this table.
    _ = lore_body
    return title, description


def build_documents(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """Build chunked text, metadata, and stable vector ids."""
    texts: List[str] = []
    metadata: List[Dict[str, str]] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        title, body = extract_lore_text(row)
        if not body.strip():
            continue

        base_id = str(row.get("hash", i))
        chunks = chunk_text(body)
        for chunk_idx, chunk in enumerate(chunks):
            vector_id = f"{base_id}-{chunk_idx}"
            texts.append(chunk)
            metadata.append(
                {
                    "title": title,
                    "hash": str(row.get("hash", base_id)),
                    "source": "DestinyLoreDefinition",
                    "chunk_index": str(chunk_idx),
                    "text": chunk,
                }
            )
            ids.append(vector_id)
    return texts, metadata, ids


def batched(items: Iterable, batch_size: int) -> Iterable[List]:
    """Yield successive item batches."""
    batch: List = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_index(pc: Pinecone, index_name: str, dimension: int, cloud: str, region: str) -> None:
    """Create index if missing, then wait for readiness."""
    index_list = pc.list_indexes()
    if hasattr(index_list, "names"):
        existing = set(index_list.names())
    else:
        existing = {idx["name"] for idx in index_list}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for Pinecone index to be ready...")
        time.sleep(2)


def main() -> None:
    """Run CSV -> embeddings -> Pinecone upsert pipeline."""
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "destiny-lore")
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is required.")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY is required.")
    if not LORE_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing lore CSV at {LORE_CSV_PATH}. Run 1_fetch_manifest.py first.")

    print(f"Loading lore CSV from {LORE_CSV_PATH}")
    df = pd.read_csv(LORE_CSV_PATH)
    texts, metadata, ids = build_documents(df)
    if not texts:
        raise ValueError("No lore text found in CSV. Check source extraction.")

    print(f"Prepared {len(texts)} chunks for embedding.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name, dimension=1536, cloud=pinecone_cloud, region=pinecone_region)
    index = pc.Index(index_name)

    total_upserted = 0
    for text_batch, meta_batch, id_batch in zip(
        batched(texts, BATCH_SIZE),
        batched(metadata, BATCH_SIZE),
        batched(ids, BATCH_SIZE),
    ):
        vectors = embeddings.embed_documents(text_batch)
        records = [
            {"id": vec_id, "values": vec, "metadata": meta}
            for vec_id, vec, meta in zip(id_batch, vectors, meta_batch)
        ]
        index.upsert(vectors=records)
        total_upserted += len(records)
        print(f"Upserted {total_upserted}/{len(texts)} vectors...")

    print(f"Done. Upserted {total_upserted} lore vectors into '{index_name}'.")


if __name__ == "__main__":
    main()

