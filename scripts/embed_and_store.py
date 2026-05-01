import json, pathlib, chromadb
from sentence_transformers import SentenceTransformer

LORE_PATH = pathlib.Path("data/lore_entries.json")
CHUNK_SIZE = 300   # tokens (approximate via word count)
CHUNK_OVERLAP = 50

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks

def main():
    entries = json.loads(LORE_PATH.read_text())
    
    # Load a lightweight but capable embedding model (runs locally, free)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Set up local ChromaDB (persisted to disk)
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection(
        name="destiny_lore",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids, docs, metas, embeddings = [], [], [], []
    
    for entry in entries:
        chunks = chunk_text(entry["body"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{entry['hash']}_chunk_{i}"
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"title": entry["title"], "hash": str(entry["hash"])})
            embeddings.append(model.encode(chunk).tolist())
    
    # Upsert in batches of 500
    batch = 500
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i:i+batch],
            documents=docs[i:i+batch],
            metadatas=metas[i:i+batch],
            embeddings=embeddings[i:i+batch],
        )
        print(f"Stored batch {i//batch + 1} / {len(ids)//batch + 1}")
    
    print(f"Done! {len(ids)} chunks stored in ChromaDB.")

if __name__ == "__main__":
    main()