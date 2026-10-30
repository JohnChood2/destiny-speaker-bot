import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import requests

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
# Chunks with a cosine distance above this are treated as unrelated to the question
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))
NO_LORE_ANSWER = (
    "I searched everything I know, Guardian, but I can't find anything in the lore about that."
)

# ── Ollama LLM backend ───────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]
    
# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("destiny_lore")
    return model, collection

embed_model, collection = load_resources()

# ── RAG function ──────────────────────────────────────────────────────────────
def answer_question(user_query: str, top_k: int = 5) -> dict:
    # 1. Embed the query
    query_embedding = embed_model.encode(user_query).tolist()
    
    # 2. Retrieve top-k most relevant lore chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    kept = [
        (c, m)
        for c, m, d in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if d <= MAX_DISTANCE
    ]
    if not kept:
        return {"answer": NO_LORE_ANSWER, "sources": [], "chunks": []}
    chunks = [c for c, _ in kept]
    metas  = [m for _, m in kept]
    
    # 3. Build the prompt
    context = "\n\n---\n\n".join(
        [f"[{m['title']}]\n{c}" for c, m in zip(chunks, metas)]
    )
    prompt = f"""You are the Ghost from Destiny 2. Answer the Guardian's question 
using ONLY the lore excerpts below. Cite the lore book title when possible.
Be concise but atmospheric.

LORE EXCERPTS:
{context}

GUARDIAN'S QUESTION: {user_query}

GHOST'S ANSWER:"""
    
    # 4. Generate the answer via Ollama
    try:
        answer = call_llm(prompt)
    except requests.RequestException as e:
        return {
            "answer": f"My connection to the model at {OLLAMA_HOST} failed. Is Ollama running "
                      f"and is `{OLLAMA_MODEL}` pulled? ({e.__class__.__name__})",
            "sources": metas,
            "chunks": chunks,
        }
    
    return {"answer": answer, "sources": metas, "chunks": chunks}

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="The Speaker's Ghost", page_icon="👻")
st.title("👻 The Speaker's Ghost")
st.caption("Ask anything about the Destiny 2 universe. Answers drawn directly from in-game lore.")

query = st.text_input(
    "Your question:",
    placeholder="Who created the Vex? What is the Darkness?"
)

if query:
    with st.spinner("Ghost is searching the lore..."):
        result = answer_question(query)
    
    st.markdown("### Answer")
    st.write(result["answer"])
    
    with st.expander("📖 Source lore passages"):
        for chunk, meta in zip(result["chunks"], result["sources"]):
            st.markdown(f"**{meta['title']}**")
            st.caption(chunk[:400] + "..." if len(chunk) > 400 else chunk)
            st.divider()
