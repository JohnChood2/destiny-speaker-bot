import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import requests

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

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
    
    chunks = results["documents"][0]
    metas  = results["metadatas"][0]
    
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
    answer = call_llm(prompt)
    
    return {"answer": answer, "sources": metas, "chunks": chunks}

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="The Speaker's Ghost", page_icon="👻", layout="centered")

# ── Destiny 2-inspired theme (custom CSS, no Bungie assets used) ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Titillium+Web:wght@300;400;600;700&display=swap');

:root {
    --d2-gold: #c8aa6e;
    --d2-gold-bright: #e8cf9a;
    --d2-cyan: #6fd6e8;
    --d2-bg: #05070d;
    --d2-panel: rgba(13, 20, 32, 0.72);
    --d2-border: rgba(200, 170, 110, 0.35);
    --d2-text: #e8e6e1;
}

html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
    color: var(--d2-text);
}

.stApp {
    background:
        radial-gradient(ellipse at 20% -10%, rgba(111, 214, 232, 0.08), transparent 55%),
        radial-gradient(ellipse at 80% 110%, rgba(200, 170, 110, 0.10), transparent 55%),
        var(--d2-bg);
}

/* ── Header block ── */
.ghost-header { text-align: center; margin-bottom: 0.5rem; }
.ghost-header .emblem {
    width: 46px; height: 46px; margin: 0 auto 0.6rem auto;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #fff, var(--d2-cyan) 40%, transparent 70%);
    box-shadow: 0 0 22px 4px rgba(111, 214, 232, 0.55);
}
.ghost-header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.18em;
    color: var(--d2-gold-bright);
    text-shadow: 0 0 18px rgba(200, 170, 110, 0.45);
    margin: 0;
    line-height: 1.1;
}
.ghost-header .rule {
    width: 220px; height: 1px; margin: 0.8rem auto;
    background: linear-gradient(90deg, transparent, var(--d2-gold), transparent);
}
.ghost-header .subtitle {
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--d2-cyan);
    opacity: 0.85;
}

/* ── Input field ── */
.stTextInput input {
    background-color: var(--d2-panel) !important;
    border: 1px solid var(--d2-border) !important;
    color: var(--d2-text) !important;
    font-family: 'Titillium Web', sans-serif;
    letter-spacing: 0.02em;
}
.stTextInput input:focus {
    border-color: var(--d2-cyan) !important;
    box-shadow: 0 0 12px rgba(111, 214, 232, 0.35) !important;
}
.stTextInput label {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.8rem;
    color: var(--d2-gold);
}

/* ── Answer panel, styled like a HUD transmission ── */
.transmission {
    border: 1px solid var(--d2-border);
    border-left: 3px solid var(--d2-gold);
    background: var(--d2-panel);
    padding: 1.1rem 1.4rem;
    border-radius: 2px;
    margin-top: 1rem;
    box-shadow: 0 0 24px rgba(0,0,0,0.35);
}
.transmission .label {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.16em;
    color: var(--d2-cyan);
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}

/* ── Expander (lore archive) ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--d2-border) !important;
    background: var(--d2-panel) !important;
    border-radius: 2px !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.1em;
    color: var(--d2-gold-bright) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--d2-cyan) !important;
}
</style>

<div class="ghost-header">
    <div class="emblem"></div>
    <h1>The Speaker's Ghost</h1>
    <div class="rule"></div>
    <div class="subtitle">Guardian // Query the Archive of Sung and Unsung Lore</div>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "Your question:",
    placeholder="Who created the Vex? What is the Darkness?"
)

if query:
    with st.spinner("Ghost is searching the lore..."):
        result = answer_question(query)

    st.markdown(
        f"""<div class="transmission">
            <div class="label">👻 Ghost Transmission</div>
            {result["answer"]}
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Archived lore // source passages"):
        for chunk, meta in zip(result["chunks"], result["sources"]):
            st.markdown(f"**{meta['title']}**")
            st.caption(chunk[:400] + "..." if len(chunk) > 400 else chunk)
            st.divider()
