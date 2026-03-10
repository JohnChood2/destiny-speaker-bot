from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

from config import load_runtime_config

TOP_K = 4
MIN_SCORE = 0.2


@st.cache_resource
def get_clients() -> dict[str, Any]:
    """Initialize shared clients for embeddings, LLM, and Pinecone index."""
    cfg = load_runtime_config()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=cfg["openai_api_key"],
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=cfg["openai_api_key"],
        temperature=0.2,
    )
    index = Pinecone(api_key=cfg["pinecone_api_key"]).Index(cfg["pinecone_index_name"])
    return {"embeddings": embeddings, "llm": llm, "index": index}


def retrieve_context(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Fetch top-k matching lore chunks from Pinecone."""
    clients = get_clients()
    embeddings: OpenAIEmbeddings = clients["embeddings"]
    index = clients["index"]

    query_vector = embeddings.embed_query(question)
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    matches = list(result.matches) if hasattr(result, "matches") else result.get("matches", [])

    normalized: list[dict[str, Any]] = []
    for match in matches:
        metadata = getattr(match, "metadata", None) if not isinstance(match, dict) else match.get("metadata", {})
        score = getattr(match, "score", None) if not isinstance(match, dict) else match.get("score")
        normalized.append(
            {
                "title": (metadata or {}).get("title", "Unknown"),
                "text": (metadata or {}).get("text", ""),
                "hash": (metadata or {}).get("hash", ""),
                "score": float(score or 0.0),
            }
        )
    return normalized


def build_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    """Create a grounded prompt using retrieved source chunks."""
    if sources:
        context = "\n\n".join([f"Title: {s['title']}\n{s['text']}" for s in sources])
    else:
        context = "No lore context was retrieved."

    return (
        "You are The Speaker's Ghost, a Destiny 2 lore assistant.\n"
        "Answer only from the provided lore context.\n"
        "If the context is insufficient, say you are unsure and ask a narrower follow-up.\n\n"
        f"Question:\n{question}\n\n"
        f"Lore Context:\n{context}"
    )


def ask_lore(question: str) -> dict[str, Any]:
    """Single wrapper for retrieval + generation used by the frontend."""
    clients = get_clients()
    llm: ChatOpenAI = clients["llm"]

    sources = retrieve_context(question)
    strong_sources = [s for s in sources if s["score"] >= MIN_SCORE]
    prompt = build_prompt(question, strong_sources)
    answer = llm.invoke(prompt).content

    return {
        "answer": str(answer),
        "sources": strong_sources,
        "source_count": len(strong_sources),
    }
