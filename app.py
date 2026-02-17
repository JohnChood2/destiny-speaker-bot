from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

TOP_K = 4


def get_config_value(name: str, default: str | None = None) -> str | None:
    """Read config from env vars first, then Streamlit secrets."""
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return default


@st.cache_resource
def get_clients() -> Dict[str, object]:
    """Initialize and cache API clients."""
    load_dotenv()

    openai_api_key = get_config_value("OPENAI_API_KEY")
    pinecone_api_key = get_config_value("PINECONE_API_KEY")
    index_name = get_config_value("PINECONE_INDEX_NAME", "destiny-lore")

    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is missing.")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY is missing.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
    index = Pinecone(api_key=pinecone_api_key).Index(index_name)
    return {"embeddings": embeddings, "llm": llm, "index": index}


def retrieve_context(query: str, clients: Dict[str, object]) -> List[Dict]:
    """Retrieve lore chunks from Pinecone for the query."""
    embeddings: OpenAIEmbeddings = clients["embeddings"]  # type: ignore[assignment]
    index = clients["index"]
    query_vector = embeddings.embed_query(query)
    result = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
    if hasattr(result, "matches"):
        return list(result.matches)
    return result.get("matches", [])


def generate_answer(question: str, matches: List[Dict], clients: Dict[str, object]) -> str:
    """Generate grounded answer using retrieved context."""
    llm: ChatOpenAI = clients["llm"]  # type: ignore[assignment]
    context_chunks = []
    for match in matches:
        md = match.get("metadata", {})
        title = md.get("title", "Unknown")
        text = md.get("text", "")
        context_chunks.append(f"Title: {title}\n{text}")

    context = "\n\n".join(context_chunks) if context_chunks else "No lore context retrieved."
    prompt = (
        "You are The Speaker's Ghost, an assistant for Destiny 2 lore.\n"
        "Use only the provided lore context to answer the user's question.\n"
        "If context is insufficient, say you are unsure and ask for a narrower question.\n\n"
        f"Question:\n{question}\n\n"
        f"Lore Context:\n{context}"
    )
    return llm.invoke(prompt).content


def main() -> None:
    """Render the Streamlit Destiny lore RAG app."""
    st.set_page_config(page_title="The Speaker's Ghost", page_icon=":crystal_ball:")
    st.title("The Speaker's Ghost")
    st.caption("Ask questions about Destiny 2 lore")

    try:
        clients = get_clients()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    question = st.text_input("Ask a lore question", placeholder="Who is Savathun?")
    if not question:
        return

    with st.spinner("Searching the archives..."):
        matches = retrieve_context(question, clients)
        answer = generate_answer(question, matches, clients)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    if not matches:
        st.write("No sources found.")
        return

    for i, match in enumerate(matches, start=1):
        md = match.get("metadata", {})
        title = md.get("title", "Unknown")
        score = match.get("score", 0.0)
        snippet = (md.get("text", "") or "")[:220]
        st.markdown(f"**{i}. {title}** (score: {score:.3f})")
        st.write(snippet + ("..." if len(snippet) == 220 else ""))


if __name__ == "__main__":
    main()

