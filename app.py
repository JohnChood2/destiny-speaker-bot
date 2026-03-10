import streamlit as st

from config import load_runtime_config
from rag_service import ask_lore


def render_sources(sources: list[dict]) -> None:
    """Render source cards for a response."""
    if not sources:
        st.write("No high-confidence sources found.")
        return

    for idx, source in enumerate(sources, start=1):
        st.markdown(f"**{idx}. {source['title']}** (score: {source['score']:.3f})")
        snippet = (source.get("text", "") or "")[:260]
        st.write(snippet + ("..." if len(snippet) == 260 else ""))


def main() -> None:
    """Render chat UI that wraps the RAG service."""
    st.set_page_config(page_title="The Speaker's Ghost", page_icon=":crystal_ball:")
    st.title("The Speaker's Ghost")
    st.caption("Ask questions about Destiny 2 lore and get grounded answers.")

    try:
        load_runtime_config()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Session")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources") is not None:
                with st.expander("Sources"):
                    render_sources(message["sources"])

    question = st.chat_input("Ask a lore question (e.g., Who is Savathun?)")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            result = ask_lore(question)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


if __name__ == "__main__":
    main()

