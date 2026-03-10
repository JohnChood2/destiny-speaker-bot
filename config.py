from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv


def load_config_value(name: str, default: str | None = None) -> str | None:
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


def load_runtime_config() -> dict[str, str]:
    """Load and validate runtime configuration values."""
    load_dotenv()

    config = {
        "openai_api_key": load_config_value("OPENAI_API_KEY", ""),
        "pinecone_api_key": load_config_value("PINECONE_API_KEY", ""),
        "pinecone_index_name": load_config_value("PINECONE_INDEX_NAME", "destiny-lore"),
    }

    if not config["openai_api_key"]:
        raise EnvironmentError("OPENAI_API_KEY is missing.")
    if not config["pinecone_api_key"]:
        raise EnvironmentError("PINECONE_API_KEY is missing.")
    if not config["pinecone_index_name"]:
        raise EnvironmentError("PINECONE_INDEX_NAME is missing.")

    return config
