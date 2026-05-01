# The Speaker's Ghost

A Destiny 2 lore chatbot powered by Retrieval-Augmented Generation (RAG). Ask any question about the Destiny universe and get answers drawn directly from in-game lore books, narrated in the voice of your Ghost.

## Architecture

```
User Question
     │
     ▼
┌──────────┐    query     ┌───────────┐    prompt    ┌────────┐
│ Streamlit │ ──────────► │ ChromaDB  │ ──────────► │ Ollama │
│    UI     │             │ (vectors) │             │(Mistral)│
└──────────┘ ◄────────── └───────────┘ ◄────────── └────────┘
                answer         context (top-k lore chunks)
```

1. The user's question is embedded using `all-MiniLM-L6-v2`
2. ChromaDB finds the most relevant lore passages via cosine similarity
3. The retrieved passages are injected into a prompt sent to Mistral via Ollama
4. The LLM generates an answer grounded in actual Destiny 2 lore

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector DB | ChromaDB (persistent, local) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| LLM | Mistral via Ollama |
| Data Source | Bungie API (Destiny 2 Manifest) |
| Containerization | Docker + Docker Compose |

## Prerequisites

- [Python 3.12+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com/) (local LLM runtime)
- [Docker](https://www.docker.com/products/docker-desktop/) (for containerized deployment)
- A [Bungie API key](https://www.bungie.net/en/Application) (free)

## Quick Start (Local)

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/speakers-ghost.git
cd speakers-ghost
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable | Description | Required |
|----------|-------------|----------|
| `BUNGIE_API_KEY` | Your Bungie API key for downloading the manifest | Yes |
| `HF_API_TOKEN` | HuggingFace token (for gated models, if needed) | No |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) | No |
| `OLLAMA_MODEL` | Which Ollama model to use (default: `mistral`) | No |

### 3. Start Ollama and pull the model

```bash
ollama serve
```

In a separate terminal:

```bash
ollama pull mistral
```

### 4. Build the lore database

Run the data pipeline scripts in order:

```bash
# Download the Destiny 2 manifest from Bungie
uv run python scripts/download_manifest.py

# Parse lore entries from the manifest database
uv run python scripts/parse_lore.py

# Embed lore chunks and store in ChromaDB
uv run python scripts/embed_and_store.py
```

### 5. Run the app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker Deployment

### 1. Build the lore database first

The data pipeline requires your Bungie API key and runs outside Docker. Follow steps 2-4 from the Quick Start above to populate the `data/` directory.

### 2. Start the containers

```bash
docker compose up --build
```

This starts two services:

- **app** — the Streamlit frontend on port `8501`
- **ollama** — the Ollama LLM server on port `11434`

### 3. Pull the model into the Ollama container

On first run, pull Mistral into the containerized Ollama instance:

```bash
docker compose exec ollama ollama pull mistral
```

The model is stored in a persistent Docker volume (`ollama_data`), so this only needs to happen once.

### 4. Open the app

Navigate to [http://localhost:8501](http://localhost:8501).

## Project Structure

```
speakers-ghost/
├── app.py                  # Streamlit app + RAG pipeline
├── scripts/
│   ├── download_manifest.py   # Fetch Destiny 2 manifest from Bungie API
│   ├── parse_lore.py          # Extract lore entries to JSON
│   └── embed_and_store.py     # Embed lore chunks into ChromaDB
├── data/                   # Generated data (gitignored)
│   ├── chroma_db/             # ChromaDB vector store
│   ├── lore_entries.json      # Parsed lore entries
│   └── lore_raw/              # Raw manifest database
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .env.example
└── .gitignore
```

## License

MIT
