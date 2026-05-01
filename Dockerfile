FROM python:3.12-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py ./
COPY scripts/ ./scripts/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
