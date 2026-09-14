FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy the rest
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]