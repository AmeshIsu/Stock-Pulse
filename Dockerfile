FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-backend.txt

# Copy application source, data, models, and migrations
COPY app/ app/
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY alembic/ alembic/
COPY alembic.ini .
COPY .env.example .env

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.backend.main:app --host 0.0.0.0 --port 8000"]
