# Build from the REPO ROOT (the backend imports vision/, knowledge/ and data/raw):
#   docker build -f backend/Dockerfile -t factory-backend .      (docker compose does this for you)
# No secrets are copied into the image (.dockerignore excludes every .env); keys are passed at run time only.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# CPU-only torch keeps the image small (the image detector runs fine on CPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# The small text-embedding model is downloaded now so the first request does not wait for it. If the build has no internet the
# step is skipped and retrieval falls back to its keyword (TF-IDF) mode at run time.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo "embedding model not downloaded: retrieval will use TF-IDF"

COPY backend backend
COPY vision vision
COPY knowledge knowledge
COPY data/raw data/raw

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
