# Build from the REPO ROOT (the backend imports vision/, knowledge/ and data/):
#   docker build -f backend/Dockerfile -t factory-backend .
# No secrets are copied into the image; pass them at run time (docker compose uses backend/.env).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image small (the detector runs fine on CPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend backend
COPY vision vision
COPY knowledge knowledge
COPY data/raw data/raw

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
