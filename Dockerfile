FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application files ──────────────────────────────────────────────────────────
# Copy application source
COPY app.py          .
COPY inference.py    .
COPY openenv.yaml    .
COPY environment/    ./environment/

# Copy graders package
COPY graders/        ./graders/

# ── Environment variables ──────────────────────────────────────────────────────
# These are overridden at runtime via docker run -e or HF Space secrets
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""
ENV ENV_BASE_URL="http://localhost:7860"

# ── Expose HF Spaces default port ─────────────────────────────────────────────
EXPOSE 7860

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start FastAPI server ───────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
