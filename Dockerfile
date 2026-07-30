FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so code edits do not invalidate the pip cache.
COPY pyproject.toml README.md ./
COPY fin_rag ./fin_rag
RUN pip install --no-cache-dir .

COPY data ./data
COPY scripts ./scripts

# Run as non-root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# NOTE: no authentication is configured. This service is meant to sit behind an
# API gateway that handles auth/rate limiting — do not expose it directly.
CMD ["uvicorn", "fin_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
