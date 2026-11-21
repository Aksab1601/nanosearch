# Multi-stage: build wheels where a compiler may be needed, then copy only the
# installed packages into a slim runtime image. The result ships no build
# toolchain, which keeps the image small and the attack surface smaller.
FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY nanosearch ./nanosearch

RUN pip install --prefix=/install ".[service]"


FROM python:3.12-slim AS runtime

# Run as a non-root user: a container process that does not need root should
# not have it.
RUN useradd --create-home --uid 10001 nanosearch

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=nanosearch:nanosearch nanosearch ./nanosearch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Serve the bundled demo corpus unless an index directory is mounted and
    # NANOSEARCH_INDEX points at it.
    NANOSEARCH_PORT=8000

USER nanosearch
EXPOSE 8000

# Hitting the real readiness endpoint, so an unhealthy container is reported as
# unhealthy rather than merely "running".
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()"

CMD ["python", "-m", "uvicorn", "nanosearch.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
