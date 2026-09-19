FROM python:3.11-slim AS wheels

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON" \
    CMAKE_BUILD_PARALLEL_LEVEL=2

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# Basic mode needs torch and optimum, not just onnxruntime: the encoder is served
# through optimum.onnxruntime and returns torch tensors. Without them the container
# starts and serves the interface but can never produce a score.
#
# Advanced mode is intentionally absent: its llama-cpp-python wheel is published for
# Windows only.
RUN pip wheel --wheel-dir /wheels . \
    && pip wheel --wheel-dir /wheels \
       "uvicorn[standard]>=0.30" \
       "torch>=2.4" \
       "transformers>=4.51" \
       "optimum[onnxruntime]>=1.20" \
       "ml-dtypes>=0.5"

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORIDJIDJI_PROFILE=lite \
    ORIDJIDJI_CONFIG_DIR=/app/configs

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

WORKDIR /app
COPY configs ./configs
RUN useradd --create-home --uid 10001 oridjidji \
    && mkdir -p /models/data/lexicons /models/history \
    && chown -R oridjidji:oridjidji /models
USER oridjidji

WORKDIR /models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"

CMD ["sh", "-c", "exec oridjidji serve --host 0.0.0.0 --port 8000 --no-warmup --profile \"$ORIDJIDJI_PROFILE\""]
