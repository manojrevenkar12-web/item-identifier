# CPU serving image. Mount a trained checkpoint (and optional ONNX export) at /models.
#   docker build -t itemid .
#   docker run -p 8000:8000 -v $PWD/runs/cars_dinov2_s:/models itemid
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,export]"

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app

ENV ITEMID_MODEL=/models/model.pt \
    ITEMID_DB=/data/feedback.sqlite \
    OMP_NUM_THREADS=2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/readyz').status != 200)"
CMD ["uvicorn", "itemid.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
