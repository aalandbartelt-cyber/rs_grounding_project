FROM python:3.12.16-slim-bookworm

WORKDIR /workspace
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY run.py /workspace/run.py
COPY weights/ /workspace/weights/
CMD ["python", "/workspace/run.py"]
