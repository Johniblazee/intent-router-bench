# HF Space (docker) + EKS image base. CPU-only, weights baked at build for fast cold starts.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user backend/ backend/
COPY --chown=user frontend/dist/ frontend/dist/

RUN python backend/preload.py

CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
