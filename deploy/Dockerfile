# Hugging Face Docker Space: runs the real FastAPI backend and the frontend it
# serves, so the deployed demo is the actual submission rather than a stand-in.
#
# Paths here are relative to the Space repo root. `deploy/push_space.py --stage`
# assembles exactly that tree locally, so a local `docker build` on the staged
# directory tests the same thing HF will build.

FROM python:3.11-slim

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WW_DEVICE=cpu

WORKDIR /home/user/app

# CPU-only torch. The default PyPI wheel bundles CUDA and is several GB, which
# is wasted bytes on a Space that has no GPU.
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user app/ ./app/
COPY --chown=user:user frontend/ ./frontend/
COPY --chown=user:user data/demo/ ./data/demo/

USER user

# Bake the classifier weights into an image layer. Downloading them at startup
# instead would add ~30s to every cold start, and free Spaces cold-start often.
# MODEL_ID comes from the Space variable of the same name; if it is unset the
# build still succeeds and the app falls back to CLIP zero-shot at runtime.
ARG MODEL_ID=""
ENV WW_MODEL_ID=${MODEL_ID}
RUN if [ -n "$MODEL_ID" ]; then \
        echo "prefetching $MODEL_ID" && \
        python -c "import os; from transformers import AutoImageProcessor, AutoModelForImageClassification; \
m = os.environ['WW_MODEL_ID']; AutoImageProcessor.from_pretrained(m); \
AutoModelForImageClassification.from_pretrained(m); print('cached', m)"; \
    else \
        echo "MODEL_ID unset - the Space will run on the CLIP zero-shot fallback"; \
    fi

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
