# intent-router-bench

Benchmarks intent classification models (accuracy + CPU latency) for a chat
router, with a React test UI. Built to pick the routing model for a larger
workflow system targeting CPU-only pods (AWS EKS).

## Deploy

- **Vercel (live demo):** import the repo at vercel.com/new — zero config. The
  FastAPI preset finds `app.py` (root) and routes everything to it; the slim
  roster (potion-8m + BYOK API models, no torch) comes from the root
  `requirements.txt`.
- **Docker / EKS:** `docker build -t intent-router .` — full roster, weights baked
  at build (`backend/preload.py`), serves on :7860. Same image works as a
  Hugging Face Docker Space, but HF now requires a PRO subscription for those.

## Setup (once)

CPU-only on purpose — the production target is CPU pods, so benchmark numbers
should come from CPU inference.

```powershell
# Windows
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch    # CPU wheel on Windows/Mac by default
.venv\Scripts\python -m pip install -r backend\requirements.txt
```

```bash
# Linux / Mac  (Linux default wheel bundles CUDA — use the CPU index to stay small)
python3.11 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r backend/requirements.txt
```

## Run

```powershell
.venv\Scripts\python backend\server.py                         # React UI at http://localhost:8000
.venv\Scripts\python backend\bench.py                          # benchmark all local models
.venv\Scripts\python backend\bench.py --device cpu --threads 2 # mirror a 2-vCPU pod (e.g. EKS)
.venv\Scripts\python backend\bench.py --models all-api         # needs GROQ_API_KEY / GEMINI_API_KEY
.venv\Scripts\python backend\test_bench.py                     # smoke check
```

(Linux/Mac: swap `.venv\Scripts\python` for `.venv/bin/python`.)

Models download from Hugging Face on first use (30–470MB each, cached in
`~/.cache/huggingface`). In the UI, the **Load all model weights** button
preloads everything up front — mirrors warm-at-boot on a prod server.

## Models

| model | what it is |
|-------|-----------|
| potion-8m | static embeddings (model2vec) + logistic regression — sub-ms CPU |
| bge-small / gte-small / e5-small | small transformer embeddings + logistic regression |
| groq-llama-3.1-8b, gemini-flash-lite | API LLMs, BYOK in the UI (or env keys); live model list fetchable per provider |

## Results (CPU, 2 threads — mirrors a 2-vCPU pod)

| model | custom acc | clinc acc | p50 | p95 | verdict |
|-------|-----------|-----------|-----|-----|---------|
| potion-8m | 0.84 | 0.89 | **0.2–0.8ms** | 1ms | fastest possible; accuracy floor acceptable |
| **bge-small** | **0.90** | **0.97** | **10–54ms** | 59ms | **recommended router** |
| gte-small | 0.92 | 0.95 | 36–38ms | 58ms | solid alternative |
| e5-small | 0.86 | 0.97 | 12–14ms | 22ms | fastest transformer; multilingual |
| qwen3-embed-0.6b † | 0.90 | 0.97 | 400–540ms | 1.2s | no accuracy gain over bge/e5, 10–40x slower |
| deberta-small-mnli † | 0.34 | 0.74 | 560–870ms | 1s | zero-shot too weak on custom intents |
| bart-large-mnli † | 0.32 | 0.73 | 1.3–2.1s | 13s | not viable on CPU |
| qwen3-0.6b (LLM) † | 0.22 | — | 892ms | 1.3s | small LLM can't hold the label format |
| arch-router-1.5b (LLM) † | 0.58 | — | 7.0s | 9.4s | purpose-built router, still loses hard on CPU |

† benchmarked, lost, and removed from the repo — numbers kept for the record.
clinc = a 15-intent subset of CLINC150 (needs the `datasets` package, also removed).

Takeaway: small embedders + logistic regression win outright — zero-shot NLI,
local LLMs, and bigger embedders add latency without accuracy on this task.

## Dataset

`backend/data/custom_intents.json` — 10 chat-router intents, 15 handwritten
utterances each; 10 train / 5 test per intent. Edit it to match your own
workflow intents, then rerun the bench.

## UI

React + shadcn-style components, served prebuilt by FastAPI — end users need only
Python (`python backend/server.py`, no node). To hack on the UI:

```bash
cd frontend
npm install
npm run dev        # Vite dev server on :5173, proxies /api to :8000
npm run build      # refresh the committed dist/ before pushing
```
