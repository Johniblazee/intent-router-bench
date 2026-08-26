# classification-task — intent-router model benchmark

Benchmarks intent classification models (accuracy + latency) for a chat router,
plus a minimal LangGraph router demo. Self-contained: own venv, no deps on
sibling folders.

## Setup (once)

CPU-only on purpose — the production target is CPU pods (EKS), so benchmark numbers
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
.venv\Scripts\python backend\bench.py                          # all local models, both datasets
.venv\Scripts\python backend\bench.py --models potion-8m,bge-small --datasets custom
.venv\Scripts\python backend\bench.py --models all-api         # needs GROQ_API_KEY / GEMINI_API_KEY
.venv\Scripts\python backend\bench.py --device cpu --threads 2 # mirror a 2-vCPU CPU pod (e.g. EKS)
.venv\Scripts\python backend\router_graph.py --model bge-small # LangGraph router demo (CLI)
.venv\Scripts\python backend\test_bench.py                     # smoke check
```

(Linux/Mac: swap `.venv\Scripts\python` for `.venv/bin/python`.)

Models download from Hugging Face on first use (~30MB–3GB per model, cached in `~/.cache/huggingface`).

## UI

React + shadcn-style components, served prebuilt by FastAPI — end users need only
Python (`python server.py`, no node). To hack on the UI:

```bash
cd frontend
npm install
npm run dev        # Vite dev server on :5173, proxies /api to :8000
npm run build      # refresh the committed dist/ before pushing
```

Results append to `results.csv`; a markdown table prints at the end.

## Models

| tier | models | notes |
|------|--------|-------|
| static embedding | potion-8m (model2vec) | sub-ms CPU |
| embedding + LR | bge-small, gte-small, e5-small, qwen3-embed-0.6b | production sweet spot |
| zero-shot NLI | deberta-small-mnli, bart-large-mnli | no training data needed |
| local LLM | qwen3-0.6b, gemma-3-270m, arch-router-1.5b | slow on CPU by design — that's the datapoint; gemma is gated on HF (needs `hf auth login` + license) |
| API | groq-llama-3.1-8b, gemini-flash-lite | skipped unless env key set |

The UI serves only the CPU-viable light tier; heavy models (qwen3-embed-0.6b,
bart-large-mnli, local LLMs) stay benchable via the CLI.

## Results (CPU, 2 threads — mirrors a 2-vCPU pod)

| model | custom acc | clinc acc | p50 | p95 | verdict |
|-------|-----------|-----------|-----|-----|---------|
| potion-8m | 0.84 | 0.89 | **0.2–0.8ms** | 1ms | fastest possible; accuracy floor acceptable |
| **bge-small** | **0.90** | **0.97** | **10–54ms** | 59ms | **recommended router** |
| gte-small | 0.92 | 0.95 | 36–38ms | 58ms | solid alternative |
| e5-small | 0.86 | 0.97 | 12–14ms | 22ms | fastest transformer; multilingual |
| qwen3-embed-0.6b | 0.90 | 0.97 | 400–540ms | 1.2s | no accuracy gain over bge/e5, 10–40x slower |
| deberta-small-mnli | 0.34 | 0.74 | 560–870ms | 1s | zero-shot too weak on custom intents |
| bart-large-mnli | 0.32 | 0.73 | 1.3–2.1s | 13s | not viable on CPU |
| qwen3-0.6b (LLM) | 0.22 | — | 892ms | 1.3s | small LLM can't hold the label format |
| arch-router-1.5b (LLM) | 0.58 | — | 7.0s | 9.4s | purpose-built router, still loses hard on CPU |

Takeaway: small embedders + logistic regression win outright — zero-shot NLI and
bigger models add latency without accuracy on this task.

## Datasets

- `custom` — 10 chat-router intents, 15 handwritten utterances each (data/custom_intents.json), 10 train / 5 test per intent
- `clinc` — 15-intent subset of CLINC150 (`clinc/clinc_oos`), 10 train / 10 test per intent
