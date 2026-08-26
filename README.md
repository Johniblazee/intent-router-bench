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

## Datasets

- `custom` — 10 chat-router intents, 15 handwritten utterances each (data/custom_intents.json), 10 train / 5 test per intent
- `clinc` — 15-intent subset of CLINC150 (`clinc/clinc_oos`), 10 train / 10 test per intent
