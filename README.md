# classification-task — intent-router model benchmark

Benchmarks intent classification models (accuracy + latency) for a chat router,
plus a minimal LangGraph router demo. Self-contained: own venv, no deps on
sibling folders.

## Setup (once)

```powershell
# Windows
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
# no NVIDIA GPU: .venv\Scripts\python -m pip install torch   (CPU — LLM tier will be slow)
.venv\Scripts\python -m pip install -r requirements.txt
```

```bash
# Linux / Mac
python3.11 -m venv .venv
.venv/bin/pip install torch
.venv/bin/pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python ui.py                             # Gradio UI at http://127.0.0.1:7860
.venv\Scripts\python bench.py                          # all local models, both datasets
.venv\Scripts\python bench.py --models potion-8m,bge-small --datasets custom
.venv\Scripts\python bench.py --models all-api         # needs GROQ_API_KEY / GEMINI_API_KEY
.venv\Scripts\python router_graph.py --model bge-small # LangGraph router demo (CLI)
.venv\Scripts\python test_bench.py                     # smoke check
```

(Linux/Mac: swap `.venv\Scripts\python` for `.venv/bin/python`.)

Models download from Hugging Face on first use (~30MB–3GB per model, cached in `~/.cache/huggingface`).

Results append to `results.csv`; a markdown table prints at the end.

## Models

| tier | models | notes |
|------|--------|-------|
| static embedding | potion-8m (model2vec) | sub-ms CPU |
| embedding + LR | bge-small, gte-small, e5-small, qwen3-embed-0.6b | production sweet spot |
| zero-shot NLI | deberta-small-mnli, bart-large-mnli | no training data needed |
| local LLM | qwen3-0.6b, gemma-3-270m, arch-router-1.5b | gemma is gated on HF (needs `hf auth login` + license) |
| API | groq-llama-3.1-8b, gemini-flash-lite | skipped unless env key set |

## Datasets

- `custom` — 10 chat-router intents, 15 handwritten utterances each (data/custom_intents.json), 10 train / 5 test per intent
- `clinc` — 15-intent subset of CLINC150 (`clinc/clinc_oos`), 10 train / 10 test per intent
