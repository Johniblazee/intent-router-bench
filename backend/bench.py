"""Benchmark intent classifiers for chat routing: accuracy + latency.

Self-contained. Usage:
    python bench.py                        # all local models
    python bench.py --models potion-8m,bge-small
    python bench.py --models all-api       # groq/gemini (needs env keys)
    python bench.py --device cpu --threads 2   # mirror a 2-vCPU pod (e.g. EKS)
Results -> results.csv + table on stdout.

Stripped to the light tier that won the bake-off (see README results table for
zero-shot / local-LLM / heavy-embedder numbers — they lost on CPU and were removed).
"""

import argparse
import csv
import difflib
import gc
import json
import os
import statistics
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

# ---------------------------------------------------------------- dataset

def load_custom():
    d = json.loads((DATA / "custom_intents.json").read_text(encoding="utf-8"))
    train, test = [], []
    for intent, utts in d.items():
        train += [(u, intent) for u in utts[:-5]]  # 10 train / 5 test per intent
        test += [(u, intent) for u in utts[-5:]]
    return train, test, sorted(d)

DATASETS = {"custom": load_custom}

# ---------------------------------------------------------------- classifiers

DEVICE = "auto"  # set from --device; "cpu" mirrors CPU-only targets like EKS pods


def _device():
    if DEVICE != "auto":
        return DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbedLR:
    """Any embedding model + logistic-regression head."""

    def __init__(self, name, model_id, prefix="", static=False):
        self.name, self.model_id, self.prefix, self.static = name, model_id, prefix, static

    def load(self):
        if self.static:
            from model2vec import StaticModel
            self.model = StaticModel.from_pretrained(self.model_id)
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_id, device=_device())

    def _encode(self, texts):
        return self.model.encode([self.prefix + t for t in texts], show_progress_bar=False)

    def train(self, pairs, labels):
        from sklearn.linear_model import LogisticRegression
        texts, y = zip(*pairs)
        self.lr = LogisticRegression(max_iter=2000).fit(self._encode(list(texts)), y)

    def predict(self, text):
        return self.lr.predict(self._encode([text]))[0]


LLM_PROMPT = (
    "Classify the user message into exactly one intent.\n"
    "Intents: {labels}\n"
    "User message: {text}\n"
    "Reply with only the intent name, nothing else."
)

def _match_label(raw, labels):
    raw = raw.strip().lower()
    for l in labels:
        if l in raw:
            return l
    close = difflib.get_close_matches(raw, labels, n=1, cutoff=0.0)
    return close[0]


class GroqLLM:
    def __init__(self, name="groq-llama-3.1-8b", model_id="llama-3.1-8b-instant"):
        self.name, self.model_id = name, model_id
        self.api_key = None  # BYOK; falls back to GROQ_API_KEY env

    def load(self):
        from groq import Groq
        self.client = Groq(api_key=self.api_key) if self.api_key else Groq()

    def train(self, pairs, labels):
        self.labels = labels

    def predict(self, text):
        prompt = LLM_PROMPT.format(labels=", ".join(self.labels), text=text)
        out = self.client.chat.completions.create(
            model=self.model_id, max_tokens=16, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        return _match_label(out.choices[0].message.content, self.labels)


class GeminiLLM:
    def __init__(self, name="gemini-flash-lite", model_id="gemini-2.5-flash-lite"):
        self.name, self.model_id = name, model_id
        self.api_key = None  # BYOK; falls back to GEMINI_API_KEY env

    def load(self):
        from google import genai
        self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()

    def train(self, pairs, labels):
        self.labels = labels

    def predict(self, text):
        from google.genai import types
        prompt = LLM_PROMPT.format(labels=", ".join(self.labels), text=text)
        out = self.client.models.generate_content(
            model=self.model_id, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=16, temperature=0,
                                               thinking_config=types.ThinkingConfig(thinking_budget=0)))
        return _match_label(out.text or "", self.labels)


ROSTER = {
    "potion-8m": lambda: EmbedLR("potion-8m", "minishlab/potion-base-8M", static=True),
    "bge-small": lambda: EmbedLR("bge-small", "BAAI/bge-small-en-v1.5"),
    "gte-small": lambda: EmbedLR("gte-small", "thenlper/gte-small"),
    "e5-small": lambda: EmbedLR("e5-small", "intfloat/multilingual-e5-small", prefix="query: "),
    "groq-llama-3.1-8b": GroqLLM,
    "gemini-flash-lite": GeminiLLM,
}
API_MODELS = {"groq-llama-3.1-8b", "gemini-flash-lite"}

# ---------------------------------------------------------------- harness

FIELDS = ["model", "dataset", "device", "accuracy", "macro_f1",
          "p50_ms", "p95_ms", "load_s", "train_s", "n_test"]


def bench_one(key, train, test, labels, dataset_name):
    clf = ROSTER[key]()
    t0 = time.perf_counter()
    clf.load()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf.train(train, labels)
    train_s = time.perf_counter() - t0

    for text, _ in test[:3]:  # warm-up
        clf.predict(text)

    lat, preds, golds = [], [], []
    for text, gold in test:
        t0 = time.perf_counter()
        pred = clf.predict(text)
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
        golds.append(gold)

    from sklearn.metrics import accuracy_score, f1_score
    row = {
        "model": clf.name,
        "dataset": dataset_name,
        "device": _device(),
        "accuracy": round(accuracy_score(golds, preds), 3),
        "macro_f1": round(f1_score(golds, preds, average="macro"), 3),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1),
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 2),
        "n_test": len(test),
    }

    del clf
    gc.collect()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all-local",
                    help="comma list, 'all', 'all-local', or 'all-api'")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="cpu = mirror CPU-only prod targets (e.g. EKS pods)")
    ap.add_argument("--threads", type=int, default=0,
                    help="cap torch CPU threads, emulates pod vCPU limit (e.g. 2)")
    args = ap.parse_args()

    global DEVICE
    DEVICE = args.device
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    if args.models == "all":
        keys = list(ROSTER)
    elif args.models == "all-local":
        keys = [k for k in ROSTER if k not in API_MODELS]
    elif args.models == "all-api":
        keys = sorted(API_MODELS)
    else:
        keys = [k.strip() for k in args.models.split(",")]
    unknown = [k for k in keys if k not in ROSTER]
    assert not unknown, f"unknown models: {unknown}; choose from {list(ROSTER)}"

    train, test, labels = load_custom()
    print(f"dataset custom: {len(train)} train / {len(test)} test / {len(labels)} intents")

    rows = []
    for key in keys:
        if key in API_MODELS:
            env = "GROQ_API_KEY" if "groq" in key else "GEMINI_API_KEY"
            if not os.environ.get(env):
                print(f"  skip {key}: {env} not set")
                continue
        print(f"  running {key} ...", flush=True)
        try:
            row = bench_one(key, train, test, labels, "custom")
            rows.append(row)
            print(f"    acc={row['accuracy']} f1={row['macro_f1']} "
                  f"p50={row['p50_ms']}ms p95={row['p95_ms']}ms")
        except Exception as e:  # missing key, OOM, network — keep going
            print(f"    FAILED {key}: {type(e).__name__}: {e}")

    out = Path(args.out)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended to {out}\n")
    for r in rows:
        print("  ".join(f"{k}={r[k]}" for k in FIELDS))


if __name__ == "__main__":
    main()
