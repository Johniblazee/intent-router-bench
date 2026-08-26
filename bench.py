"""Benchmark intent classifiers for chat routing: accuracy + latency.

Self-contained. Usage:
    python bench.py                        # all local models, both datasets
    python bench.py --models potion-8m,bge-small --datasets custom
    python bench.py --models all-api       # groq/gemini (needs env keys)
Results -> results.csv + markdown table on stdout.
"""

import argparse
import difflib
import gc
import json
import os
import statistics
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

# ---------------------------------------------------------------- datasets

def load_custom():
    d = json.loads((DATA / "custom_intents.json").read_text(encoding="utf-8"))
    train, test = [], []
    for intent, utts in d.items():
        train += [(u, intent) for u in utts[:-5]]  # 10 train / 5 test per intent
        test += [(u, intent) for u in utts[-5:]]
    return train, test, sorted(d)

# ponytail: fixed 15-intent subset, full 150 makes zero-shot/LLM runs crawl
CLINC_KEEP = [
    "balance", "bill_due", "pay_bill", "transfer", "freeze_account",
    "order_status", "greeting", "weather", "translate", "flight_status",
    "restaurant_reservation", "book_flight", "book_hotel", "goodbye", "thank_you",
]

def load_clinc(train_per_intent=10, test_per_intent=10):
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    missing = [k for k in CLINC_KEEP if k not in names]
    assert not missing, f"CLINC intents not found: {missing}"
    keep = set(CLINC_KEEP)

    def pick(split, per_intent):
        out, seen = [], {}
        for row in ds[split]:
            intent = names[row["intent"]]
            if intent in keep and seen.get(intent, 0) < per_intent:
                out.append((row["text"], intent))
                seen[intent] = seen.get(intent, 0) + 1
        return out

    return pick("train", train_per_intent), pick("test", test_per_intent), sorted(keep)

DATASETS = {"custom": load_custom, "clinc": load_clinc}

# ---------------------------------------------------------------- classifiers

def _device():
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


class ZeroShot:
    """NLI zero-shot pipeline; no training data used."""

    def __init__(self, name, model_id):
        self.name, self.model_id = name, model_id

    def load(self):
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model=self.model_id,
                             device=0 if _device() == "cuda" else -1)

    def train(self, pairs, labels):
        self.labels = labels
        self.natural = {l.replace("_", " "): l for l in labels}

    def predict(self, text):
        out = self.pipe(text, candidate_labels=list(self.natural),
                        hypothesis_template="The user's intent is {}.")
        return self.natural[out["labels"][0]]


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


class LocalLLM:
    """Small causal LM prompted as a classifier."""

    def __init__(self, name, model_id):
        self.name, self.model_id = name, model_id

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=_device())

    def train(self, pairs, labels):
        self.labels = labels

    def _chat(self, messages, max_new_tokens=16):
        import torch
        kw = {}
        if "qwen3" in self.model_id.lower():
            kw["enable_thinking"] = False
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", **kw).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def predict(self, text):
        prompt = LLM_PROMPT.format(labels=", ".join(self.labels), text=text)
        return _match_label(self._chat([{"role": "user", "content": prompt}]), self.labels)


ARCH_PROMPT = """You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>
{routes}
</routes>

<conversation>
user: {text}
</conversation>

Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags. Follow the instruction:
1. Analyze the route descriptions and find the best match route for user latest intent.
2. Respond only with the route name that best matches the user's request, using the exact name from <routes></routes>.

Based on your analysis, provide your response in the following JSON format:
{{"route": "route_name"}}"""


class ArchRouter(LocalLLM):
    """katanemo/Arch-Router-1.5B — purpose-built routing model, JSON route output."""

    def __init__(self):
        super().__init__("arch-router-1.5b", "katanemo/Arch-Router-1.5B")

    def train(self, pairs, labels):
        self.labels = labels
        routes = [{"name": l, "description": l.replace("_", " ")} for l in labels]
        self.routes_json = json.dumps(routes)

    def predict(self, text):
        prompt = ARCH_PROMPT.format(routes=self.routes_json, text=text)
        raw = self._chat([{"role": "user", "content": prompt}], max_new_tokens=32)
        try:
            return _match_label(json.loads(raw.strip())["route"], self.labels)
        except Exception:
            return _match_label(raw, self.labels)


class GroqLLM:
    def __init__(self, name="groq-llama-3.1-8b", model_id="llama-3.1-8b-instant"):
        self.name, self.model_id = name, model_id

    def load(self):
        from groq import Groq
        self.client = Groq()  # GROQ_API_KEY

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

    def load(self):
        from google import genai
        self.client = genai.Client()  # GEMINI_API_KEY

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
    # static embeddings — sub-ms CPU floor
    "potion-8m": lambda: EmbedLR("potion-8m", "minishlab/potion-base-8M", static=True),
    # embedding + LR head — production sweet spot
    "bge-small": lambda: EmbedLR("bge-small", "BAAI/bge-small-en-v1.5"),
    "gte-small": lambda: EmbedLR("gte-small", "thenlper/gte-small"),
    "e5-small": lambda: EmbedLR("e5-small", "intfloat/multilingual-e5-small", prefix="query: "),
    "qwen3-embed-0.6b": lambda: EmbedLR("qwen3-embed-0.6b", "Qwen/Qwen3-Embedding-0.6B"),
    # zero-shot NLI — no training data
    "deberta-small-mnli": lambda: ZeroShot("deberta-small-mnli", "cross-encoder/nli-deberta-v3-small"),
    "bart-large-mnli": lambda: ZeroShot("bart-large-mnli", "facebook/bart-large-mnli"),
    # small local LLMs
    "qwen3-0.6b": lambda: LocalLLM("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    "gemma-3-270m": lambda: LocalLLM("gemma-3-270m", "google/gemma-3-270m-it"),
    "arch-router-1.5b": ArchRouter,
    # APIs — secondary, network latency included
    "groq-llama-3.1-8b": GroqLLM,
    "gemini-flash-lite": GeminiLLM,
}
API_MODELS = {"groq-llama-3.1-8b", "gemini-flash-lite"}

# ---------------------------------------------------------------- harness

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
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all-local",
                    help="comma list, 'all', 'all-local', or 'all-api'")
    ap.add_argument("--datasets", default="both", help="custom, clinc, or both")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    args = ap.parse_args()

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

    ds_names = ["custom", "clinc"] if args.datasets == "both" else [args.datasets]

    rows = []
    for ds_name in ds_names:
        train, test, labels = DATASETS[ds_name]()
        print(f"\n== dataset {ds_name}: {len(train)} train / {len(test)} test / {len(labels)} intents")
        for key in keys:
            if key in API_MODELS:
                env = "GROQ_API_KEY" if "groq" in key else "GEMINI_API_KEY"
                if not os.environ.get(env):
                    print(f"  skip {key}: {env} not set")
                    continue
            print(f"  running {key} ...", flush=True)
            try:
                row = bench_one(key, train, test, labels, ds_name)
                rows.append(row)
                print(f"    acc={row['accuracy']} f1={row['macro_f1']} "
                      f"p50={row['p50_ms']}ms p95={row['p95_ms']}ms")
            except Exception as e:  # gated model, OOM, network — keep going
                print(f"    FAILED {key}: {type(e).__name__}: {e}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(args.out)
    header = not out.exists()
    df.to_csv(out, mode="a", header=header, index=False)
    print(f"\nappended to {out}\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
