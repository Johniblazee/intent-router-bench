"""Gradio UI: type a chat message, see which workflow each model routes it to.

Usage: python ui.py   ->  http://127.0.0.1:7860
"""

import time

import gradio as gr

from bench import API_MODELS, DATASETS, ROSTER
from router_graph import DEMO_MESSAGES, build_graph

_cache = {}  # model key -> compiled router graph; ponytail: grows unbounded, fine for a test UI


def _get_graph(key):
    if key not in _cache:
        train, _, labels = DATASETS["custom"]()
        clf = ROSTER[key]()
        clf.load()
        clf.train(train, labels)
        _cache[key] = build_graph(clf, labels)
    return _cache[key]


def route(message, model_key):
    if not message.strip():
        return "", "", ""
    t0 = time.perf_counter()
    graph = _get_graph(model_key)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = graph.invoke({"message": message})
    ms = (time.perf_counter() - t0) * 1000
    note = f"{ms:.1f} ms" + (f"  (first call: +{load_s:.1f}s model load)" if load_s > 1 else "")
    return out["intent"], note, out["response"]


local = [k for k in ROSTER if k not in API_MODELS]
api = sorted(API_MODELS)

with gr.Blocks(title="Intent Router Test") as demo:
    gr.Markdown("# Intent Router Test\nPick a model, type a message, see the route. "
                "First call per model downloads/loads it — later calls show true latency.")
    with gr.Row():
        model = gr.Dropdown(local + api, value="bge-small", label="Model (API ones need env keys)")
    msg = gr.Textbox(label="User message", placeholder="my payment failed twice and I got charged anyway")
    btn = gr.Button("Route", variant="primary")
    with gr.Row():
        intent = gr.Textbox(label="Intent", interactive=False)
        latency = gr.Textbox(label="Latency", interactive=False)
    response = gr.Textbox(label="Workflow response (stub)", interactive=False)
    gr.Examples([[m] for m in DEMO_MESSAGES], inputs=[msg])
    btn.click(route, [msg, model], [intent, latency, response])
    msg.submit(route, [msg, model], [intent, latency, response])

if __name__ == "__main__":
    demo.launch()
