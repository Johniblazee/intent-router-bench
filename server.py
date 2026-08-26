"""FastAPI backend serving the intent-router API + the built React UI.

Usage: python server.py   ->  http://localhost:8000
UI devs: cd frontend && npm install && npm run dev  (Vite proxies /api here)
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bench import API_MODELS, DATASETS, ROSTER
from router_graph import DEMO_MESSAGES, build_graph

HERE = Path(__file__).parent
app = FastAPI(title="Intent Router Test")

_cache = {}  # ponytail: unbounded, single-user test tool


def get_graph(key):
    if key not in _cache:
        train, _, labels = DATASETS["custom"]()
        clf = ROSTER[key]()
        clf.load()
        clf.train(train, labels)
        _cache[key] = build_graph(clf, labels)
    return _cache[key]


@app.get("/api/models")
def models():
    local = [k for k in ROSTER if k not in API_MODELS]
    return {"models": local + sorted(API_MODELS), "demos": DEMO_MESSAGES}


class RouteReq(BaseModel):
    message: str
    model: str


@app.post("/api/route")
def route(req: RouteReq):
    if req.model not in ROSTER:
        raise HTTPException(400, f"unknown model {req.model!r}")
    try:
        t0 = time.perf_counter()
        graph = get_graph(req.model)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = graph.invoke({"message": req.message})
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:  # missing API key, gated model, OOM — surface to UI
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"intent": out["intent"], "ms": ms, "load_s": load_s, "response": out["response"]}


dist = HERE / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
