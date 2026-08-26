"""FastAPI backend serving the intent-router API + the built React UI.

Usage: python server.py   ->  http://localhost:8000
UI devs: cd frontend && npm install && npm run dev  (Vite proxies /api here)
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bench import API_MODELS, ROSTER, load_custom

HERE = Path(__file__).parent
app = FastAPI(title="Intent Router Test")

import os

LIGHT_MODELS = [k for k in ROSTER if k not in API_MODELS]
if os.environ.get("VERCEL"):
    LIGHT_MODELS = ["potion-8m"]  # slim serverless deploy; full tier runs locally / on EKS

DEMO_MESSAGES = [
    "hey, how's it going?",
    "my payment failed twice and I got charged anyway",
    "where's my package? it's been a week",
    "the app crashes every time I open settings",
    "I want to talk to a real human right now",
    "what's a good recipe for ramen?",
]

_cache = {}  # ponytail: unbounded, single-user test tool


def get_clf(key, api_key=None, provider_model=None):
    cache_k = (key, api_key, provider_model) if key in API_MODELS else key
    if cache_k not in _cache:
        train, _, labels = load_custom()
        clf = ROSTER[key]()
        if api_key:
            clf.api_key = api_key
        if provider_model:
            clf.model_id = provider_model  # live-picked model overrides the default
        clf.load()
        clf.train(train, labels)
        _cache[cache_k] = clf
    return _cache[cache_k]


@app.get("/api/models")
def models():
    return {"models": LIGHT_MODELS + sorted(API_MODELS), "demos": DEMO_MESSAGES}


@app.post("/api/preload")
def preload():
    """Load every light model up front — mirrors warm-at-boot on a prod server."""
    out = {}
    for k in LIGHT_MODELS:
        t0 = time.perf_counter()
        get_clf(k)
        out[k] = round(time.perf_counter() - t0, 2)
    return {"loaded": out}


class RouteReq(BaseModel):
    message: str
    model: str
    api_key: str | None = None  # BYOK for API models; never stored or logged
    provider_model: str | None = None  # e.g. a specific Groq/Gemini model id


class ProviderModelsReq(BaseModel):
    provider: str  # "groq" | "gemini"
    api_key: str | None = None


@app.post("/api/provider-models")
def provider_models(req: ProviderModelsReq):
    """Live model list from the provider, so the UI dropdown shows what's actually up."""
    try:
        if req.provider == "groq":
            from groq import Groq
            client = Groq(api_key=req.api_key) if req.api_key else Groq()
            skip = ("whisper", "tts", "guard", "embed")  # non-chat models can't classify
            ids = [m.id for m in client.models.list().data
                   if not any(s in m.id.lower() for s in skip)]
        elif req.provider == "gemini":
            from google import genai
            client = genai.Client(api_key=req.api_key) if req.api_key else genai.Client()
            ids = []
            for m in client.models.list():
                acts = (getattr(m, "supported_actions", None)
                        or getattr(m, "supported_generation_methods", None) or [])
                name = m.name.removeprefix("models/")
                if name.startswith("gemini") and (not acts or "generateContent" in acts):
                    ids.append(name)
        else:
            raise HTTPException(400, f"unknown provider {req.provider!r}")
    except HTTPException:
        raise
    except Exception as e:  # bad key, network — surface to UI
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"models": sorted(set(ids))}


@app.post("/api/route")
def route(req: RouteReq):
    if req.model not in ROSTER:
        raise HTTPException(400, f"unknown model {req.model!r}")
    try:
        t0 = time.perf_counter()
        clf = get_clf(req.model, req.api_key, req.provider_model)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        intent = clf.predict(req.message)
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:  # missing API key, OOM — surface to UI
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    # ponytail: stub — the real system dispatches to a workflow per intent here
    response = f"[{intent} workflow] handling: {req.message!r}"
    return {"intent": intent, "ms": ms, "load_s": load_s, "response": response}


dist = HERE.parent / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
