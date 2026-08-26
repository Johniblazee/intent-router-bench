"""Vercel entrypoint — FastAPI preset loads `app` from here and routes all requests to it."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
os.environ.setdefault("HF_HOME", "/tmp/hf")  # only /tmp is writable on Vercel

from server import app  # noqa: E402,F401
