"""Vercel entry: ASGI app for /api/* (rewritten here by vercel.json)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("HF_HOME", "/tmp/hf")  # only /tmp is writable on Vercel

from server import app  # noqa: E402,F401
