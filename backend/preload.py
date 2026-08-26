"""Download all local model weights — run at Docker build to bake them into the image."""
from bench import API_MODELS, ROSTER

for k in ROSTER:
    if k not in API_MODELS:
        print("loading", k, flush=True)
        ROSTER[k]().load()
print("all weights cached")
