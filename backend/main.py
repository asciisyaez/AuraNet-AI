"""
Thin entrypoint that aliases to the unified wall detection app in main_v2.
This keeps a single backend surface while maintaining compatibility with
`uvicorn backend.main:app`.
"""
from backend.main_v2 import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
