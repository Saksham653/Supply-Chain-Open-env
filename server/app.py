"""
Validator-compatible server entry point.

This wraps the project FastAPI app so the repository can be run via:
    uv run --project . server
or:
    python -m server.app
"""

from app import app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.port == 7860:
        main()
    else:
        main(port=args.port)
