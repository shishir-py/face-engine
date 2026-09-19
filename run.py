#!/usr/bin/env python3
"""Run the FaceMatch AI server."""

import uvicorn
from app.config import load_config


def main():
    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
