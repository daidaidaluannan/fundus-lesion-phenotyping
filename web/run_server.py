#!/usr/bin/env python3
"""
Convenience launcher for the FastAPI app with explicit model paths.

Usage:
  python web/run_server.py \
    --aae-c /path/to/aae_c.pth \
    --aot-gan /path/to/aotgan_netG.pth \
    --aae-s /path/to/aae_s_multimap.pth \
    --device cuda \
    --host 0.0.0.0 --port 8075 --reload
"""
import argparse
import uvicorn
from pathlib import Path
import json
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
from web import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Launch PHANES web demo with model paths.")
    p.add_argument("--aae-c", required=True, help="Path to AAE_C checkpoint.")
    p.add_argument("--aot-gan", required=True, help="Path to AOT-GAN netG checkpoint.")
    p.add_argument("--aae-s", required=True, help="Path to AAE_S (multimap) checkpoint.")
    p.add_argument("--device", default="cuda", help="Device for inference (cuda/cpu).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8075)
    p.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    p.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical",
                   help="Front-end layout style.")
    return p.parse_args()


def main():
    args = parse_args()
    # Set global config before importing app
    config.MODEL_PATHS.update({
        "aae_c": str(Path(args.aae_c).expanduser()),
        "aot_gan": str(Path(args.aot_gan).expanduser()),
        "aae_s": str(Path(args.aae_s).expanduser()),
        "device": args.device,
    })
    config.UI_SETTINGS.update({
        "layout": args.layout,
    })
    # Persist to file for reload worker
    cfg_file = Path(__file__).parent / "model_paths.json"
    cfg_file.write_text(json.dumps({
        "paths": config.MODEL_PATHS,
        "ui": config.UI_SETTINGS,
    }), encoding="utf-8")
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
