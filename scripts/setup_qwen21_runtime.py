"""Install Qwen-Image 2.1 libraries without changing Forge's environment.

The runtime inherits Forge's CUDA PyTorch through a .pth file. Newer generation
libraries are installed only into runtimes/qwen-image-2.1. Re-run to reproduce setup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DIFFUSERS_REVISION = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
DEPENDENCIES = [
    f"https://github.com/huggingface/diffusers/archive/{DIFFUSERS_REVISION}.zip",
    "transformers==5.17.0", "accelerate==1.15.0", "huggingface-hub==1.32.0",
    "safetensors==0.8.0", "tokenizers==0.23.2", "peft==0.21.0",
    "pillow==12.3.0", "httpx==0.28.1", "httpcore==1.0.9", "h11==0.16.0",
    "hf-xet==1.6.0", "click==8.5.0", "tomli==2.4.1",
    "kornia==0.8.2", "kornia-rs==0.1.14",
    "bitsandbytes==0.50.2",
]


def site_packages(python):
    """Query the interpreter instead of assuming a Windows venv layout."""
    result = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True, encoding="utf-8",
    )
    return Path(result.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--uv", default=shutil.which("uv"))
    args = parser.parse_args()
    if not args.uv:
        parser.error("uv is required to create the isolated runtime.")
    root = args.root.resolve()
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    source = root / "venv" / executable
    if not source.is_file():
        parser.error(f"Forge Python is missing: {source}")
    runtime = root / "runtimes" / "qwen-image-2.1"
    python = runtime / executable
    if not python.is_file():
        subprocess.run([args.uv, "venv", "--python", str(source), str(runtime)], check=True)
    site = site_packages(python)
    site.mkdir(parents=True, exist_ok=True)
    (site / "forge_torch.pth").write_text(str(site_packages(source)) + "\n", encoding="utf-8")
    # --no-deps prevents a second CUDA PyTorch download and prevents uv from
    # replacing Forge's inherited installation. Explicit dependencies override
    # the older Forge versions only within this interpreter.
    subprocess.run([
        args.uv, "pip", "install", "--python", str(python), "--no-deps", *DEPENDENCIES,
    ], check=True)
    checked = subprocess.run(
        [str(python), "-u", str(root / "modules_forge" / "qwen21_worker.py")],
        input='{"command":"status","request_id":"setup"}\n{"command":"shutdown"}\n',
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    events = [json.loads(line) for line in checked.stdout.splitlines() if line.strip()]
    errors = [event for event in events if event.get("event") == "error"]
    statuses = [event for event in events if event.get("event") == "status"]
    if errors or not statuses or not statuses[0].get("cuda_available") or statuses[0].get("dependency_errors"):
        sys.stderr.write(checked.stderr)
        raise RuntimeError(f"Qwen-Image 2.1 runtime check failed: {events}")
    print(json.dumps({"python": str(python), "status": statuses[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
