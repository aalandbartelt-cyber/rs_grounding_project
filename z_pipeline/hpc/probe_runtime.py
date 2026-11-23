"""Read-only environment checks plus a tiny CUDA calculation; no downloads/training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def probe():
    result = {"utc_time": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": sys.version, "executable": sys.executable, "cwd": str(Path.cwd()),
              "home": str(Path.home()), "packages": {}, "cuda_smoke_passed": False}
    disk = shutil.disk_usage(Path.home())
    result["home_disk_bytes"] = {"total": disk.total, "used": disk.used, "free": disk.free}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow", "huggingface-hub"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        result["nvidia_smi"] = {"returncode": process.returncode, "stdout": process.stdout.strip(),
                                "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi"] = {"error": str(exc)}
    try:
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["visible_cuda_devices"] = torch.cuda.device_count()
        if result["cuda_available"]:
            result["cuda_devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())]
            with torch.inference_mode():
                value = torch.ones((32, 32), device="cuda")
                average = (value @ value).mean().item()
                result["cuda_matmul_mean"] = average
                result["cuda_smoke_passed"] = average == 32.0
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    result["next_step"] = ("Check dataset export and baseline dependencies" if result["cuda_smoke_passed"]
                           else "Inspect assigned GPU and image Python environment before installing anything")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new report filename")
    result = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if result["cuda_smoke_passed"] else 2)
