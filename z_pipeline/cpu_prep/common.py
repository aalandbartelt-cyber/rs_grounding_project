"""Small shared utilities; no models, GPU libraries or network calls."""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "z_evaluation"))
from evalkit import read, write, sha, number, integer, strict_json, iou

KEYWORDS = {
    "position_direction_flag": "左|右|上方|下方|中央|中心|边缘|顶部|底部|北|南|东|西",
    "relation_context_flag": "靠近|附近|旁边|之间|相邻|排列|一排|码头|跑道|道路|建筑|停车场",
    "color_brightness_flag": "白色|黑色|灰色|红色|蓝色|黄色|绿色|深色|浅色",
    "orientation_shape_flag": "朝向|方向|水平|垂直|长条|圆形|椭圆",
}


def memory_bytes():
    if sys.platform == "win32":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                "PeakPagefileUsage", "PrivateUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info = Counters()
        info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            raise OSError(ctypes.get_last_error(), "Cannot measure process memory")
        return {"rss": info.WorkingSetSize, "peak_rss": info.PeakWorkingSetSize}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    current = peak
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
    return {"rss": current, "peak_rss": peak}


class Resources:
    def __init__(self, limit_mb=256):
        self.start = time.perf_counter()
        self.limit = limit_mb * 1024 * 1024

    def check(self):
        result = memory_bytes()
        if result["rss"] > self.limit:
            raise MemoryError("RSS exceeded the checkpoint memory budget; stopping before the next item")
        return result

    def report(self):
        result = self.check()
        return {"seconds": round(time.perf_counter() - self.start, 3),
                "rss_mib": round(result["rss"] / 1024**2, 2),
                "peak_rss_mib": round(result["peak_rss"] / 1024**2, 2),
                "checkpoint_budget_mib": self.limit / 1024**2,
                "scope": "This Python process only; checkpoint guard, not an OS-enforced memory limit"}


def fresh_output(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Put generated reports and private data outside the code repository")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Choose a new nonempty-output-free directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Input path escapes its allowed root")
    return path


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4 and all(number(v) for v in box)
            and box[2] > 0 and box[3] > 0
            and all(math.isfinite(v) for v in (box[0] + box[2], box[1] + box[3], box[2] * box[3])))


def side_bin(box):
    side = min(box[2:])
    return "short_lt16px" if side < 16 else "short_16_to_lt32px" if side < 32 else "short_ge32px"
