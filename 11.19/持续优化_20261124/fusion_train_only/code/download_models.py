"""Download pinned official model files, excluding all datasets and examples."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_HOST = os.environ.get("FUSION_MODEL_DOWNLOAD_HOST", "huggingface.co")
if DOWNLOAD_HOST not in ("huggingface.co", "hf-mirror.com", "www.modelscope.cn"):
    raise ValueError("Unrecognized model download transport")


def transport_url(url):
    if DOWNLOAD_HOST == "www.modelscope.cn":
        prefix = "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/"
        if not url.startswith(prefix):
            raise ValueError("ModelScope continuation is only configured for the official Qwen repository")
        filename = url[len(prefix):].split("/", 1)[1]
        return "https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/resolve/master/" + filename
    return url.replace("https://huggingface.co/", f"https://{DOWNLOAD_HOST}/", 1)
MODELS = {
    "RemoteCLIP": ("chendelong/RemoteCLIP", ("RemoteCLIP-ViT-L-14.pt",)),
    "GeoRSCLIP": ("Zilun/GeoRSCLIP", ("ckpt/RS5M_ViT-L-14-336.pt", "README.md")),
    "Qwen3-VL-4B": ("Qwen/Qwen3-VL-4B-Instruct", (
        "*.json", "*.txt", "*.safetensors", "README.md")),
    "GeoGround": ("erenzhou/GeoGround", (
        "llava-v1.5-7b-task-lora-geoground/*", "inference_scripts/inference_hbb.py", "README.md")),
}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Codex-model-setup"}), timeout=45) as response:
        return json.load(response)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ranged_download(entry, partial):
    state_path = partial.with_suffix(".ranges.json")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"origin": partial.stat().st_size if partial.exists() else 0,
                 "chunk_bytes": 32 * 1024 * 1024, "completed": []}
        state_path.write_text(json.dumps(state), encoding="utf-8")
    start, chunk_size = state["origin"], state["chunk_bytes"]
    with partial.open("ab") as stream:
        stream.truncate(entry["bytes"])
    lock = threading.Lock()

    def chunk(number):
        low = start + number * chunk_size
        high = min(entry["bytes"] - 1, low + chunk_size - 1)
        for attempt in range(8):
            try:
                headers = {"User-Agent": "Codex-model-setup", "Range": f"bytes={low}-{high}"}
                url = transport_url(entry["url"])
                if DOWNLOAD_HOST == "huggingface.co":
                    url += "?download=true&segment=" + str(low) + "&t=" + str(int(time.time()))
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                    if response.status != 206:
                        raise OSError("Expected a ranged model response")
                    if response.headers.get("Content-Range") != f"bytes {low}-{high}/{entry['bytes']}":
                        raise OSError("Server returned a different model byte range")
                    payload = response.read()
                if len(payload) != high - low + 1:
                    raise OSError("Incomplete model segment")
                with lock:
                    with partial.open("r+b") as stream:
                        stream.seek(low)
                        stream.write(payload)
                    state["completed"].append(number)
                    pending = state_path.with_suffix(".tmp")
                    pending.write_text(json.dumps(state), encoding="utf-8")
                    pending.replace(state_path)
                return
            except Exception as error:
                if attempt == 7:
                    raise
                print(f"Retry segment {entry['folder']}/{number}: {error}", flush=True)
                time.sleep(min(attempt + 1, 6))
    total = (entry["bytes"] - start + chunk_size - 1) // chunk_size
    remaining = [i for i in range(total) if i not in state["completed"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(chunk, remaining):
            pass
    return state_path


def fetch(entry):
    destination = ROOT / "weights" / entry["folder"] / entry["file"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == entry["bytes"]:
        actual = sha256(destination)
        if not entry["expected_sha256"] or actual == entry["expected_sha256"]:
            return {**entry, "sha256": actual, "verified": True}
    partial = destination.with_name(destination.name + ".part")
    if entry["bytes"] > 128 * 1024 * 1024:
        progress = ranged_download(entry, partial)
        actual = sha256(partial)
        if entry["expected_sha256"] and actual != entry["expected_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {destination}")
        partial.replace(destination)
        progress.unlink()
        print(f"Verified {entry['folder']}/{entry['file']} ({entry['bytes']/1e9:.3f} GB)", flush=True)
        return {**entry, "sha256": actual, "verified": True}
    url = transport_url(entry["url"])
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry["bytes"]:
            break
        headers = {"User-Agent": "Codex-model-setup", "Range": f"bytes={offset}-"}
        try:
            request = urllib.request.Request(url + "?download=true&fresh=" + str(int(time.time())), headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(2 * 1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size == entry["bytes"]:
                break
        except Exception as error:
            print(f"Retry {entry['folder']}/{entry['file']} {attempt+1}: {error}", flush=True)
            time.sleep(min(2 * (attempt + 1), 10))
    if not partial.exists() or partial.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"Incomplete model download {destination}")
    actual = sha256(partial)
    if entry["expected_sha256"] and actual != entry["expected_sha256"]:
        raise ValueError(f"SHA-256 mismatch for {destination}")
    partial.replace(destination)
    print(f"Verified {entry['folder']}/{entry['file']} ({entry['bytes']/1e9:.3f} GB)", flush=True)
    return {**entry, "sha256": actual, "verified": True}


def main():
    import fnmatch
    entries, revisions = [], {}
    ROOT.mkdir(parents=True, exist_ok=True)
    saved_path = ROOT / "weights/download_manifest.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        entries, revisions = saved["files"], saved["models"]
    for folder, (repo, patterns) in (() if entries else MODELS.items()):
        meta = get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        revision = meta["sha"]
        revisions[folder] = {"repository": repo, "revision": revision}
        for row in meta["siblings"]:
            name = row["rfilename"]
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                entries.append({"folder": folder, "repository": repo, "revision": revision,
                                "file": name, "bytes": row["size"],
                                "expected_sha256": row.get("lfs", {}).get("sha256"),
                                "url": f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"})
    required = sum(e["bytes"] for e in entries)
    existing = sum(p.stat().st_size for p in (ROOT / "weights").rglob("*") if p.is_file())
    remaining = max(0, required - existing)
    if shutil.disk_usage(ROOT).free < remaining + 4 * 1024**3:
        raise OSError(f"Remaining model files need {remaining / 1e9:.2f} GB plus 4 GiB free space")
    manifest = {"created_unix": int(time.time()), "total_bytes": required, "models": revisions,
                "files": entries, "dataset_files_downloaded": False,
                "continuation_transport_requested": DOWNLOAD_HOST}
    manifest_path = ROOT / "weights/download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloading {required/1e9:.2f} GB of official model files", flush=True)
    completed = []
    # Small configuration files and the two retrieval models arrive first.
    entries.sort(key=lambda e: (e["bytes"] > 2_000_000_000, e["folder"], e["bytes"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, entries):
            completed.append(result)
            (ROOT / "weights/verified_downloads.json").write_text(
                json.dumps(completed, indent=2), encoding="utf-8")
    manifest["files"] = completed
    manifest["all_verified"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("All official model files verified", flush=True)


if __name__ == "__main__":
    main()
