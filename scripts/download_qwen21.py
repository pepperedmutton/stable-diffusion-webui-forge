"""Download official Qwen-Image 2.1 with parallel, resumable HTTP ranges.

Every range response is checked against Content-Range. Completed model files are
published only after the SHA-256 from the pinned Hugging Face manifest matches.
Small Git files are checked with their Git blob SHA-1. No model substitutions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import quote

import requests


REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({"Accept-Encoding": "identity", "User-Agent": "Forge-Qwen21-downloader/1"})
    return LOCAL.session


def digest(path, item):
    algorithm = hashlib.sha256() if item.get("lfs") else hashlib.sha1()
    if not item.get("lfs"):
        algorithm.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            algorithm.update(block)
    return algorithm.hexdigest()


def expected_digest(item):
    return item.get("lfs", {}).get("oid", item["oid"])


class RangeFile:
    def __init__(self, destination, item, cache, chunk_size, log, source_url=None):
        self.destination, self.item = destination, item
        self.chunk_size, self.log = chunk_size, log
        self.size = item["size"]
        self.huggingface_url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(item['path'], safe='/')}?download=true"
        self.resolve_url = source_url or self.huggingface_url
        identifier = expected_digest(item)
        self.partial = cache / (identifier + ".partial")
        self.state_path = cache / (identifier + ".json")
        self.lock = threading.Lock()
        self.url_lock = threading.Lock()
        self.url, self.url_timestamp = None, 0
        self.done = set()
        self.complete = False
        self.chunk_count = (self.size + chunk_size - 1) // chunk_size
        if destination.is_file() and destination.stat().st_size == self.size:
            if digest(destination, item) == expected_digest(item):
                self.complete = True
                log("verified", path=item["path"], bytes=self.size, existing=True)
                return
        if self.state_path.exists() and self.partial.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (state.get("revision") == REVISION and state.get("digest") == identifier
                    and state.get("size") == self.size and state.get("chunk_size") == chunk_size
                    and self.partial.stat().st_size == self.size):
                self.done = {int(index) for index in state["done"] if 0 <= int(index) < self.chunk_count}
        if not self.done:
            with self.partial.open("wb") as handle:
                handle.truncate(self.size)
            self.save_state()

    def bounds(self, index):
        start = index * self.chunk_size
        return start, min(start + self.chunk_size, self.size) - 1

    def completed_bytes(self):
        if self.complete:
            return self.size
        with self.lock:
            return sum(self.bounds(index)[1] - self.bounds(index)[0] + 1 for index in self.done)

    def save_state(self):
        temporary = self.state_path.with_suffix(".json.new")
        temporary.write_text(json.dumps({
            "revision": REVISION, "path": self.item["path"], "size": self.size,
            "digest": expected_digest(self.item), "chunk_size": self.chunk_size,
            "done": sorted(self.done),
        }), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def download_url(self, refresh=False):
        with self.url_lock:
            if not refresh and self.url and time.monotonic() - self.url_timestamp < 1200:
                return self.url
            # Resolve one signed CDN URL and reuse it across range requests.
            with session().get(self.resolve_url, headers={"Range": "bytes=0-0"},
                               stream=True, timeout=(20, 40)) as response:
                response.raise_for_status()
                self.url = response.url
                self.url_timestamp = time.monotonic()
            return self.url

    def fetch_chunk(self, index, retries, on_bytes):
        start, end = self.bounds(index)
        expected = f"bytes {start}-{end}/{self.size}"
        last_error = None
        for attempt in range(retries):
            try:
                url = self.download_url(refresh=attempt > 0 and attempt % 2 == 0)
                with session().get(url, headers={"Range": f"bytes={start}-{end}"},
                                   stream=True, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    complete_small_file = start == 0 and end == self.size - 1 and response.status_code == 200
                    valid_range = response.status_code == 206 and response.headers.get("Content-Range") == expected
                    if not valid_range and not complete_small_file:
                        raise IOError(f"Unexpected HTTP range: {response.status_code} {response.headers.get('Content-Range')}; expected {expected}")
                    written = 0
                    with self.partial.open("r+b", buffering=0) as output:
                        output.seek(start)
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if written + len(block) > end - start + 1:
                                raise IOError("Range response exceeded the requested byte count")
                            output.write(block)
                            written += len(block)
                            on_bytes(len(block))
                        if written != end - start + 1:
                            raise IOError(f"Short range response: {written} bytes")
                        output.flush()
                with self.lock:
                    self.done.add(index)
                    self.save_state()
                return
            except (requests.RequestException, OSError) as error:
                last_error = error
                if attempt == 2 and self.resolve_url != self.huggingface_url:
                    with self.url_lock:
                        self.resolve_url = self.huggingface_url
                        self.url = None
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 8))
        raise IOError(f"{self.item['path']} range {start}-{end} failed: {last_error}")

    def finalize(self):
        if self.complete or len(self.done) != self.chunk_count:
            return
        self.log("verifying", path=self.item["path"], bytes=self.size)
        actual = digest(self.partial, self.item)
        if actual != expected_digest(self.item):
            # Do not publish an invalid checkpoint, and do not trust its chunks
            # on the next run. The .partial remains available for diagnosis.
            with self.lock:
                self.done.clear()
                self.save_state()
            raise IOError(f"SHA checksum mismatch for {self.item['path']}: {actual}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.partial, self.destination)
        self.state_path.unlink(missing_ok=True)
        self.complete = True
        self.log("verified", path=self.item["path"], bytes=self.size, existing=False)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root / "models" / "diffusers" / "Qwen-Image-2.1")
    parser.add_argument("--manifest", type=Path, default=root / "tmp" / "qwen21-install" / "hub-manifest.json")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--source", choices=("auto", "huggingface"), default="auto",
                        help="auto uses official ModelScope for weights whose SHA matches the pinned HF manifest")
    parser.add_argument("--log", type=Path, default=root / "tmp" / "qwen21-install" / "range-download.jsonl")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 1 <= args.chunk_mib <= 128 or args.retries < 1:
        parser.error("workers must be 1..64, chunk-mib 1..128, and retries positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    log_handle = args.log.open("a", encoding="utf-8", buffering=1)

    def log(event, **values):
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **values}, ensure_ascii=False)
        print(line, flush=True)
        log_handle.write(line + "\n")

    if not args.manifest.is_file():
        manifest = []
        url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true&expand=false"
        while url:
            response = session().get(url, timeout=(15, 30))
            response.raise_for_status()
            manifest.extend(response.json())
            url = response.links.get("next", {}).get("url")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    items = [item for item in manifest if item["type"] == "file" and not item["path"].startswith("assets/")]
    source_urls = {}
    if args.source == "auto":
        try:
            response = session().get(f"https://modelscope.cn/api/v1/models/{REPO}/repo/files",
                                     params={"Revision": "master", "Recursive": "true"}, timeout=(15, 30))
            response.raise_for_status()
            scope_manifest = response.json()
            (args.log.parent / "modelscope-manifest.json").write_text(
                json.dumps(scope_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            scope_files = {file["Path"]: file for file in scope_manifest["Data"]["Files"]}
            for item in items:
                source = scope_files.get(item["path"], {})
                if (item.get("lfs") and source.get("Sha256") == item["lfs"]["oid"]
                        and source.get("Size") == item["size"]):
                    # Each source path is pinned to the commit that last changed
                    # that file, and must match the independent HF SHA-256.
                    source_urls[item["path"]] = (
                        f"https://modelscope.cn/models/{REPO}/resolve/{source['Revision']}/"
                        f"{quote(item['path'], safe='/')}"
                    )
            log("sources", modelscope_sha_matched=len(source_urls), fallback="huggingface")
        except (requests.RequestException, KeyError, ValueError) as error:
            log("source_fallback", source="huggingface", reason=str(error))
    cache = args.model_dir / ".range-download"
    cache.mkdir(exist_ok=True)
    files = []
    for item in items:
        destination = (args.model_dir / item["path"]).resolve()
        if not destination.is_relative_to(args.model_dir.resolve()):
            raise ValueError(f"Unsafe model manifest path: {item['path']}")
        files.append(RangeFile(destination, item, cache, args.chunk_mib * 1024 * 1024, log,
                               source_url=source_urls.get(item["path"])))
    total = sum(item["size"] for item in items)
    initial = sum(file.completed_bytes() for file in files)
    transferred = 0
    transfer_lock = threading.Lock()

    def on_bytes(count):
        nonlocal transferred
        with transfer_lock:
            transferred += count

    log("start", repo=REPO, revision=REVISION, total_bytes=total,
        resumed_bytes=initial, workers=args.workers, chunk_mib=args.chunk_mib)
    started, previous_time, previous_bytes = time.monotonic(), time.monotonic(), 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        # Round-robin across files avoids tying every connection to one object.
        for index in range(max(file.chunk_count for file in files)):
            for file in files:
                if not file.complete and index < file.chunk_count and index not in file.done:
                    future = executor.submit(file.fetch_chunk, index, args.retries, on_bytes)
                    pending[future] = file
        while pending:
            completed, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in completed:
                file = pending.pop(future)
                try:
                    future.result()
                    file.finalize()
                except Exception as error:
                    failures.append(str(error))
                    log("error", message=str(error))
            now = time.monotonic()
            if now - previous_time >= 5 or not pending:
                completed_bytes = sum(file.completed_bytes() for file in files)
                log("progress", completed_bytes=completed_bytes, total_bytes=total,
                    percent=round(completed_bytes * 100 / total, 2),
                    network_mib_per_second=round((transferred - previous_bytes) / (now - previous_time) / 1048576, 2),
                    elapsed_seconds=round(now - started), pending_ranges=len(pending))
                previous_time, previous_bytes = now, transferred
    # Already-downloaded ranges from a previous run may require only assembly
    # verification, with no pending future to trigger finalization.
    for file in files:
        try:
            file.finalize()
        except Exception as error:
            failures.append(str(error))
            log("error", message=str(error))
    if failures or not all(file.complete for file in files):
        log("incomplete", failed_ranges=len(failures), message="Run this command again to resume verified ranges.")
        raise SystemExit(1)
    log("complete", total_bytes=total, files=len(files), revision=REVISION)
    log_handle.close()


if __name__ == "__main__":
    main()
