"""Link uploaded Drive models; optional integrity checks and explicit local caches.

The default reads file sizes, not every byte of a large model collection. No
model is downloaded, removed from Drive, or copied to Colab unless --cache is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("Model paths must be nonempty relative paths")
    path = PurePosixPath(value)
    if (path.is_absolute() or "\\" in value or ":" in value or
            any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Unsafe model path: {value}")
    return path.as_posix()


def inventory(drive: Path, verify_sha256: bool = False):
    models = drive / "models"
    if not models.is_dir() or models.is_symlink():
        raise ValueError("Upload your models to a real Drive/models directory first")
    files = {}
    for folder, directories, names in os.walk(models, followlinks=False):
        for name in directories:
            if (Path(folder) / name).is_symlink():
                raise ValueError(f"Copy the contents of model directory symlinks before uploading: {name}")
        for name in names:
            source = Path(folder) / name
            relative = relative_path(source.relative_to(models).as_posix())
            if not source.resolve().is_relative_to(models.resolve()) or not source.is_file():
                raise ValueError(f"Model file points outside Drive/models: {relative}")
            files[relative] = {"path": relative, "bytes": source.stat().st_size}
    if not files:
        raise ValueError("Drive/models is empty; upload your existing weights before starting")
    manifest_path = drive / "models-manifest.json"
    if not manifest_path.exists():
        if verify_sha256:
            raise ValueError("--verify-sha256 needs models-manifest.json with SHA256 for every listed file")
        return files, {"manifest": False, "listed_files": 0, "extra_files": len(files)}
    if manifest_path.is_symlink():
        raise ValueError("models-manifest.json must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("models-manifest.json must contain a nonempty files array")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each model manifest entry must be an object")
        relative = relative_path(entry.get("path"))
        if relative in seen:
            raise ValueError(f"Duplicate model manifest entry: {relative}")
        seen.add(relative)
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid model size: {relative}")
        if relative not in files or files[relative]["bytes"] != size:
            raise ValueError(f"Model missing or upload incomplete (size mismatch): {relative}")
        digest = entry.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
            raise ValueError(f"Invalid SHA256 in model manifest: {relative}")
        if verify_sha256 and (digest is None or sha256(models / relative) != digest.lower()):
            raise ValueError(f"Model SHA256 missing or mismatched: {relative}")
        if digest:
            files[relative]["sha256"] = digest.lower()
    return files, {"manifest": True, "listed_files": len(seen), "extra_files": len(files) - len(seen)}


def safe_destination(root: Path, relative: str) -> Path:
    destination = root / "models" / relative
    cursor = destination.parent
    while cursor != root:
        if cursor.is_symlink() or not cursor.resolve().is_relative_to(root):
            raise ValueError(f"Refusing to write through a model-directory symlink: {cursor}")
        if cursor.exists() and not cursor.is_dir():
            raise ValueError(f"A file blocks the model directory: {cursor}")
        cursor = cursor.parent
    return destination


def stage(root: Path, drive_root: Path, cache=(), verify_sha256=False, verify_only=False):
    root, drive = root.resolve(), drive_root.resolve()
    if root == drive or root in drive.parents or drive in root.parents:
        raise ValueError("The local Forge root and Drive storage must be separate")
    if not root.is_dir():
        raise ValueError("The isolated Forge checkout is missing")
    files, verification = inventory(drive, verify_sha256)
    cache = [relative_path(value.rstrip("/")) for value in cache]
    for prefix in cache:
        if not any(path == prefix or path.startswith(prefix + "/") for path in files):
            raise ValueError(f"Cache selection was not uploaded: {prefix}")
    if verify_only:
        return {**verification, "files": len(files), "verification_only": True}
    plan, needed = [], 0
    # Validate the entire plan before modifying a single destination.
    for relative, entry in sorted(files.items()):
        source, target = drive / "models" / relative, safe_destination(root, relative)
        caching = any(relative == prefix or relative.startswith(prefix + "/") for prefix in cache)
        existing_link = target.is_symlink()
        if existing_link and target.resolve() != source.resolve():
            raise ValueError(f"An unrelated model link already exists: {target}")
        if target.exists() and not existing_link:
            # Keep identical tracked helper files and an already cached model.
            # A different existing file is never replaced automatically.
            if not target.is_file() or target.stat().st_size != entry["bytes"]:
                raise ValueError(f"An existing local model differs; keep it or move it explicitly: {target}")
            expected = entry.get("sha256") or sha256(source)
            if sha256(target) != expected:
                raise ValueError(f"An existing local model differs; keep it or move it explicitly: {target}")
            continue
        if caching:
            needed += entry["bytes"]
        elif existing_link:
            continue
        plan.append((source, target, entry, caching))
    if needed and needed + 15 * 2**30 > shutil.disk_usage(root).free:
        raise ValueError(f"The requested cache needs {needed / 2**30:.1f} GiB plus 15 GiB free; choose fewer --cache entries")
    for source, target, entry, caching in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not caching:
            target.symlink_to(source)
            continue
        print("Caching model:", entry["path"], flush=True)
        fd, temporary = tempfile.mkstemp(prefix=".colab-model-", dir=target.parent)
        os.close(fd)
        part = Path(temporary)
        try:
            shutil.copyfile(source, part)
            if part.stat().st_size != entry["bytes"]:
                raise ValueError(f"Incomplete cache copy: {entry['path']}")
            if entry.get("sha256") and sha256(part) != entry["sha256"]:
                raise ValueError(f"Cache checksum failed: {entry['path']}")
            # Atomic replacement replaces the link itself, never its Drive target.
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    return {**verification, "files": len(files), "cached_bytes_this_run": needed,
            "sha256_checked": bool(verify_sha256), "models_downloaded": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--cache", action="append", default=[], help="Explicit model path relative to models/; repeatable")
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run against the isolated Linux Colab checkout, not Windows Forge")
    print(json.dumps(stage(args.root, args.drive_root, args.cache, args.verify_sha256, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
