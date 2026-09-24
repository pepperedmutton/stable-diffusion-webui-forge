"""Validate persistent Drive models for a Drive-resident Forge checkout.

Forge receives Drive/models and Drive/embeddings directly through --models-dir
and --embeddings-dir. No files are copied, downloaded, linked, or removed. The
default validates file sizes; --verify-sha256 reads manifest-listed model bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


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
        raise ValueError("Download models to a real Drive/models directory first")
    files = {}
    for folder, directories, names in os.walk(models, followlinks=False):
        for name in directories:
            if (Path(folder) / name).is_symlink():
                raise ValueError(f"Model directories must contain real files, not symlinks: {name}")
        for name in names:
            source = Path(folder) / name
            relative = relative_path(source.relative_to(models).as_posix())
            if source.is_symlink() or not source.resolve().is_relative_to(models.resolve()) or not source.is_file():
                raise ValueError(f"Model file points outside Drive/models: {relative}")
            files[relative] = {"path": relative, "bytes": source.stat().st_size}
    if not files:
        raise ValueError("Drive/models is empty; download model weights before starting")
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
            raise ValueError(f"Model missing or download incomplete (size mismatch): {relative}")
        digest = entry.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
            raise ValueError(f"Invalid SHA256 in model manifest: {relative}")
        if verify_sha256 and (digest is None or sha256(models / relative) != digest.lower()):
            raise ValueError(f"Model SHA256 missing or mismatched: {relative}")
        if digest:
            files[relative]["sha256"] = digest.lower()
    return files, {"manifest": True, "listed_files": len(seen), "extra_files": len(files) - len(seen)}


def embeddings_inventory(directory: Path) -> int:
    """An absent embeddings directory is valid; prepare() creates it in Drive."""
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Drive/embeddings must be a real directory")
    if not directory.exists():
        return 0
    count = 0
    for folder, directories, names in os.walk(directory, followlinks=False):
        for name in directories:
            if (Path(folder) / name).is_symlink():
                raise ValueError("Embedding directories must not be symlinks")
        for name in names:
            source = Path(folder) / name
            if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Embedding files must stay inside Drive/embeddings without symlinks")
            count += 1
    return count


def stage(root: Path, drive_root: Path, cache=(), verify_sha256=False, verify_only=False):
    if cache:
        raise ValueError("Local model caching is disabled; Forge reads the persistent Drive/models directory directly")
    if root.is_symlink() or drive_root.is_symlink():
        raise ValueError("Forge and Drive storage paths must not be symlinks")
    root, drive = root.resolve(), drive_root.resolve()
    if root != drive / "forge":
        raise ValueError("Forge source must be the forge/ directory directly inside --drive-root")
    if not root.is_dir() or not (root / "launch.py").is_file():
        raise ValueError("The persistent Forge checkout is missing launch.py")
    files, verification = inventory(drive, verify_sha256)
    embedding_files = embeddings_inventory(drive / "embeddings")
    return {**verification, "files": len(files), "embedding_files": embedding_files,
            "models_directory": str(drive / "models"), "embeddings_directory": str(drive / "embeddings"),
            "verification_only": bool(verify_only), "sha256_checked": bool(verify_sha256),
            "runtime_model_copies": 0, "model_links_created": 0, "models_downloaded": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run against the isolated Linux Colab checkout, not Windows Forge")
    print(json.dumps(stage(args.root, args.drive_root, verify_sha256=args.verify_sha256, verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
