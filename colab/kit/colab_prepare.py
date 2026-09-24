"""Normalize settings for Drive/ForgeColab/{forge,models,embeddings,state,outputs}.

Forge reads the sibling model and embedding directories through launch arguments.
No model links or runtime copies are created here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

WINDOWS_ROOT = "D:/Forge/stable-diffusion-webui-forge"
DEFAULT_CHECKPOINT = "waiIllustriousSDXL_v150.safetensors"
STATE_FILES = ("config.json", "ui-config.json")
OUTPUT_FOLDERS = {
    "outdir_txt2img_samples": "txt2img-images",
    "outdir_img2img_samples": "img2img-images",
    "outdir_extras_samples": "extras-images",
    "outdir_txt2img_grids": "txt2img-grids",
    "outdir_img2img_grids": "img2img-grids",
    "outdir_save": "saved-images",
    "outdir_init_images": "init-images",
}
PATH_OPTIONS = {
    "control_net_models_path", "control_net_modules_path", "sd_model_checkpoint",
    "sd_vae", "font", "styles_file",
}
UI_PATH_LABELS = {
    "input directory", "output directory", "mask directory",
    "controlnet input directory", "png info directory",
    "inpaint batch mask directory (required for inpaint batch processing only)",
}


def safe_target(base: Path, relative: str) -> Path:
    """Never follow an existing directory/file symlink when writing settings."""
    base = base.resolve()
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise ValueError(f"Path escapes its storage directory: {path}")
    cursor = path
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError(f"A writable settings path must not be a symlink: {cursor}")
        cursor = cursor.parent
    return path


def atomic_json(path: Path, value: dict) -> bool:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8-sig") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def remap_path(value, root: Path, warnings: list[str], label: str, drive_root: Path | None = None):
    drive_root = drive_root if drive_root is not None else root.parent
    if isinstance(value, list):
        return [remap_path(item, root, warnings, label, drive_root) for item in value]
    if not isinstance(value, str) or not value:
        return value
    normalized = re.sub(r"\\+", "/", value)
    current = root.as_posix()
    def relocated(suffix):
        parts = suffix.lstrip("/").split("/", 1)
        if parts[0].casefold() in {"models", "embeddings"}:
            base = (drive_root / parts[0].casefold()).as_posix()
            return base + ("/" + parts[1] if len(parts) == 2 else "")
        return current + ("/" + suffix.lstrip("/") if suffix else "")

    # Already migrated sibling paths are stable. Source-relative models and
    # embeddings are remapped too, because --models-dir does not rewrite JSON.
    for folder in ("models", "embeddings"):
        sibling = (drive_root / folder).as_posix()
        if normalized == sibling or normalized.startswith(sibling + "/"):
            return normalized
        if normalized.casefold() == folder or normalized.casefold().startswith(folder + "/"):
            return relocated(normalized)
    for old in (current, WINDOWS_ROOT, "/content/forge-web", "/content/forge"):
        if normalized.casefold() == old.casefold():
            return current
        if normalized.casefold().startswith(old.casefold() + "/"):
            return relocated(normalized[len(old):])
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("//"):
        warnings.append(f"Unmapped Windows path in {label}: {value}")
    return normalized


def normalize_settings(config: dict, ui: dict, root: Path, drive_root: Path, first_run: bool):
    config, ui, warnings, removed = dict(config), dict(ui), [], []
    config["localization"] = "None"
    config["forge_canvas_toolbar_always"] = True
    for key, value in list(config.items()):
        if key in PATH_OPTIONS or key.startswith(("forge_additional_modules", "forge_checkpoint_")):
            config[key] = remap_path(value, root, warnings, key, drive_root)
    for key, value in list(ui.items()):
        parts = key.rsplit("/", 2)
        if len(parts) == 3 and parts[-1] == "value" and parts[-2].casefold() in UI_PATH_LABELS:
            ui[key] = remap_path(value, root, warnings, key, drive_root)
    # Only path-valued settings are remapped; prompts and metadata stay verbatim.
    config["outdir_samples"] = ""
    config["outdir_grids"] = ""
    for key, folder in OUTPUT_FOLDERS.items():
        config[key] = (drive_root / "outputs" / folder).as_posix()
    config["control_net_detectedmap_dir"] = (drive_root / "outputs" / "detected_maps").as_posix()
    config["temp_dir"] = (root / "tmp" / "gradio").as_posix()
    for key in list(config):
        if key.endswith("_GPU_MB") or key == "forge_inference_memory":
            removed.append(key)
            del config[key]
    for key in list(ui):
        if "/GPU Weights (MB)/" in key:
            removed.append(key)
            del ui[key]
    # The only requested model substitution is Novsw -> the existing WAI v150.
    # Preserve all other explicit selections, including an existing Qwen choice.
    def is_novsw(value):
        name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
        name = re.sub(r"\s*\[[a-fA-F0-9]+\]$", "", name)
        return name.casefold() in {"novsw", "novsw.safetensors"}

    current = config.get("sd_model_checkpoint")
    replace_current = is_novsw(current)
    wai_exists = (drive_root / "models/Stable-diffusion" / DEFAULT_CHECKPOINT).is_file()
    if replace_current and not wai_exists:
        raise ValueError("Download WAI Illustrious SDXL v150 to Drive/models before replacing the Novsw selection")
    if replace_current or (not current and wai_exists):
        config.update({
            "forge_preset": "xl", "sd_model_checkpoint": DEFAULT_CHECKPOINT,
            "forge_checkpoint_xl": DEFAULT_CHECKPOINT,
            "forge_additional_modules": [], "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
    elif is_novsw(config.get("forge_checkpoint_xl")):
        config["forge_checkpoint_xl"] = DEFAULT_CHECKPOINT
        config["forge_additional_modules_xl"] = []
        config["forge_additional_modules_xl_configured"] = False
    if first_run:
        for tab in ("txt2img", "img2img"):
            for label in ("Width", "Height"):
                ui[f"{tab}/{label}/value"] = 512
            for label in ("Batch size", "Batch count"):
                ui[f"{tab}/{label}/value"] = 1
            ui[f"customscript/sampler.py/{tab}/Sampling steps/value"] = 20
    return config, ui, removed, warnings


def prepare(root: Path, drive_root: Path) -> dict:
    if root.is_symlink() or drive_root.is_symlink():
        raise ValueError("Forge and Drive storage paths must not be symlinks")
    root, drive = root.resolve(), drive_root.resolve()
    if root != drive / "forge":
        raise ValueError("Forge source must be the forge/ directory directly inside --drive-root")
    if not (root / "launch.py").is_file():
        raise ValueError(f"Forge launch.py is missing: {root}")
    # Validate every write target before creating or replacing anything.
    outputs = safe_target(drive, "outputs")
    embeddings = safe_target(drive, "embeddings")
    model_directory = safe_target(drive, "models")
    if not model_directory.is_dir():
        raise ValueError("Drive/models is missing; download models before starting")
    safe_target(root, "tmp/gradio")
    local = [safe_target(root, name) for name in STATE_FILES]
    targets = [safe_target(drive, "state/" + name) for name in STATE_FILES]
    marker = safe_target(root, ".colab/prepared.json")
    first_run = not targets[0].exists()
    source = [target if target.exists() else old for target, old in zip(targets, local)]
    values = [read_json(path) if path.exists() else {} for path in source]
    config, ui, removed, warnings = normalize_settings(*values, root, drive, first_run)
    if warnings:
        raise ValueError("Correct these paths in Drive/state before starting:\n" + "\n".join(warnings))
    # A fresh Git clone has no personal config files. Existing local files are
    # read as migration input only; Forge runs directly against Drive/state.
    outputs.mkdir(parents=True, exist_ok=True)
    embeddings.mkdir(parents=True, exist_ok=True)
    (root / "tmp/gradio").mkdir(parents=True, exist_ok=True)
    changed = [str(path) for path, value in zip(targets, (config, ui)) if atomic_json(path, value)]
    summary = {
        "settings_directory": str(drive / "state"), "outputs_directory": str(outputs),
        "models_directory": str(model_directory), "embeddings_directory": str(embeddings),
        "first_run_defaults_applied": first_run, "removed_gpu_settings": removed,
        "changed_files": changed,
    }
    atomic_json(marker, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run against the isolated Linux Colab checkout, not Windows Forge")
    print(json.dumps(prepare(args.root, args.drive_root), indent=2))


if __name__ == "__main__":
    main()
