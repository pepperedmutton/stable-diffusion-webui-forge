#!/usr/bin/env python3
"""Real Forge API smoke matrix. Run inside the Colab runtime, against localhost.

Example (exclusive QA session; do not use the UI concurrently):
  %run /content/drive/MyDrive/ForgeColab/forge/colab/verify_models.py --run

Keep this script and qa-models.json together in the Forge colab/ directory.
The expected Forge source root is /content/drive/MyDrive/ForgeColab/forge.
The current Colab interpreter can run this API client; it needs Pillow, not the
Forge interpreter or model libraries. Results default to a new Drive/qa/results
session directory. No QA result is stored in the temporary runtime by default.

Start normal Forge with --api --port 7860, without --share or --listen. Avoid
--nowebui: this fork initializes its module registry while building the UI.
--gradio-auth-path does NOT protect /sdapi routes;
if the server is shared publicly, also use --api-auth and supply its value through
FORGE_QA_API_AUTH (username:password), never a command-line argument.

No model is downloaded and no GPU workload runs without --run. --plan is offline.
Success proves actual image output/model identity, not aesthetic quality or a
successful Pixel UI workflow. Inspect the PNGs/contact sheet before acceptance.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid


DRIVE_BASE = Path("/content/drive/MyDrive/ForgeColab")
QA_ROOT = DRIVE_BASE / "qa"
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("qa-models.json")


PROFILES = [
    {"id": "qwen-nf4", "preset": "qwen", "model": "Qwen-Image-2.1-NF4", "precision": "nf4", "modules": [], "steps": 20, "cfg": 1.0, "sampler": "Euler", "scheduler": "Simple"},
    {"id": "qwen-int8", "preset": "qwen", "model": "Qwen-Image-2.1-INT8", "precision": "int8", "modules": [], "steps": 20, "cfg": 1.0, "sampler": "Euler", "scheduler": "Simple"},
    {"id": "qwen-bf16", "preset": "qwen", "model": "Qwen-Image-2.1-BF16", "precision": "bf16", "modules": [], "steps": 20, "cfg": 1.0, "sampler": "Euler", "scheduler": "Simple"},
    {"id": "illustrious", "preset": "xl", "model": "miaomiaoHarem_v20.safetensors", "modules": [], "steps": 20, "cfg": 6.0, "sampler": "Euler a", "scheduler": "Automatic"},
    {"id": "xl-wai-v150", "preset": "xl", "model": "waiIllustriousSDXL_v150.safetensors", "modules": [], "steps": 20, "cfg": 6.0, "sampler": "Euler a", "scheduler": "Automatic"},
    {"id": "anima", "preset": "anima", "model": "miaomiaoHarem_anima15.safetensors", "modules": ["anima_text_encoder.safetensors", "anima_vae.safetensors"], "steps": 20, "cfg": 4.0, "sampler": "Euler a", "scheduler": "Normal"},
    {"id": "krea", "preset": "krea", "model": "krea2Cocoamixzero_v10.safetensors", "modules": ["qwen3vl_4b_fp8_scaled.safetensors", "qwen_image_vae.safetensors"], "steps": 8, "cfg": 1.0, "sampler": "Euler", "scheduler": "Simple"},
]
DEFAULTS = {preset: next(profile for profile in PROFILES if profile["preset"] == preset)
            for preset in ("qwen", "xl", "anima", "krea")}
SEED = 902104
TEXT_PROMPT = "A single green ceramic mug on a pale wooden table, a lemon beside the mug, soft daylight, clean still life, detailed, no text, no watermark"
IMAGE_PROMPT = "A single bright red ceramic mug on a pale wooden table, a lemon beside the mug, soft daylight, clean still life, detailed, no text, no watermark"
EDIT_PROMPT = "Change the green mug to bright red. Keep the lemon, table, lighting and camera angle unchanged."
NEGATIVE = "text, watermark, blurry, low quality"
TOUCH_KEYS = {
    "sd_model_checkpoint", "forge_preset", "forge_additional_modules",
    "forge_additional_modules_qwen", "forge_additional_modules_krea", "forge_additional_modules_anima",
    "forge_additional_modules_xl", "forge_additional_modules_xl_configured",
    "forge_checkpoint_qwen", "forge_checkpoint_krea", "forge_checkpoint_anima", "forge_checkpoint_xl",
    "forge_qwen21_precision", "forge_unet_storage_dtype", "forge_qwen21_profile_version",
    "live_previews_enable", "show_progress_every_n_steps", "samples_format", "grid_save",
    "CLIP_stop_at_last_layers", "xl_t2i_sampler", "xl_i2i_sampler", "xl_t2i_scheduler", "xl_i2i_scheduler",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    value = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\s*\[[a-fA-F0-9]+\]$", "", value).removesuffix(".safetensors").casefold()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_profiles(path):
    if path is None:
        raw_profiles = PROFILES
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_profiles = data.get("profiles") if isinstance(data, dict) else data
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("The manifest must contain a nonempty profiles array")
    profiles, seen = [], set()
    for raw in raw_profiles:
        if (not isinstance(raw, dict) or raw.get("preset") not in DEFAULTS
                or "id" not in raw or "model" not in raw):
            raise ValueError("Each profile needs an explicit id, model, and preset qwen/xl/anima/krea")
        profile = {**DEFAULTS[raw["preset"]], **raw}
        identifier = profile.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", identifier) or identifier in seen:
            raise ValueError("Each profile id must be unique, lowercase ASCII letters/digits/hyphens")
        seen.add(identifier)
        if not isinstance(profile.get("model"), str) or not profile["model"].strip():
            raise ValueError(f"Missing model name for {identifier}")
        if profile["preset"] == "qwen" and profile.get("precision") not in {"nf4", "int8", "bf16"}:
            raise ValueError(f"Invalid Qwen precision for {identifier}")
        if not isinstance(profile["modules"], list) or not all(isinstance(value, str) for value in profile["modules"]):
            raise ValueError(f"Invalid modules for {identifier}")
        if not isinstance(profile["steps"], int) or not 1 <= profile["steps"] <= 150 or not 1 <= profile["cfg"] <= 30:
            raise ValueError(f"Invalid steps or CFG for {identifier}")
        profile["modes"] = profile.get("modes", ["txt2img", "img2img"])
        if profile["modes"] not in (["txt2img"], ["txt2img", "img2img"]):
            raise ValueError("Modes must be [txt2img] or [txt2img,img2img] in that order")
        profiles.append(profile)
    return profiles


class API:
    def __init__(self, base, timeout):
        parsed = urlsplit(base)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Use an HTTP loopback origin inside the Colab runtime; public API endpoints are rejected")
        self.base, self.timeout = base.rstrip("/"), timeout
        credential = os.environ.get("FORGE_QA_API_AUTH", "")
        self.authorization = "Basic " + base64.b64encode(credential.encode()).decode() if credential else None

    def call(self, method, endpoint, payload=None, timeout=None):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        request = urllib.request.Request(self.base + "/sdapi/v1/" + endpoint, body, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(12000).decode("utf-8", "replace")
            credential = os.environ.get("FORGE_QA_API_AUTH", "")
            if credential:
                detail = detail.replace(credential, "[private]").replace(credential.partition(":")[2], "[private]")
            raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from None

    def idle(self, seconds=90):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            status = self.call("GET", "progress?skip_current_image=true", timeout=15)
            state = status.get("state") or {}
            if not state.get("job_count", 0) and not state.get("job", ""):
                return True
            time.sleep(2)
        return False


def resolve_model(profile, models):
    matches = [model for model in models if canonical(profile["model"]) in {
        canonical(model.get("title", "")), canonical(model.get("model_name", "")), canonical(model.get("filename", "")),
    }]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered {profile['model']}; found {len(matches)}. Download/repair the full model and refresh checkpoints")
    return matches[0]


def resolve_modules(profile, modules):
    found = []
    for name in profile["modules"]:
        matches = [module for module in modules if canonical(name) in {canonical(module.get("model_name", "")), canonical(module.get("filename", ""))}]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one additional module {name}; found {len(matches)}")
        found.append(matches[0]["filename"])
    return sorted(found)


def disable_optional_scripts(infos, is_img2img):
    """Only override documented enable booleans; keep unrelated script arguments."""
    result = {}
    for info in infos:
        if not info.get("is_alwayson") or info.get("is_img2img") != is_img2img:
            continue
        arguments = info.get("args") or []
        values, changed = [], False
        for argument in arguments:
            value = argument.get("value")
            label = str(argument.get("label") or "").casefold()
            if isinstance(value, bool) and (label.startswith("enable") or label == "dynamic prompts enabled"):
                value, changed = False, True
            values.append(value)
        if changed:
            result[info["name"]] = {"args": values}
    return result


def select_profile(api, profile, model, modules):
    # sysinfo.set_config infers preset from checkpoint and refreshes loading
    # parameters at the end. Required module profiles must be set BEFORE it.
    options = {
        "live_previews_enable": False, "samples_format": "png", "grid_save": False,
        "CLIP_stop_at_last_layers": 1, "forge_unet_storage_dtype": "Automatic",
        "forge_preset": profile["preset"],
    }
    if profile["preset"] in {"qwen", "krea", "anima"}:
        options["forge_additional_modules_" + profile["preset"]] = modules
    else:
        # Let the checkpoint's audited structure select built-in/external VAE.
        options["forge_additional_modules_xl"] = []
        options["forge_additional_modules_xl_configured"] = False
    if profile.get("precision"):
        options["forge_qwen21_precision"] = profile["precision"]
    options["sd_model_checkpoint"] = model["title"]
    api.call("POST", "options", options)
    actual = api.call("GET", "options")
    if canonical(actual.get("sd_model_checkpoint")) != canonical(profile["model"]):
        raise ValueError("Checkpoint option did not switch to the requested model")
    if actual.get("forge_preset") != profile["preset"]:
        raise ValueError("Forge preset did not follow the checkpoint")
    selected = actual.get("forge_additional_modules") or []
    if profile["preset"] != "xl" and sorted(map(canonical, selected)) != sorted(map(canonical, modules)):
        raise ValueError(f"Additional modules do not match {profile['id']}: {selected}")
    if profile.get("precision") and actual.get("forge_qwen21_precision") != profile["precision"]:
        raise ValueError("Qwen precision did not follow the checkpoint")
    return {key: actual.get(key) for key in ("sd_model_checkpoint", "forge_preset", "forge_qwen21_precision", "forge_additional_modules", "forge_unet_storage_dtype")}


def image_proof(data, expected_size=(512, 512)):
    from PIL import Image, ImageStat
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != expected_size:
            raise ValueError(f"Expected {expected_size} PNG; received {image.format} {image.size}")
        rgb = image.convert("RGB")
        stats = ImageStat.Stat(rgb)
        if max(stats.stddev) <= 1.0:
            raise ValueError("The image is blank or nearly uniform")
        if image.mode == "RGBA" and image.getchannel("A").getextrema()[1] == 0:
            raise ValueError("The image is fully transparent")
        return {"width": image.width, "height": image.height, "mode": image.mode,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "rgb_mean": stats.mean, "rgb_stddev": stats.stddev}


def verify_response(profile, response, path, request, reference=None):
    encoded = response.get("images") or []
    info = response.get("info")
    if isinstance(info, str):
        info = json.loads(info)
    if not isinstance(info, dict):
        raise ValueError("Forge returned no generation metadata")
    first = info.get("index_of_first_image", 0)
    if not isinstance(first, int) or not 0 <= first < len(encoded):
        raise ValueError("Forge returned no generated image")
    data = base64.b64decode(encoded[first].split(",", 1)[-1], validate=True)
    path.write_bytes(data)  # Keep even failed output for diagnosis.
    atomic_json(path.with_suffix(".generation.json"), info)
    proof = image_proof(data)
    if canonical(info.get("sd_model_name")) != canonical(profile["model"]):
        raise ValueError(f"Wrong model actually generated: {info.get('sd_model_name')!r}")
    if info.get("seed") != request["seed"] or info.get("all_seeds") != [request["seed"]]:
        raise ValueError(f"The output did not use the fixed seed: {info.get('all_seeds')}")
    if info.get("steps") != request["steps"] or (info.get("width"), info.get("height")) != (512, 512):
        raise ValueError("The requested step count or dimensions were not honored")
    if profile.get("precision"):
        actual_precision = (info.get("extra_generation_params") or {}).get("Qwen precision", "")
        if str(actual_precision).lower() != profile["precision"]:
            raise ValueError(f"Wrong Qwen backend precision in actual generation metadata: {actual_precision!r}")
    if reference and proof["pixel_sha256"] == image_proof(reference)["pixel_sha256"]:
        raise ValueError("Image-to-image returned the unchanged input")
    return {"image": path.name, **proof, "actual_model": info.get("sd_model_name"),
            "actual_model_hash": info.get("sd_model_hash"), "actual_vae": info.get("sd_vae_name"),
            "seed": info.get("seed"), "steps": info.get("steps"),
            "infotexts": info.get("infotexts"), "extra_generation_params": info.get("extra_generation_params")}


def contact_sheet(output, cases):
    from PIL import Image, ImageDraw
    images = [case for case in cases if case.get("image") and (output / case["image"]).is_file()]
    if not images:
        return
    sheet = Image.new("RGB", (4 * 256, ((len(images) + 3) // 4) * 292), "white")
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(images):
        x, y = index % 4 * 256, index // 4 * 292
        with Image.open(output / case["image"]) as image:
            thumbnail = image.convert("RGB").resize((256, 256))
            sheet.paste(thumbnail, (x, y + 36))
        draw.text((x + 5, y + 5), case["id"] + "\n" + case["status"], fill="black")
    sheet.save(output / "contact-sheet.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Perform real GPU generation for the selected matrix")
    mode.add_argument("--inventory-only", action="store_true", help="Read and verify API inventory; no settings or generation")
    mode.add_argument("--plan", action="store_true", help="Print the offline matrix (the default)")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=QA_ROOT / "results" / ("session-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]), help="Persistent Drive result directory; must not already exist")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="JSON profiles array; defaults to qa-models.json beside this script")
    parser.add_argument("--only", nargs="+", help="Run only these profile ids from the manifest")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per model switch/generation, without automatic retry")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-server-save", action="store_true", help="Only save images in this report directory")
    args = parser.parse_args(argv)
    try:
        profiles = load_profiles(args.manifest)
        unknown = set(args.only or ()) - {profile["id"] for profile in profiles}
        if unknown:
            raise ValueError("Unknown --only profiles: " + ", ".join(sorted(unknown)))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    selected = [profile for profile in profiles if not args.only or profile["id"] in args.only]
    expected_cases = sum(len(profile["modes"]) for profile in selected)
    if not args.run and not args.inventory_only:
        print(json.dumps({"mode": "plan_only_no_generation", "size": [512, 512], "seed": args.seed,
                          "matrix": selected, "expected_cases": expected_cases}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    api = API(args.base_url, args.timeout)
    report = {"started_at": now(), "status": "running", "scope": "Real Forge API/backend generation; visual quality and browser UI require inspection", "expected_cases": expected_cases, "profiles": selected, "cases": []}
    destination = args.output / "results.json"
    atomic_json(destination, report)
    initial = None
    try:
        if not api.idle(5):
            raise RuntimeError("Forge is busy. Use an exclusive idle QA session; no settings were changed")
        models = api.call("GET", "sd-models")
        modules = api.call("GET", "sd-modules")
        infos = api.call("GET", "script-info")
        atomic_json(args.output / "inventory.json", {"models": models, "modules": modules})
        if not modules and any(profile["modules"] for profile in selected):
            report["inventory_warning"] = "The sd-modules registry is empty. Start normal Forge with --api and no --share/--listen; avoid --nowebui for this fork, then check downloaded text encoders/VAEs. Profiles requiring those modules will be not_ready."
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            report["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else "nvidia-smi unavailable"
        except (OSError, subprocess.SubprocessError):
            report["gpu"] = "nvidia-smi unavailable"
        if args.run:
            initial = {key: value for key, value in api.call("GET", "options").items() if key in TOUCH_KEYS}
        all_pixels = set()
        for profile in selected:
            prepared = None
            try:
                model = resolve_model(profile, models)
                required = resolve_modules(profile, modules)
            except ValueError as error:
                for kind in profile["modes"]:
                    report["cases"].append({"id": profile["id"] + "-" + kind, "status": "not_ready", "error": str(error), "generation_attempted": False})
                atomic_json(destination, report)
                continue
            try:
                if args.run:
                    prepared = select_profile(api, profile, model, required)
                else:
                    report["cases"].append({"id": profile["id"], "status": "inventory_present_not_generated", "model": model, "modules": required})
                    continue
            except (RuntimeError, ValueError, urllib.error.URLError, TimeoutError) as error:
                for kind in profile["modes"]:
                    report["cases"].append({"id": profile["id"] + "-" + kind, "status": "blocked", "error": str(error)})
                atomic_json(destination, report)
                continue
            reference = None
            for kind in profile["modes"]:
                case = {"id": profile["id"] + "-" + kind, "status": "running", "started_at": now(), "selected_options": prepared}
                report["cases"].append(case)
                report["current_case"] = case["id"]
                atomic_json(destination, report)
                if kind == "img2img" and reference is None:
                    case.update(status="blocked", error="No verified text-to-image output to use as the reference")
                    continue
                request = {"prompt": TEXT_PROMPT if kind == "txt2img" else EDIT_PROMPT if profile["preset"] == "qwen" else IMAGE_PROMPT,
                           "negative_prompt": "" if profile["preset"] in {"qwen", "krea"} else NEGATIVE,
                           "seed": args.seed, "subseed": args.seed, "subseed_strength": 0,
                           "width": 512, "height": 512, "steps": profile["steps"], "cfg_scale": profile["cfg"],
                           "batch_size": 1, "n_iter": 1, "sampler_name": profile["sampler"], "scheduler": profile["scheduler"],
                           "restore_faces": False, "tiling": False, "styles": [], "script_name": None,
                           "send_images": True, "save_images": not args.no_server_save,
                           "alwayson_scripts": disable_optional_scripts(infos, kind == "img2img"),
                           "force_task_id": "task(forge-live-qa-" + uuid.uuid4().hex + ")"}
                if kind == "txt2img":
                    request["enable_hr"] = False
                else:
                    request.update(init_images=[base64.b64encode(reference).decode()], denoising_strength=0.65, resize_mode=0, include_init_images=False)
                request_summary = {key: value for key, value in request.items() if key != "init_images"}
                if reference:
                    request_summary["reference_sha256"] = hashlib.sha256(reference).hexdigest()
                atomic_json(args.output / (case["id"] + ".request.json"), request_summary)
                print(f"RUN {case['id']}: 512x512, {profile['steps']} steps, seed {args.seed}", flush=True)
                started = time.monotonic()
                try:
                    response = api.call("POST", kind, request)
                    image = args.output / (case["id"] + ".png")
                    case["image"] = image.name
                    proof = verify_response(profile, response, image, request, reference)
                    if proof["pixel_sha256"] in all_pixels:
                        raise ValueError("This result duplicates an earlier test's exact pixels")
                    all_pixels.add(proof["pixel_sha256"])
                    case.update(proof, status="pass", visual_review="pending")
                    if kind == "txt2img":
                        reference = image.read_bytes()
                except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
                    case.update(status="fail", error=str(error))
                    # Never launch the next job while a timed-out request might
                    # still be generating on the server.
                    api.call("POST", "interrupt", {}, timeout=15)
                    if not api.idle(120):
                        raise RuntimeError("Timed-out job did not stop; matrix halted to avoid overlapping requests")
                except (RuntimeError, ValueError, KeyError) as error:
                    case.update(status="fail", error=str(error))
                finally:
                    case["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    case["finished_at"] = now()
                    print(case["status"].upper() + " " + case["id"], flush=True)
                    atomic_json(destination, report)
        passed = sum(case["status"] == "pass" for case in report["cases"])
        report["status"] = ("generated_all_pending_visual_review" if passed == expected_cases else "incomplete") if args.run else "inventory_only_not_generated"
    except BaseException as error:
        report.update(status="aborted", error=str(error))
        if args.run and report.get("current_case"):
            try:
                api.call("POST", "interrupt", {}, timeout=15)
            except Exception:
                pass
        if isinstance(error, KeyboardInterrupt):
            print("Interrupted; partial evidence retained.", flush=True)
    finally:
        if initial is not None:
            try:
                if not api.idle(120):
                    raise RuntimeError("Server is still busy; restore deferred")
                # Keep the checkpoint last so its loading hook sees restored profiles.
                checkpoint = initial.pop("sd_model_checkpoint", None)
                if checkpoint:
                    initial["sd_model_checkpoint"] = checkpoint
                api.call("POST", "options", initial)
                report["settings_restore"] = "complete"
            except Exception as error:
                report["settings_restore"] = "failed: " + str(error)
        report["finished_at"] = now()
        atomic_json(destination, report)
        try:
            contact_sheet(args.output, report["cases"])
        except Exception as error:
            report["contact_sheet_error"] = str(error)
            atomic_json(destination, report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "results": str(destination)}, indent=2), flush=True)
    if args.inventory_only:
        return int(report["status"] != "inventory_only_not_generated" or any(case["status"] in {"blocked", "not_ready"} for case in report["cases"]))
    return int(report["status"] != "generated_all_pending_visual_review" or str(report.get("settings_restore", "")).startswith("failed"))


if __name__ == "__main__":
    raise SystemExit(main())
