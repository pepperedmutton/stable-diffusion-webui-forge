#!/usr/bin/env python3
"""Real auxiliary inference QA against an exclusive localhost Forge session.

Default --plan makes no requests and runs no inference. Example after installing
the weights and starting normal Forge with --api and no --share/--listen:
  %run /content/drive/MyDrive/ForgeColab/forge/colab/verify_auxiliary.py --run \
      --input-image /content/drive/MyDrive/ForgeColab/qa/fixtures/person.png

Use a generated adult or a consented adult image with a visible face, full body,
and visible hands. No image is bundled or downloaded by this script. The resized
input and actual outputs are saved in a new persistent Drive/qa/results session.
Pose detection without enough hand/face landmarks is inconclusive for those
submodels. Pixel/metadata checks do not prove aesthetic quality or pose fidelity;
review the contact sheet. Optional caption/restoration cases require their full
dependencies to be installed in Drive before use. No inference retry is automatic.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
import uuid


spec = importlib.util.spec_from_file_location("forge_model_qa", Path(__file__).with_name("verify_models.py"))
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

CASES = {
    "openpose-full": "Run body, hand and face pose detectors; save real pose image and landmark counts",
    "sdxl-baseline": "Generate a fixed-seed WAI v150 baseline with all ControlNet units disabled",
    "openpose-control": "Generate with the actual detected pose and SDXL OpenPose ControlNet; compare to baseline",
    "textual-inversion": "Generate with all four SDXL embeddings; require all four names in actual TI metadata",
    "realesrgan-anime6b": "Run neural 4x upscaling; reject interpolation-only fallback",
    "faceid-plusv2": "Generate with the actual FaceID adapter, companion LoRA, CLIP-H and InsightFace; require active metadata",
    "codeformer": "Optional: restore a visible face; unchanged pixels are inconclusive",
    "interrogate-deepdanbooru": "Optional: require nonempty output tags from the actual image",
    "interrogate-blip-clip": "Optional: require a caption using BLIP and the separate OpenAI CLIP ViT-L/14",
}
CORE_CASES = list(CASES)[:6]
LIMITATIONS = [
    {"id": "legacy-qwen-image", "status": "unsupported", "generation_attempted": False,
     "reason": "Two legacy Qwen weights are archival files; this fork's current Qwen backend is Image 2.1, not those legacy checkpoints."},
    {"id": "vae-preview-networks", "status": "not_covered", "generation_attempted": False,
     "reason": "VAE preview approximators have no direct inference API. Final image generation does not prove preview model execution; a separate backend trace is required."},
    {"id": "private-loras", "status": "excluded_by_request", "generation_attempted": False,
     "reason": "The three locally trained LoRAs are not publicly downloadable and are excluded under the no-upload instruction."},
]
EMBEDDINGS = ("lazyhand", "lazyneg", "lazypos", "lazywet")


class NotReady(ValueError):
    pass


class Inconclusive(ValueError):
    pass


class AuxiliaryAPI(qa.API):
    def control(self, method, endpoint, payload=None):
        if endpoint not in {"model_list", "module_list", "detect"}:
            raise ValueError("Unknown ControlNet API endpoint")
        headers = {"Content-Type": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + "/controlnet/" + endpoint, body, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # Avoid logging server headers or a response containing credentials.
            raise RuntimeError(f"ControlNet {endpoint} returned HTTP {error.code}") from None


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def encode(data):
    return base64.b64encode(data).decode()


def decode(value):
    return base64.b64decode(value.split(",", 1)[-1], validate=True)


def require_files(drive, relatives):
    missing = [relative for relative in relatives if not (drive / relative).is_file()]
    if missing:
        raise NotReady("Required Drive files are missing: " + ", ".join(missing))


def pose_counts(poses):
    counts = {"body": 0, "left_hand": 0, "right_hand": 0, "face": 0, "people": 0}
    keys = {"body": "pose_keypoints_2d", "left_hand": "hand_left_keypoints_2d",
            "right_hand": "hand_right_keypoints_2d", "face": "face_keypoints_2d"}
    for frame in poses:
        if isinstance(frame, str):
            frame = json.loads(frame)
        for person in frame.get("people", []):
            counts["people"] += 1
            for label, key in keys.items():
                points = person.get(key) or []
                counts[label] += sum(float(points[index]) > 0 for index in range(2, len(points), 3))
    return counts


def unit_scripts(infos, unit=None):
    scripts = qa.disable_optional_scripts(infos, False)
    matches = [info for info in infos if info.get("is_alwayson") and not info.get("is_img2img")
               and str(info.get("name")).casefold() == "controlnet"]
    if len(matches) != 1:
        raise NotReady("ControlNet always-on script is not registered exactly once")
    info = matches[0]
    units = [{"enabled": False} for _ in range(max(1, len(info.get("args") or [])))]
    if unit:
        units[0] = unit
    scripts[info["name"]] = {"args": units}
    return scripts


def require_changed(before, after):
    if before == after:
        raise Inconclusive("Output pixels are unchanged; this does not prove the requested model ran")


def require_neural_upscale(input_data, output_data):
    from PIL import Image
    with Image.open(io.BytesIO(input_data)) as source, Image.open(io.BytesIO(output_data)) as result:
        actual = result.convert("RGB").tobytes()
        for mode in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS):
            if actual == source.convert("RGB").resize(result.size, mode).tobytes():
                raise ValueError("Upscaling returned interpolation-only pixels; neural execution is not proven")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--plan", action="store_true")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--drive-root", type=Path, default=qa.DRIVE_BASE)
    parser.add_argument("--output", type=Path, default=qa.QA_ROOT / "results" / ("auxiliary-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]))
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=CORE_CASES)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    selected = set(args.cases)
    if "openpose-control" in selected:
        selected.update(("openpose-full", "sdxl-baseline"))
    if selected.intersection({"textual-inversion", "faceid-plusv2"}):
        selected.add("sdxl-baseline")
    ordered = [name for name in CASES if name in selected]
    if not args.run:
        print(json.dumps({"mode": "plan_only_no_inference", "cases": {name: CASES[name] for name in ordered}, "limitations": LIMITATIONS}, indent=2))
        return 0
    if args.input_image is None or not args.input_image.is_file():
        parser.error("--run requires --input-image with visible body, face and hands")
    from PIL import Image, ImageOps
    api = AuxiliaryAPI(args.base_url, args.timeout)
    args.output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.input_image) as original:
        source = ImageOps.pad(ImageOps.exif_transpose(original).convert("RGB"), (512, 512), color="white")
    source_data = png_bytes(source)
    (args.output / "input.png").write_bytes(source_data)
    qa.image_proof(source_data)
    report = {"status": "running", "started_at": qa.now(), "scope": "Actual auxiliary inference, with output evidence; visual review remains required", "cases": [], "limitations": copy.deepcopy(LIMITATIONS)}
    report_path = args.output / "results.json"
    qa.atomic_json(report_path, report)
    initial = None
    context = {}
    profile = next(profile for profile in qa.load_profiles(qa.DEFAULT_MANIFEST) if profile["id"] == "xl-wai-v150")
    try:
        if not api.idle(5):
            raise RuntimeError("Forge is busy; use an exclusive idle QA session")
        models = api.call("GET", "sd-models")
        infos = api.call("GET", "script-info")
        for name in ordered:
            case = {"id": name, "status": "running", "started_at": qa.now(), "inference_attempted": False}
            report["cases"].append(case)
            report["current_case"] = name
            qa.atomic_json(report_path, report)
            start = time.monotonic()
            try:
                if name == "openpose-full":
                    require_files(args.drive_root, ["models/ControlNetPreprocessor/openpose/" + file for file in ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth")])
                    modules = api.control("GET", "module_list").get("module_list", [])
                    if "openpose_full" not in modules:
                        raise NotReady("openpose_full is not registered")
                    case["inference_attempted"] = True
                    response = api.control("POST", "detect", {"controlnet_module": "openpose_full", "controlnet_input_images": [encode(source_data)], "controlnet_processor_res": 512})
                    counts = pose_counts(response.get("poses", []))
                    qa.atomic_json(args.output / "openpose-poses.json", response.get("poses", []))
                    case["landmark_counts"] = counts
                    data = decode(response["images"][0])
                    (args.output / "openpose-full.png").write_bytes(data)
                    case["image"] = "openpose-full.png"
                    if counts["body"] < 6:
                        raise Inconclusive("No usable body detected; choose a clear full-body reference image")
                    case.update(qa.image_proof(data))
                    context["pose"] = data
                    if not counts["left_hand"] or not counts["right_hand"] or counts["face"] < 10:
                        raise Inconclusive("Body pose is usable, but the fixture did not exercise both hand models and face landmarks")
                elif name in {"sdxl-baseline", "openpose-control", "textual-inversion", "faceid-plusv2"}:
                    if initial is None:
                        try:
                            model = qa.resolve_model(profile, models)
                        except ValueError as error:
                            raise NotReady(str(error)) from None
                        initial = {key: value for key, value in api.call("GET", "options").items() if key in qa.TOUCH_KEYS or key == "lora_add_hashes_to_infotext"}
                        qa.select_profile(api, profile, model, [])
                    request = {"prompt": "A fully clothed adult standing, both arms stretched sideways, full body, visible hands, plain gray studio background, detailed illustration",
                               "negative_prompt": "text, watermark, blurry", "seed": qa.SEED, "subseed": qa.SEED,
                               "subseed_strength": 0, "width": 512, "height": 512, "steps": 20,
                               "cfg_scale": 6, "sampler_name": "Euler a", "scheduler": "Automatic",
                               "batch_size": 1, "n_iter": 1, "enable_hr": False, "restore_faces": False,
                               "tiling": False, "styles": [], "save_images": False, "send_images": True,
                               "alwayson_scripts": unit_scripts(infos)}
                    if name != "sdxl-baseline" and "baseline" not in context:
                        raise NotReady("No verified baseline output is available")
                    if name == "openpose-control":
                        if "pose" not in context:
                            raise NotReady("No usable pose was detected")
                        require_files(args.drive_root, ["models/ControlNet/openpose.safetensors"])
                        cn_models = api.control("GET", "model_list").get("model_list", [])
                        matches = [model for model in cn_models if qa.canonical(model) == "openpose"]
                        if len(matches) != 1:
                            raise NotReady("The exact OpenPose ControlNet model is not registered once")
                        request["alwayson_scripts"] = unit_scripts(infos, {"enabled": True, "module": "None", "model": matches[0], "weight": 1, "image": encode(context["pose"]), "resize_mode": "Just Resize", "processor_res": 512, "guidance_start": 0, "guidance_end": 1, "control_mode": "Balanced", "save_detected_map": False})
                        case["control_model"] = matches[0]
                    elif name == "textual-inversion":
                        require_files(args.drive_root, ["embeddings/" + embedding + ".safetensors" for embedding in EMBEDDINGS])
                        api.call("POST", "refresh-embeddings", {})
                        request["prompt"] += ", lazypos"
                        request["negative_prompt"] += ", lazyhand, lazyneg, lazywet"
                    elif name == "faceid-plusv2":
                        adapter = "ip-adapter-faceid-plusv2_sdxl"
                        companion = adapter + "_lora"
                        require_files(args.drive_root, ["models/ControlNet/" + adapter + ".bin", "models/Lora/" + companion + ".safetensors", "models/ControlNetPreprocessor/CLIP-ViT-H-14.safetensors"] + ["models/insightface/models/buffalo_l/" + file for file in ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")])
                        module = "InsightFace+CLIP-H (IPAdapter)"
                        if module not in api.control("GET", "module_list").get("module_list", []):
                            raise NotReady("The FaceID preprocessor is not registered")
                        matches = [model for model in api.control("GET", "model_list").get("model_list", []) if qa.canonical(model) == adapter]
                        if len(matches) != 1:
                            raise NotReady("The actual FaceID adapter is not registered exactly once; the companion LoRA alone is insufficient")
                        api.call("POST", "refresh-loras", {})
                        if companion not in {model.get("name") for model in api.call("GET", "loras")}:
                            raise NotReady("The FaceID companion LoRA is not available in the LoRA registry")
                        api.call("POST", "options", {"lora_add_hashes_to_infotext": True})
                        request["prompt"] += " <lora:" + companion + ":0.6>"
                        request["alwayson_scripts"] = unit_scripts(infos, {"enabled": True, "module": module, "model": matches[0], "weight": 1, "image": encode(source_data), "resize_mode": "Just Resize", "processor_res": 512, "guidance_start": 0, "guidance_end": 1, "control_mode": "Balanced", "save_detected_map": False})
                        case["control_model"] = matches[0]
                    # Keep request settings without any image payload in evidence.
                    summary = copy.deepcopy(request)
                    for script in summary["alwayson_scripts"].values():
                        for unit in script.get("args", []):
                            if isinstance(unit, dict) and "image" in unit:
                                unit["image"] = "[saved input.png]" if name == "faceid-plusv2" else "[saved openpose-full.png]"
                    qa.atomic_json(args.output / (name + ".request.json"), summary)
                    case["inference_attempted"] = True
                    response = api.call("POST", "txt2img", request)
                    proof = qa.verify_response(profile, response, args.output / (name + ".png"), request)
                    case.update(proof)
                    extra = proof.get("extra_generation_params") or {}
                    if name == "sdxl-baseline":
                        context["baseline"] = proof["pixel_sha256"]
                    else:
                        require_changed(context["baseline"], proof["pixel_sha256"])
                    if name in {"openpose-control", "faceid-plusv2"}:
                        evidence = json.dumps({key: value for key, value in extra.items() if key.lower().startswith("controlnet")})
                        if qa.canonical(case["control_model"]) not in evidence.casefold():
                            raise ValueError("The real output metadata did not record the requested ControlNet/adapter")
                    if name == "faceid-plusv2" and companion not in str(extra.get("Lora hashes", "")):
                        raise ValueError("The FaceID companion LoRA is missing from actual output metadata")
                    if name == "textual-inversion":
                        used = {value.strip() for value in str(extra.get("TI", "")).split(",")}
                        if not set(EMBEDDINGS).issubset(used):
                            raise ValueError("Not all four embeddings appear in actual TI metadata: " + str(sorted(used)))
                elif name in {"realesrgan-anime6b", "codeformer"}:
                    if name == "realesrgan-anime6b":
                        require_files(args.drive_root, ["models/RealESRGAN/RealESRGAN_x4plus_anime_6B.pth"])
                        target = "R-ESRGAN 4x+ Anime6B"
                        if target not in [model["name"] for model in api.call("GET", "upscalers")]:
                            raise NotReady("The Anime6B upscaler is not enabled in Forge")
                        input_data = png_bytes(source.resize((128, 128), Image.Resampling.LANCZOS))
                        request = {"image": encode(input_data), "upscaler_1": target, "upscaling_resize": 4, "codeformer_visibility": 0, "gfpgan_visibility": 0}
                    else:
                        require_files(args.drive_root, ["models/Codeformer/codeformer-v0.1.0.pth", "models/GFPGAN/detection_Resnet50_Final.pth", "models/GFPGAN/parsing_parsenet.pth"])
                        input_data = source_data
                        request = {"image": encode(input_data), "upscaler_1": "None", "upscaling_resize": 1, "codeformer_visibility": 1, "codeformer_weight": 0.5, "gfpgan_visibility": 0}
                    case["inference_attempted"] = True
                    response = api.call("POST", "extra-single-image", request)
                    data = decode(response["image"])
                    (args.output / (name + ".png")).write_bytes(data)
                    case.update(qa.image_proof(data), image=name + ".png", html_info=response.get("html_info"))
                    if name == "realesrgan-anime6b":
                        require_neural_upscale(input_data, data)
                    else:
                        require_changed(qa.image_proof(input_data)["pixel_sha256"], case["pixel_sha256"])
                        raise Inconclusive("CodeFormer returned changed pixels, but the backend can catch restoration exceptions and paste an unrestored face. A clean backend inference log is required to confirm restoration.")
                else:
                    model = "deepdanbooru" if name.endswith("deepdanbooru") else "clip"
                    paths = ["models/torch_deepdanbooru/model-resnet_custom_v3.pt"] if model == "deepdanbooru" else ["models/BLIP/model_base_caption_capfilt_large.pth", "models/CLIP/ViT-L-14.pt"]
                    require_files(args.drive_root, paths)
                    case["inference_attempted"] = True
                    response = api.call("POST", "interrogate", {"image": encode(source_data), "model": model})
                    caption = str(response.get("caption") or "").strip()
                    if not caption or "error" in caption.casefold() or "exception" in caption.casefold():
                        raise ValueError("The interrogator did not return a usable caption")
                    case["caption"] = caption
                case.update(status="pass", visual_review="pending")
            except NotReady as error:
                case.update(status="not_ready", error=str(error))
            except Inconclusive as error:
                case.update(status="inconclusive", error=str(error))
            except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
                case.update(status="fail", error=str(error))
                # Extras and /controlnet/detect do not reliably expose job_count
                # or honor /interrupt. Stop the matrix on every transport error.
                raise RuntimeError("Transport failed; stop the server or confirm the request ended before any further QA") from error
            except (RuntimeError, ValueError, KeyError, IndexError) as error:
                detail = str(error).casefold()
                if name == "faceid-plusv2" and "no module named" in detail and any(module in detail for module in ("insightface", "onnxruntime")):
                    case.update(status="not_ready", error=str(error))
                elif name == "faceid-plusv2" and "no face" in detail:
                    case.update(status="inconclusive", error=str(error))
                else:
                    case.update(status="fail", error=str(error))
            finally:
                case.update(finished_at=qa.now(), elapsed_seconds=round(time.monotonic() - start, 3))
                qa.atomic_json(report_path, report)
                try:
                    qa.contact_sheet(args.output, report["cases"])
                except (OSError, ValueError) as error:
                    report["contact_sheet_error"] = str(error)
                print(case["status"].upper() + " " + name, flush=True)
        report["status"] = "selected_cases_pass_pending_visual_review" if all(case["status"] == "pass" for case in report["cases"]) else "incomplete"
    except BaseException as error:
        report.update(status="aborted", error=str(error))
    finally:
        if initial is not None and report["status"] != "aborted":
            try:
                if not api.idle(120):
                    raise RuntimeError("Forge is busy; settings restore deferred")
                checkpoint = initial.pop("sd_model_checkpoint", None)
                if checkpoint:
                    initial["sd_model_checkpoint"] = checkpoint
                api.call("POST", "options", initial)
                report["settings_restore"] = "complete"
            except Exception as error:
                report["settings_restore"] = "failed: " + str(error)
        elif initial is not None:
            report["settings_restore"] = "deferred_after_abort_to_avoid_overlapping_requests"
        report["finished_at"] = qa.now()
        qa.atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "results": str(report_path), "limitations": report["limitations"]}, indent=2))
    return int(report["status"] != "selected_cases_pass_pending_visual_review" or str(report.get("settings_restore", "")).startswith("failed"))


if __name__ == "__main__":
    raise SystemExit(main())
