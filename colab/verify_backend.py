#!/usr/bin/env python3
"""Direct execution proof for Forge VAEs, VAE preview weights and CodeFormer.

Run in a NEW process with the persistent Forge Python interpreter. Default --plan
does not import Torch, load weights, or run inference. --run uses CPU by default:
  forge/venv/bin/python forge/colab/verify_backend.py --run --input-image PERSON.png

All checkpoints are loaded by real Forge backend classes. VAE previews use
fixed synthetic latent tensors; this proves network execution, not live UI preview
routing. CodeFormer runs the normal face detection, restoration and paste-back
path, with successful forward hooks and captured swallowed errors. No network
downloads are performed by the harness. Existing downloaded dependencies are
required. Results default to Drive/ForgeColab/qa/results. Do not run this inside
the active Forge server process. Actual visual quality still requires inspection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import uuid


DRIVE_ROOT = Path("/content/drive/MyDrive/ForgeColab")
CASE_FILES = {
    "vae-preview-sd15": "VAE-approx/model.pt",
    "vae-preview-sdxl": "VAE-approx/vaeapprox-sdxl.pt",
    "codeformer": "Codeformer/codeformer-v0.1.0.pth",
    "vae-flux-ae": "VAE/ae.safetensors",
    "vae-ppp-sdxl": "VAE/pppanimixVAE_il.safetensors",
}
VAE_CONFIGS = {"vae-flux-ae": "black-forest-labs/FLUX.1-schnell",
               "vae-ppp-sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}


class NotReady(ValueError):
    pass


class Inconclusive(ValueError):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(models, names):
    absent = [name for name in names if not (models / name).is_file()]
    if absent:
        raise NotReady("Missing downloaded model files: " + ", ".join(absent))


def tensor_summary(value):
    import torch
    tensors = []
    def visit(item):
        if isinstance(item, torch.Tensor):
            detached = item.detach()
            if not detached.numel() or not bool(torch.isfinite(detached).all()):
                raise ValueError("A backend forward returned an empty or non-finite tensor")
            tensors.append({"shape": list(detached.shape), "dtype": str(detached.dtype),
                            "device": str(detached.device), "finite": True,
                            "min": float(detached.min()), "max": float(detached.max())})
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
    visit(value)
    if not tensors:
        raise ValueError("A backend forward returned no tensor output")
    return tensors


@contextmanager
def trace_forwards(named_modules):
    traces = {name: [] for name in named_modules}
    handles = []
    try:
        for name, module in named_modules.items():
            def completed(_module, _inputs, output, label=name):
                traces[label].append(tensor_summary(output))
            handles.append(module.register_forward_hook(completed))
        yield traces
    finally:
        for handle in handles:
            handle.remove()


def require_completed_forwards(traces, minimum=1):
    absent = [name for name, calls in traces.items() if len(calls) < minimum]
    if absent:
        raise Inconclusive("No completed model inference recorded for: " + ", ".join(absent))


def prepare_forge(args):
    if "modules.shared" in sys.modules:
        raise RuntimeError("Run this script in a new process, not inside a running Forge backend")
    root = args.forge_root.resolve()
    if not (root / "modules/sd_vae_approx.py").is_file():
        raise NotReady("Forge source modules are missing")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "packages_3rdparty"))
    os.environ["COMMANDLINE_ARGS"] = ""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    sys.argv = [str(root / "launch.py"), "--models-dir", str(args.models_dir),
                "--ui-settings-file", str(args.output / "unused-settings.json"),
                "--skip-version-check", "--skip-torch-cuda-test", "--disable-xformers"]
    if args.device == "cpu":
        sys.argv.append("--always-cpu")
    from modules import shared, options, shared_options, devices
    # Standard option objects, with defaults only; no personal settings are read
    # or written and no diffusion checkpoint is loaded for these small networks.
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.opts.face_restoration_unload = False
    return devices


def verify_vae(models, filename, output, identifier, device):
    import torch
    from PIL import Image
    from modules.sd_vae_approx import VAEApprox
    path = models / filename
    state = torch.load(path, map_location="cpu", weights_only=True)
    channels = int(state["conv1.weight"].shape[1])
    model = VAEApprox(latent_channels=channels).eval()
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(902104)
    latent = torch.randn((1, channels, 16, 16), generator=generator).to(device)
    layers = {name: getattr(model, name) for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8")}
    with trace_forwards(layers) as trace, torch.inference_mode():
        zero = model(torch.zeros_like(latent))
        preview = model(latent)
    require_completed_forwards(trace, minimum=2)
    tensor_summary(preview)
    if tuple(preview.shape) != (1, 3, 32, 32):
        raise ValueError("Unexpected preview output shape: " + str(tuple(preview.shape)))
    if torch.equal(zero, preview) or float(preview.std()) <= 1e-6:
        raise ValueError("Preview network output did not respond to its latent input")
    image_array = ((preview[0].float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    image_path = output / (identifier + ".png")
    Image.fromarray(image_array, mode="RGB").save(image_path)
    return {"status": "pass", "execution": "actual Forge VAEApprox forwards on fixed synthetic latents",
            "checkpoint": str(path), "checkpoint_sha256": digest_file(path), "source_class": "modules.sd_vae_approx.VAEApprox",
            "input_shape": list(latent.shape), "output_shape": list(preview.shape), "forward_trace": trace,
            "output_tensor_sha256": hashlib.sha256(preview.detach().cpu().numpy().tobytes()).hexdigest(),
            "image": image_path.name, "visual_review": "pending", "scope_limit": "Direct weight execution; no claim that live UI selected this preview model"}


def verify_autoencoder(models, identifier, root, output, device, input_image=None):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from backend.nn.vae import IntegratedAutoencoderKL
    path = models / CASE_FILES[identifier]
    config_path = root / "backend/huggingface" / VAE_CONFIGS[identifier] / "vae/config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = inspect.signature(IntegratedAutoencoderKL.__init__).parameters
    config = {key: value for key, value in raw_config.items() if key in parameters}
    model = IntegratedAutoencoderKL(**config).eval()
    state = load_file(str(path), device="cpu")
    model.load_state_dict(state, strict=True)
    tensor_count = len(state)
    del state
    model = model.to(device=device, dtype=torch.float32)
    if input_image is not None:
        if not input_image.is_file():
            raise NotReady("The input image is missing")
        with Image.open(input_image) as image:
            source = ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), (256, 256), color="white")
        fixture = "provided image, padded to 256px"
    else:
        # A deterministic multi-channel fixture checks encode/decode without
        # downloading or bundling a photo. It is not a generated image claim.
        y, x = np.mgrid[:256, :256]
        pixels = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)
        pixels[48:112, 64:160] = (180, 30, 60)
        pixels[160:216, 96:208] = (20, 160, 200)
        source = Image.fromarray(pixels, mode="RGB")
        fixture = "deterministic RGB gradient and colored rectangles"
    source.save(output / (identifier + "-input.png"))
    image_tensor = torch.from_numpy(np.array(source).astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
    named = {"encoder": model.encoder, "decoder": model.decoder}
    if model.quant_conv is not None:
        named["quant_conv"] = model.quant_conv
    if model.post_quant_conv is not None:
        named["post_quant_conv"] = model.post_quant_conv
    with trace_forwards(named) as trace, torch.inference_mode():
        latent = model.encode(image_tensor, regulation=lambda posterior: posterior.mode())
        # Exercise the checkpoint-specific latent scale/shift route too.
        prepared_latent = model.process_in(latent)
        restored_latent = model.process_out(prepared_latent)
        if not torch.allclose(latent, restored_latent, atol=1e-5, rtol=1e-5):
            raise ValueError("Latent scale/shift round-trip changed the encoded latent")
        reconstruction = model.decode(restored_latent)
    require_completed_forwards(trace)
    tensor_summary(latent)
    tensor_summary(reconstruction)
    channels = int(config["latent_channels"])
    if tuple(latent.shape) != (1, channels, 32, 32) or tuple(reconstruction.shape) != tuple(image_tensor.shape):
        raise ValueError("Unexpected native VAE latent or decoded image dimensions")
    if float(latent.std()) <= 1e-6 or float(reconstruction.std()) <= 1e-6:
        raise ValueError("The native VAE returned an effectively constant latent or reconstruction")
    normalized = ((reconstruction.float().cpu() + 1) / 2).clamp(0, 1)
    original = ((image_tensor.float().cpu() + 1) / 2).clamp(0, 1)
    mse = float(torch.mean((normalized - original) ** 2))
    # This is a deliberately low reconstruction floor, not a visual-quality
    # score. A random, swapped or broken decoder must not pass on finiteness alone.
    if mse >= 0.1:
        raise ValueError("VAE reconstruction error exceeds the basic correctness floor: " + str(mse))
    pixels = (normalized[0] * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels, mode="RGB").save(output / (identifier + ".png"))
    return {"status": "pass", "execution": "actual Forge IntegratedAutoencoderKL strict-load, mean encode and decode",
            "checkpoint": str(path), "checkpoint_sha256": digest_file(path), "loaded_tensors": tensor_count,
            "config_source": str(config_path), "config_sha256": digest_file(config_path),
            "source_class": "backend.nn.vae.IntegratedAutoencoderKL", "fixture": fixture,
            "input_shape": list(image_tensor.shape), "latent_shape": list(latent.shape),
            "output_shape": list(reconstruction.shape), "reconstruction_mse_0_to_1": mse,
            "forward_trace": trace, "latent_sha256": hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest(),
            "output_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "image": identifier + ".png",
            "visual_review": "pending", "scope_limit": "Native component inference; no claim that a generation preset selected this external VAE"}


def verify_codeformer(models, input_image, output):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from modules import codeformer_model, errors, face_restoration_utils, modelloader
    dependencies = [CASE_FILES["codeformer"], "GFPGAN/detection_Resnet50_Final.pth", "GFPGAN/parsing_parsenet.pth"]
    require_files(models, dependencies)
    if input_image is None or not input_image.is_file():
        raise NotReady("CodeFormer requires --input-image with a visible face")
    with Image.open(input_image) as image:
        source = ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), (512, 512), color="white")
    source.save(output / "codeformer-input.png")
    source_array = np.asarray(source)
    face_restoration_utils.patch_facexlib(str(models / "GFPGAN"))
    restorer = codeformer_model.FaceRestorerCodeFormer(str(models / "Codeformer"))
    reports, loaded = [], []
    original_report = errors.report
    original_loader = modelloader.load_spandrel_model
    def report_error(message, *args, **kwargs):
        reports.append(str(message))
        return original_report(message, *args, **kwargs)
    def record_loader(path, *args, **kwargs):
        if Path(path).resolve() != (models / CASE_FILES["codeformer"]).resolve():
            raise ValueError("CodeFormer loader selected an unexpected checkpoint: " + str(path))
        descriptor = original_loader(path, *args, **kwargs)
        loaded.append({"path": str(path), "architecture": descriptor.architecture.id})
        return descriptor
    errors.report = report_error
    modelloader.load_spandrel_model = record_loader
    trace = {}
    try:
        restorer.net = restorer.load_net()
        helper = restorer.face_helper
        with trace_forwards({"codeformer": restorer.net, "retinaface": helper.face_det, "face_parser": helper.face_parse}) as trace, torch.inference_mode():
            restored = restorer.restore(source_array, w=0.5)
        # Persist the output and forward trace even if a later assertion fails.
        Image.fromarray(restored).save(output / "codeformer.png")
        atomic_json(output / "codeformer-forward-trace.json", {"loaded": loaded, "forward_trace": trace, "reported_errors": reports})
        if reports:
            raise ValueError("The normal restoration path swallowed errors: " + "; ".join(reports))
        require_completed_forwards(trace)
        if restored.shape != source_array.shape or not np.isfinite(restored).all():
            raise ValueError("Face restoration returned invalid pixels or dimensions")
        if np.array_equal(restored, source_array):
            raise Inconclusive("Networks ran but pixels were unchanged; choose a better face fixture")
        return {"status": "pass", "execution": "actual Forge FaceRestorerCodeFormer.restore with completed network hooks",
                "image": "codeformer.png", "visual_review": "pending", "shape": list(restored.shape),
                "input_pixel_sha256": hashlib.sha256(source_array.tobytes()).hexdigest(),
                "output_pixel_sha256": hashlib.sha256(restored.tobytes()).hexdigest(), "loaded": loaded,
                "completed_forward_counts": {name: len(calls) for name, calls in trace.items()},
                "reported_errors": reports, "checkpoints": [{"path": name, "sha256": digest_file(models / name)} for name in dependencies]}
    finally:
        errors.report = original_report
        modelloader.load_spandrel_model = original_loader
        if trace:
            atomic_json(output / "codeformer-forward-trace.json", {"loaded": loaded, "forward_trace": trace, "reported_errors": reports})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--plan", action="store_true")
    parser.add_argument("--forge-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models-dir", type=Path, default=DRIVE_ROOT / "models")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cases", nargs="+", choices=list(CASE_FILES), default=list(CASE_FILES))
    parser.add_argument("--output", type=Path, default=DRIVE_ROOT / "qa/results" / ("backend-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]))
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"mode": "plan_only_no_inference", "device": args.device, "cases": {name: CASE_FILES[name] for name in args.cases}}, indent=2))
        return 0
    args.models_dir, args.output = args.models_dir.resolve(), args.output.resolve()
    if args.input_image:
        args.input_image = args.input_image.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "device": args.device, "cases": [], "scope": "Actual backend model forward execution; visual review and live UI routing are separate"}
    report_path = args.output / "results.json"
    atomic_json(report_path, report)
    try:
        devices = prepare_forge(args)
        if devices.device.type != args.device:
            raise RuntimeError("Forge selected an unexpected device: " + str(devices.device))
        for name in args.cases:
            case = {"id": name, "status": "running"}
            report["cases"].append(case)
            start = time.monotonic()
            try:
                require_files(args.models_dir, [CASE_FILES[name]])
                if name == "codeformer":
                    case.update(verify_codeformer(args.models_dir, args.input_image, args.output))
                elif name in VAE_CONFIGS:
                    case.update(verify_autoencoder(args.models_dir, name, args.forge_root.resolve(), args.output, devices.device, args.input_image))
                else:
                    case.update(verify_vae(args.models_dir, CASE_FILES[name], args.output, name, devices.device))
            except NotReady as error:
                case.update(status="not_ready", error=str(error))
            except Inconclusive as error:
                case.update(status="inconclusive", error=str(error))
            except Exception as error:
                case.update(status="fail", error=str(error))
            finally:
                case["elapsed_seconds"] = round(time.monotonic() - start, 3)
                atomic_json(report_path, report)
                print(case["status"].upper() + " " + name, flush=True)
        report["status"] = "backend_forwards_pass_pending_visual_review" if all(case["status"] == "pass" for case in report["cases"]) else "incomplete"
    except BaseException as error:
        report.update(status="aborted", error=str(error))
    finally:
        atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "results": str(report_path)}, indent=2))
    return int(report["status"] != "backend_forwards_pass_pending_visual_review")


if __name__ == "__main__":
    raise SystemExit(main())
