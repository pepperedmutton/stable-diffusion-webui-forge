"""Isolated, persistent Qwen-Image 2.1 inference worker.

Run with the Python created by scripts/setup_qwen21_runtime.py. Standard output
is a JSON-lines protocol; all library output and tracebacks go to standard error.
The Forge process may terminate this worker to interrupt loading or generation.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
from pathlib import Path
import secrets
import sys
import traceback
from typing import Any


MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DIFFUSERS_REVISION = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
MAX_REFERENCE_IMAGES = 10


def runtime_dependency_errors():
    """Inspect both isolated and inherited distributions (uv ignores .pth)."""
    import importlib.metadata as metadata
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    queue = ["diffusers", "transformers", "accelerate", "huggingface-hub", "peft", "pillow", "kornia", "bitsandbytes"]
    checked, errors = set(), []
    while queue:
        package = queue.pop()
        name = canonicalize_name(package)
        if name in checked:
            continue
        checked.add(name)
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {package}")
            continue
        for specification in distribution.requires or []:
            requirement = Requirement(specification)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(requirement.name)
                if requirement.specifier and version not in requirement.specifier:
                    errors.append(f"{package} requires {requirement}; installed {version}")
            except metadata.PackageNotFoundError:
                errors.append(f"{package} requires missing {requirement.name}")
            queue.append(requirement.name)
    return errors


def validate_generate(request: dict[str, Any]) -> dict[str, Any]:
    """Validate protocol input without importing the GPU libraries."""
    model_dir = Path(request["model_dir"]).resolve()
    index = model_dir / "model_index.json"
    if not index.is_file():
        raise ValueError(f"Qwen-Image 2.1 model_index.json is missing: {model_dir}")
    with index.open(encoding="utf-8") as handle:
        if json.load(handle).get("_class_name") != "QwenImage21Pipeline":
            raise ValueError("The selected directory is not a Qwen-Image 2.1 pipeline.")
    expected_precision = request.get("precision")
    if expected_precision is not None:
        if str(expected_precision).upper() not in {"BF16", "NF4", "INT8"}:
            raise ValueError("precision must be BF16, NF4, or INT8.")
        if quantization_profile(model_dir) != str(expected_precision).upper():
            raise ValueError("Selected precision does not match the saved model configuration.")
    width, height = int(request.get("width", 1024)), int(request.get("height", 1024))
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("Qwen-Image 2.1 width and height must be positive multiples of 32.")
    steps = int(request.get("steps", 40))
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000.")
    cfg = float(request.get("cfg", 1.0))
    if not 1.0 <= cfg <= 30.0:
        raise ValueError("cfg must be between 1 and 30.")
    prompt = request.get("prompt", "")
    negative = request.get("negative_prompt", "")
    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise ValueError("prompt and negative_prompt must be strings.")
    image_paths = request.get("input_images") or []
    if not isinstance(image_paths, list) or len(image_paths) > MAX_REFERENCE_IMAGES:
        raise ValueError("input_images must be a list containing at most 10 image paths.")
    images = [Path(item).resolve() for item in image_paths]
    for path in images:
        if not path.is_file():
            raise ValueError(f"Reference image does not exist: {path}")
    output = Path(request["output_path"]).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png to retain the alpha channel.")
    if output in images:
        raise ValueError("output_path must differ from every reference image.")
    seed = int(request.get("seed", -1))
    seed = secrets.randbits(63) if seed < 0 else seed % (2**63)
    return {
        "model_dir": str(model_dir), "width": width, "height": height,
        "steps": steps, "cfg": cfg, "prompt": prompt,
        "negative_prompt": negative, "input_images": images,
        "output_path": output, "seed": seed,
    }


def quantization_profile(model_dir):
    """Read the saved component configs; do not infer precision from a filename."""
    components = {}
    for name in ("transformer", "text_encoder"):
        path = Path(model_dir) / name / "config.json"
        config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        quant = config.get("quantization_config") or {}
        components[name] = quant
    if not any(components.values()):
        return "BF16"
    profiles = []
    for name, quant in components.items():
        four_bit = quant.get("load_in_4bit", quant.get("_load_in_4bit", False))
        eight_bit = quant.get("load_in_8bit", quant.get("_load_in_8bit", False))
        if quant.get("quant_method") != "bitsandbytes" or four_bit == eight_bit:
            raise ValueError(f"Unsupported or incomplete quantization configuration for {name}.")
        if four_bit:
            if not (quant.get("bnb_4bit_quant_type") == "nf4"
                    and quant.get("bnb_4bit_use_double_quant")
                    and quant.get("bnb_4bit_compute_dtype") == "bfloat16"):
                raise ValueError(f"Unsupported NF4 configuration for {name}.")
            profiles.append("NF4")
        else:
            profiles.append("INT8")
    if len(set(profiles)) != 1:
        raise ValueError("Transformer and text encoder must use the same precision profile.")
    return profiles[0]


def verify_quantized_components(pipe, precision):
    """Check real packed weights after loading, including their quantization state."""
    import bitsandbytes as bnb
    import torch

    details = {}
    for name in ("transformer", "text_encoder"):
        model = getattr(pipe, name)
        layer_class = bnb.nn.Linear4bit if precision == "NF4" else bnb.nn.Linear8bitLt
        layers = [module for module in model.modules() if isinstance(module, layer_class)]
        if precision == "NF4":
            valid = layers and all(
                layer.weight.dtype == torch.uint8
                and layer.weight.bnb_quantized
                and layer.weight.quant_state is not None
                and layer.weight.quant_state.quant_type == "nf4"
                and layer.weight.quant_state.nested
                and layer.compute_dtype == torch.bfloat16
                for layer in layers
            )
        else:
            valid = layers and all(
                layer.weight.dtype == torch.int8
                and not layer.weight.has_fp16_weights
                and (layer.weight.SCB is not None or layer.state.SCB is not None)
                for layer in layers
            )
        if not valid:
            raise RuntimeError(f"{name} did not load valid packed {precision} weights.")
        details[name] = {
            "quantized_linear_layers": len(layers),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "quant_type": precision.lower(),
        }
        if precision == "NF4":
            details[name].update(double_quant=True, compute_dtype="bfloat16")
    return details


class Qwen21Worker:
    def __init__(self, emit):
        self.emit = emit
        self.pipe = None
        self.model_dir = None
        self.precision = None
        self.quantization_details = {}

    def unload(self):
        if self.pipe is not None:
            pipe = self.pipe
            self.pipe = None
            self.model_dir = None
            self.precision = None
            self.quantization_details = {}
            pipe.maybe_free_model_hooks()
            del pipe
        gc.collect()
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _load(self, model_dir: str):
        if self.pipe is not None and self.model_dir == model_dir:
            return
        self.unload()
        self.emit("loading", phase="imports")
        import torch
        from diffusers import QwenImage21Pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen-Image 2.1 requires the configured CUDA GPU; CUDA is unavailable.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The configured GPU does not support BF16 required by this runtime.")
        precision = quantization_profile(model_dir)
        self.emit("loading", phase="weights", model_dir=model_dir)
        pipe = QwenImage21Pipeline.from_pretrained(
            model_dir, dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.pipe = pipe
        if precision in {"NF4", "INT8"}:
            self.quantization_details = verify_quantized_components(pipe, precision)
        if precision == "INT8":
            from qwen21_quantization import patch_int8_cpu_offload
            for name in ("transformer", "text_encoder"):
                patch_int8_cpu_offload(getattr(pipe, name))
        self.precision = precision
        # Move each whole component as needed. Quantized weights remain packed during
        # offload, and unloading the DiT leaves room for the untiled VAE decode.
        pipe.enable_model_cpu_offload(gpu_id=0)
        # Preserve the official untiled decode. This VAE's default 256-pixel
        # tiles with 192-pixel stride produced periodic magenta/green seams in
        # a real 1024-pixel validation image. CPU offload still frees the DiT
        # before VAE decoding, leaving the GPU available for the whole image.
        pipe.set_progress_bar_config(disable=True)
        self.model_dir = model_dir
        self.emit("loading", phase="ready", gpu=torch.cuda.get_device_name(0),
                  precision=precision, quantization=self.quantization_details)

    def generate(self, request):
        values = validate_generate(request)
        self._load(values["model_dir"])
        import torch
        from PIL import Image

        images = []
        for path in values["input_images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGBA"))

        def on_step_end(pipe, step, timestep, callback_kwargs):
            self.emit("progress", step=step + 1, total=values["steps"])
            return callback_kwargs

        arguments = {
            "prompt": values["prompt"], "width": values["width"],
            "height": values["height"], "num_inference_steps": values["steps"],
            "true_cfg_scale": values["cfg"],
            "generator": torch.Generator(device="cuda").manual_seed(values["seed"]),
            "callback_on_step_end": on_step_end, "use_kv_cache": True,
        }
        if values["cfg"] > 1:
            arguments["negative_prompt"] = values["negative_prompt"]
        if images:
            arguments["image"] = images
            # Reference-image area follows the requested output area, avoiding
            # an implicit 1024 reference resize for a 2048 output.
            arguments["output_resolution"] = int((values["width"] * values["height"]) ** 0.5)
        self.emit("progress", step=0, total=values["steps"])
        result = self.pipe(**arguments).images[0]
        output = values["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        try:
            result.save(temporary, format="PNG")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "output_path": str(output), "width": result.width,
            "height": result.height, "mode": result.mode, "seed": values["seed"],
            "steps": values["steps"], "cfg": values["cfg"],
            "model_revision": MODEL_REVISION, "use_kv_cache": True,
            "precision": self.precision, "quantization": self.quantization_details,
        }

    def status(self):
        import importlib.metadata
        import diffusers
        import torch
        from diffusers import QwenImage21Pipeline
        from diffusers.pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        # from_pretrained checks every importable base class, including optional
        # guidance helpers. Exercise that same lazy-import path before promising
        # readiness; merely importing QwenImage21Pipeline misses stale optional
        # libraries inherited from Forge (for example, the old Kornia package).
        loader_classes = {}
        for library, name in (
            ("transformers", "Qwen3VLProcessor"),
            ("transformers", "Qwen3VLForConditionalGeneration"),
            ("diffusers", "FlowMatchEulerDiscreteScheduler"),
            ("diffusers", "QwenImage21Transformer2DModel"),
            ("diffusers", "AutoencoderKLQwenImage21"),
        ):
            loaded_class, _ = get_class_obj_and_candidates(
                library, name, ALL_IMPORTABLE_CLASSES, diffusers.pipelines, False,
            )
            loader_classes[name] = loaded_class.__name__

        return {
            "loaded": self.pipe is not None, "model_dir": self.model_dir,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "diffusers", "accelerate", "huggingface_hub", "bitsandbytes",
            )},
            "precision": self.precision, "quantization": self.quantization_details,
            "pipeline": QwenImage21Pipeline.__name__,
            "text_encoder": Qwen3VLForConditionalGeneration.__name__,
            "processor": Qwen3VLProcessor.__name__,
            "loader_classes": loader_classes,
            "dependency_errors": runtime_dependency_errors(),
        }


def main():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    protocol_output = sys.stdout
    request_id = None

    def emit(event, **fields):
        protocol_output.write(json.dumps(
            {"event": event, "request_id": request_id, **fields}, ensure_ascii=False,
        ) + "\n")
        protocol_output.flush()

    worker = Qwen21Worker(emit)
    with contextlib.redirect_stdout(sys.stderr):
        emit("ready", protocol=1)
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Each protocol line must be a JSON object.")
                request_id = request.get("request_id")
                command = request.get("command", request.get("cmd"))
                if command == "generate":
                    emit("result", **worker.generate(request))
                elif command == "status":
                    emit("status", **worker.status())
                elif command == "unload":
                    worker.unload()
                    emit("unloaded")
                elif command == "shutdown":
                    worker.unload()
                    emit("shutdown")
                    break
                else:
                    raise ValueError(f"Unknown worker command: {command!r}")
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                # A failed inference can leave device hooks and large tensors
                # live. Release them before accepting another command.
                try:
                    worker.unload()
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                emit("error", message=str(error), error_type=type(error).__name__)
        worker.unload()


if __name__ == "__main__":
    main()
