"""Create a reusable local NF4 or INT8 Qwen-Image 2.1 from the original weights.

Run with runtimes/qwen-image-2.1/Scripts/python.exe. Each component is converted
and cold-reloaded in its own process, keeping host/GPU memory bounded. The VAE,
processor and scheduler are copied unchanged. The source is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("transformer", "text_encoder")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_records(folder):
    return [{"path": str(path.relative_to(folder)).replace("\\", "/"),
             "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(folder.rglob("*")) if path.is_file()]


def valid_records(folder, records):
    return bool(records) and all(
        (folder / item["path"]).is_file()
        and (folder / item["path"]).stat().st_size == item["size"]
        and sha256(folder / item["path"]) == item["sha256"] for item in records
    )


def inspect_quantized(model, precision):
    import bitsandbytes as bnb
    import torch
    kind = bnb.nn.Linear4bit if precision == "nf4" else bnb.nn.Linear8bitLt
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, kind)]
    if not layers:
        raise RuntimeError(f"No actual {precision} layers were loaded.")
    for name, module in layers:
        weight = module.weight
        if precision == "nf4":
            state = weight.quant_state
            if (weight.dtype != torch.uint8 or not weight.bnb_quantized or state is None
                    or state.quant_type != "nf4" or not state.nested
                    or module.compute_dtype != torch.bfloat16):
                raise RuntimeError(f"Incorrect NF4 storage, nested scales or compute dtype: {name}")
        elif (weight.dtype != torch.int8 or weight.has_fp16_weights
              or (weight.SCB is None and module.state.SCB is None)):
            raise RuntimeError(f"Incorrect INT8 storage or missing scale: {name}")
    return {"quantized_linear_layers": len(layers), "storage": "uint8" if precision == "nf4" else "int8",
            "model_memory_bytes": model.get_memory_footprint()}


def convert_component(args):
    import bitsandbytes as bnb
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersConfig, QwenImage21Transformer2DModel
    from transformers import BitsAndBytesConfig as TransformersConfig, Qwen3VLForConditionalGeneration

    if bnb.__version__ != "0.50.2" or not torch.cuda.is_available():
        raise RuntimeError("Conversion requires bitsandbytes 0.50.2 and a CUDA GPU in the isolated runtime.")
    source = args.source / args.component
    destination = args.output / args.component
    receipt_path = args.output / f"{args.component}.quantization.json"
    source_config = read_json(source / "config.json")
    if source_config.get("quantization_config"):
        raise ValueError(f"Source must be original unquantized weights: {source}")
    source_records = file_records(source)
    if receipt_path.is_file():
        previous = read_json(receipt_path)
        if (previous.get("precision") == args.precision and previous.get("source_files") == source_records
                and valid_records(destination, previous.get("files", []))):
            print(f"{args.component}: verified existing {args.precision} conversion; skipping.", flush=True)
            return
        raise RuntimeError(f"Existing conversion does not match source or hashes: {receipt_path}")
    if destination.exists():
        raise RuntimeError(f"Incomplete component folder exists: {destination}. Move it aside before retrying.")
    config_class, model_class = ((DiffusersConfig, QwenImage21Transformer2DModel)
                                 if args.component == "transformer" else (TransformersConfig, Qwen3VLForConditionalGeneration))
    settings = (dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                     bnb_4bit_compute_dtype=torch.bfloat16)
                if args.precision == "nf4" else dict(load_in_8bit=True, llm_int8_threshold=6.0))
    dtype_kw = {"torch_dtype" if args.component == "transformer" else "dtype": torch.bfloat16}
    started = time.monotonic()
    print(f"{args.component}: loading and quantizing {args.precision} on CUDA.", flush=True)
    model = model_class.from_pretrained(str(source), quantization_config=config_class(**settings),
                                        device_map={"": 0}, low_cpu_mem_usage=True,
                                        local_files_only=True, **dtype_kw)
    converted = inspect_quantized(model, args.precision)
    model.save_pretrained(str(destination), safe_serialization=True, max_shard_size="4GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{args.component}: verifying prequantized reload and CPU/GPU transfer.", flush=True)
    reloaded = model_class.from_pretrained(str(destination), local_files_only=True,
                                           low_cpu_mem_usage=True, device_map={"": 0}, **dtype_kw)
    restored = inspect_quantized(reloaded, args.precision)
    reloaded.to("cpu").to("cuda")
    inspect_quantized(reloaded, args.precision)
    if converted != restored:
        raise RuntimeError(f"Reload changed quantized model layout: {converted} != {restored}")
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    report = {"component": args.component, "precision": args.precision,
              "source": str(source), "source_files": source_records,
              "verification": restored, "cold_reload_and_cpu_gpu_transfer": True,
              "files": file_records(destination), "elapsed_seconds": round(time.monotonic() - started, 2),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(receipt_path, report)
    print(json.dumps(report["verification"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("nf4", "int8"), required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "models" / "diffusers" / "Qwen-Image-2.1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--component", choices=COMPONENTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = (args.output or args.source.with_name("Qwen-Image-2.1-" + args.precision.upper())).resolve()
    if args.source == args.output or args.source in args.output.parents or args.output in args.source.parents:
        parser.error("The source and output must be separate sibling model directories.")
    index = read_json(args.source / "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline":
        parser.error("Source is not a QwenImage21Pipeline.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.component:
        convert_component(args)
        return
    completed = args.output / "quantization-manifest.json"
    if completed.is_file():
        manifest = read_json(completed)
        if manifest.get("precision") == args.precision and valid_records(args.output, manifest.get("files", [])):
            print(f"Verified existing complete {args.precision} pipeline: {args.output}")
            return
        raise RuntimeError(f"Existing pipeline manifest failed validation: {completed}")
    if (args.output / "model_index.json").exists():
        raise RuntimeError(f"Output already contains an unverified pipeline: {args.output}")
    for component in COMPONENTS:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--precision", args.precision,
                        "--source", str(args.source), "--output", str(args.output),
                        "--component", component], check=True)
    for item in args.source.iterdir():
        if item.name in {*COMPONENTS, "model_index.json", ".cache"}:
            continue
        target = args.output / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)
    # Publish the pipeline index only after both components are verified.
    write_json(args.output / "model_index.json", index)
    manifest = {"precision": args.precision, "source": str(args.source),
                "original_weights_retained": True, "vae": "unchanged original weights; BF16 inference",
                "components": {name: read_json(args.output / f"{name}.quantization.json") for name in COMPONENTS},
                "versions": {name: importlib.metadata.version(name) for name in ("bitsandbytes", "torch", "diffusers", "transformers", "accelerate")},
                "files": file_records(args.output), "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(completed, manifest)
    print(json.dumps({"output": str(args.output), "precision": args.precision,
                      "total_bytes": sum(item["size"] for item in manifest["files"]),
                      "manifest": str(completed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
