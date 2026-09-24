"""Bridge the Qwen Image 2.1 Diffusers runtime into Forge's normal UI.

The separate interpreter keeps the new Transformers/Diffusers dependencies out
of Forge's environment. No inference calls leave this machine.
"""

import atexit
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid

from modules.paths_internal import models_path


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(models_path)
MODEL_NAME = "Qwen-Image-2.1-NF4"
MODEL_DIR = MODELS_ROOT / "diffusers" / MODEL_NAME
MODEL_PROFILES = {
    "nf4": {"name": MODEL_NAME, "directory": "Qwen-Image-2.1-NF4",
            "runtime": "DiT + TE NF4 (double quantization) / VAE BF16 / CPU offload"},
    "int8": {"name": "Qwen-Image-2.1-INT8", "directory": "Qwen-Image-2.1-INT8",
             "runtime": "DiT + TE INT8 / VAE BF16 / CPU offload"},
    "bf16": {"name": "Qwen-Image-2.1-BF16", "directory": "Qwen-Image-2.1",
             "runtime": "DiT + TE + VAE BF16 / CPU offload"},
}
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
RUNTIME_PYTHON = ROOT / "runtimes" / "qwen-image-2.1" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
WORK_DIR = ROOT / "tmp" / "qwen21-jobs"
_generation_lock = threading.RLock()
_worker = None
_tokenizer = None


def is_checkpoint(info):
    return bool(info and getattr(info, "qwen21", False))


def selected_for(p):
    from modules import sd_models, shared
    name = (getattr(p, "override_settings", None) or {}).get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
    return is_checkpoint(sd_models.get_closet_checkpoint_match(name))


def checkpoint_profile(info):
    precision = getattr(info, "qwen21_precision", "nf4")
    profile = MODEL_PROFILES[precision]
    model_dir = Path(getattr(info, "model_dir", MODELS_ROOT / "diffusers" / profile["directory"]))
    return model_dir, getattr(info, "name", profile["name"]), precision


def _component_weights_ready(directory, prefix):
    index_path = directory / f"{prefix}.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        names = set(weight_map.values())
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            return False
        files = [directory / name for name in names]
    else:
        files = [directory / f"{prefix}.safetensors"]
    return all(path.is_file() and path.stat().st_size > 0 for path in files)


def _component_precision_matches(directory, precision):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if precision == "bf16":
        return not quantization
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "bitsandbytes":
        return False
    four_bit = quantization.get("load_in_4bit", quantization.get("_load_in_4bit", False))
    eight_bit = quantization.get("load_in_8bit", quantization.get("_load_in_8bit", False))
    if precision == "nf4":
        return (four_bit is True and not eight_bit
                and quantization.get("bnb_4bit_quant_type") == "nf4"
                and quantization.get("bnb_4bit_use_double_quant") is True)
    return precision == "int8" and eight_bit is True and not four_bit


def model_files_ready(model_dir=None, precision="nf4"):
    """Validate component precision and every shard (or an unsharded file)."""
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    try:
        if json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))["_class_name"] != "QwenImage21Pipeline":
            return False
        for component, prefix in (
            ("transformer", "diffusion_pytorch_model"),
            ("text_encoder", "model"),
        ):
            directory = model_dir / component
            if not _component_precision_matches(directory, precision) or not _component_weights_ready(directory, prefix):
                return False
        return (_component_precision_matches(model_dir / "vae", "bf16")
                and _component_weights_ready(model_dir / "vae", "diffusion_pytorch_model"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def register_checkpoint():
    from modules import sd_models

    for precision, profile in MODEL_PROFILES.items():
        model_dir = MODELS_ROOT / "diffusers" / profile["directory"]
        if not model_files_ready(model_dir, precision):
            continue
        info = sd_models.CheckpointInfo(str(model_dir / "model_index.json"))
        info.qwen21 = True
        info.qwen21_precision = precision
        info.model_dir = str(model_dir)
        info.name = info.name_for_extra = info.model_name = profile["name"]
        info.title = info.short_title = profile["name"]
        info.sha256 = None  # A Hub revision is not a weight-file SHA256.
        info.hash = info.shorthash = None
        info.ids = [profile["name"], f"Qwen Image 2.1 {precision.upper()}", str(model_dir), str(model_dir / "model_index.json")]
        if precision == "nf4":
            info.ids.extend(["Qwen-Image-2.1", "Qwen Image 2.1"])
        info.metadata = {"modelspec.architecture": "qwen-image-2.1", "source_revision": MODEL_REVISION,
                         "precision": precision, "runtime": profile["runtime"]}
        info.register()


def prompt_lengths(prompt):
    global _tokenizer
    try:
        if _tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer_path = MODEL_DIR / "processor" / "tokenizer.json"
            if not tokenizer_path.is_file():
                tokenizer_path = MODELS_ROOT / "diffusers" / MODEL_PROFILES["bf16"]["directory"] / "processor" / "tokenizer.json"
            _tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return len(_tokenizer.encode(str(prompt), add_special_tokens=False).ids), 1024
    except Exception:
        return len(str(prompt).encode("utf-8")), 1024


def load_forge_model(info):
    from backend import memory_management
    from modules import sd_models, shared
    import gc

    model_dir, model_name, precision = checkpoint_profile(info)
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError("Qwen Image 2.1 runtime is missing. Run scripts/setup_qwen21_runtime.py with Forge's Python.")
    if not model_files_ready(model_dir, precision):
        raise RuntimeError(f"{model_name} weights are incomplete or their precision configuration does not match.")
    stop_worker()
    memory_management.unload_all_models()
    sd_models.model_data.sd_model = None
    sd_models.clear_forge_model_caches_for_switch()
    gc.collect()
    memory_management.soft_empty_cache()

    class Qwen21Model(sd_models.FakeInitialModel):
        forge_preset = "qwen"
        qwen21 = True
        use_distilled_cfg_scale = False
        is_sd1 = is_sd2 = is_sdxl = is_sd3 = is_inpaint = False
        is_ssd = is_sdxl_inpaint = False
        latent_channels = 64

        def get_prompt_lengths_on_ui(self, prompt):
            return prompt_lengths(prompt)

    model = Qwen21Model()
    model.qwen21_precision = precision
    model.qwen21_model_dir = str(model_dir)
    model.sd_checkpoint_info = info
    model.filename = info.filename
    model.sd_model_hash = None
    model.extra_generation_params = {}
    model.comments = []
    sd_models.model_data.set_sd_model(model)
    sd_models.model_data.forge_hash = str(sd_models.model_data.forge_loading_parameters)
    shared.opts.data["sd_checkpoint_hash"] = None
    return model, True


class Worker:
    def __init__(self):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.log = open(WORK_DIR / "worker.stderr.log", "a", encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            self.process = subprocess.Popen(
                [str(RUNTIME_PYTHON), "-u", str(ROOT / "modules_forge" / "qwen21_worker.py")],
                cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding="utf-8", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.log.close()
            raise
        self.events = queue.Queue()
        self.reader_thread = threading.Thread(target=self._read, daemon=True)
        self.reader_thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.events.put(json.loads(line))
                except ValueError:
                    self.log.write(line)
                    self.log.flush()
        except (OSError, ValueError):
            pass  # The parent can close streams while terminating a worker.
        finally:
            self.events.put({"event": "exit"})

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        reader = getattr(self, "reader_thread", None)
        if reader is not None:
            reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.log.close()

    def generate(self, payload, state):
        request_id = uuid.uuid4().hex
        try:
            self.process.stdin.write(json.dumps({"command": "generate", "request_id": request_id, **payload}) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            stop_worker()
            raise RuntimeError(f"Qwen runtime connection closed. Details: {WORK_DIR / 'worker.stderr.log'}") from None
        last_event = time.monotonic()
        while True:
            if state.interrupted or state.skipped or state.stopping_generation:
                stop_worker()
                return None
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                if self.process.poll() is not None:
                    stop_worker()
                    raise RuntimeError(f"Qwen runtime exited. Details: {WORK_DIR / 'worker.stderr.log'}")
                if time.monotonic() - last_event > 1800:
                    stop_worker()
                    raise RuntimeError("Qwen runtime did not respond for 30 minutes; it was stopped.")
                continue
            if event.get("event") == "exit":
                stop_worker()
                raise RuntimeError(f"Qwen runtime exited. Details: {WORK_DIR / 'worker.stderr.log'}")
            if event.get("request_id") != request_id:
                continue
            last_event = time.monotonic()
            kind = event.get("event")
            if kind == "progress":
                state.sampling_step = int(event["step"])
                state.sampling_steps = int(event["total"])
                state.textinfo = f"Qwen Image 2.1: {state.sampling_step}/{state.sampling_steps}"
            elif kind == "loading":
                state.textinfo = event.get("message", "Loading Qwen Image 2.1")
            elif kind == "error":
                stop_worker()
                raise RuntimeError(event.get("error", event.get("message", "Qwen generation failed")))
            elif kind == "result":
                return event


def stop_worker():
    global _worker
    if _worker is not None:
        worker, _worker = _worker, None
        worker.stop()


def release_if_inactive(checkpoint_info):
    if not is_checkpoint(checkpoint_info) and _generation_lock.acquire(blocking=False):
        try:
            stop_worker()
        finally:
            _generation_lock.release()


def process_images(p):
    """Produce ordinary Forge results, including PNG metadata, seeds and gallery."""
    global _worker
    from PIL import Image, ImageOps
    from modules import images, processing, shared, sd_models

    with _generation_lock:
        info = sd_models.model_data.forge_loading_parameters.get("checkpoint_info")
        model_dir, model_name, precision = checkpoint_profile(info)
        if (not getattr(shared.sd_model, "qwen21", False)
                or getattr(shared.sd_model, "qwen21_model_dir", str(model_dir)) != str(model_dir)):
            load_forge_model(info)
        processing.need_global_unload = False
        p.clear_prompt_cache()
        p.enable_hr = False
        p.refiner_checkpoint = None
        p.restore_faces = p.tiling = False
        p.token_merging_ratio = p.token_merging_ratio_hr = 0
        p.get_token_merging_ratio = lambda for_hr=False: 0
        p.subseed_strength = 0
        p.seed_resize_from_w = p.seed_resize_from_h = 0
        p.clip_skip = 1
        p.sampler_name, p.scheduler, p.cfg_scale = "Euler", "Simple", 1.0
        p.negative_prompt = ""
        p.width, p.height = max(256, int(p.width) // 32 * 32), max(256, int(p.height) // 32 * 32)
        p.sd_model_name, p.sd_model_hash = model_name, None
        p.qwen21 = True
        p.sd_vae_name, p.sd_vae_hash = "Qwen Image 2.1 RGBA VAE", None
        p.fill_fields_from_opts()
        p.setup_prompts()
        p.all_negative_prompts = [""] * len(p.all_prompts)
        sanitized = [re.sub(r"<(?:lora|lyco|hypernet):[^>]*>", "", prompt, flags=re.I).strip() for prompt in p.all_prompts]
        if sanitized != p.all_prompts:
            p.extra_generation_params["Qwen prompt adjustment"] = "Removed incompatible legacy network tags"
            p.all_prompts = sanitized
        seed = processing.get_fixed_seed(p.seed)
        subseed = processing.get_fixed_seed(p.subseed)
        p.all_seeds = seed if isinstance(seed, list) else [int(seed) + i for i in range(len(p.all_prompts))]
        p.all_subseeds = subseed if isinstance(subseed, list) else [int(subseed)] * len(p.all_prompts)
        p.extra_generation_params.update({"Qwen revision": MODEL_REVISION,
                                         "Qwen precision": precision.upper(),
                                         "Qwen runtime": MODEL_PROFILES[precision]["runtime"],
                                         "Qwen scheduler": "FlowMatchEulerDiscreteScheduler"})
        shared.state.job_count = len(p.all_prompts)
        shared.state.sampling_steps = p.steps
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        job_dir = WORK_DIR / uuid.uuid4().hex
        job_dir.mkdir()
        input_paths = []
        for i, source in enumerate(getattr(p, "init_images", None) or []):
            path = job_dir / f"input-{i}.png"
            source.save(path)
            input_paths.append(str(path))
        mask = getattr(p, "image_mask", None)
        if mask is not None:
            mask = processing.create_binary_mask(mask)
            if getattr(p, "inpainting_mask_invert", 0):
                mask = ImageOps.invert(mask)
            mask_path = job_dir / "mask.png"
            mask.save(mask_path)
        output_images, infotexts, completed_indexes = [], [], []
        try:
            if _worker is None or _worker.process.poll() is not None:
                stop_worker()
                _worker = Worker()
            for i, prompt in enumerate(p.all_prompts):
                if shared.state.interrupted or shared.state.stopping_generation:
                    break
                shared.state.skipped = False
                if _worker is None or _worker.process.poll() is not None:
                    stop_worker()
                    _worker = Worker()
                shared.state.job = f"{model_name} {i + 1}/{len(p.all_prompts)}"
                shared.state.sampling_step = 0
                refs = [input_paths[i % len(input_paths)]] if input_paths else []
                if mask is not None and refs:
                    refs.append(str(mask_path))
                    prompt += " Edit only the white region in the second reference image (the mask), keeping the rest of the first image unchanged. "
                path = job_dir / f"output-{i}.png"
                result = _worker.generate({
                    "model_dir": str(model_dir), "precision": precision, "output_path": str(path), "prompt": prompt,
                    "negative_prompt": "", "width": p.width, "height": p.height,
                    "steps": p.steps, "cfg": 1.0, "seed": p.all_seeds[i], "input_images": refs,
                }, shared.state)
                if result is None:
                    if shared.state.skipped and not shared.state.interrupted and not shared.state.stopping_generation:
                        shared.state.nextjob()
                        continue
                    break
                with Image.open(path) as source:
                    output = source.copy()
                if mask is not None and refs:
                    with Image.open(refs[0]) as source:
                        original = source.convert(output.mode).resize(output.size, Image.Resampling.LANCZOS)
                    output = Image.composite(output, original, mask.resize(output.size, Image.Resampling.NEAREST))
                    p.extra_generation_params["Qwen edit mode"] = "Mask edit; preserve unmasked pixels"
                p.width, p.height = output.size
                infotext = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=i)
                output.info["parameters"] = infotext
                if p.save_samples():
                    images.save_image(output, p.outpath_samples or shared.opts.outdir_txt2img_samples, "", p.all_seeds[i], p.all_prompts[i], "png", infotext, p=p, skip_stealth_pnginfo=True)
                output_images.append(output)
                infotexts.append(infotext)
                completed_indexes.append(i)
                shared.state.assign_current_image(output)
                shared.state.nextjob()
            return processing.Processed(p, output_images,
                                        seed=p.all_seeds[completed_indexes[0]] if completed_indexes else p.all_seeds[0],
                                        subseed=p.all_subseeds[0],
                                        info=infotexts[0] if infotexts else "Generation interrupted",
                                        infotexts=infotexts,
                                        all_prompts=[p.all_prompts[i] for i in completed_indexes],
                                        all_negative_prompts=[""] * len(completed_indexes),
                                        all_seeds=[p.all_seeds[i] for i in completed_indexes],
                                        all_subseeds=[p.all_subseeds[i] for i in completed_indexes])
        finally:
            # All paths here are in this request's freshly created job directory.
            for path in job_dir.iterdir():
                path.unlink()
            job_dir.rmdir()


atexit.register(stop_worker)
