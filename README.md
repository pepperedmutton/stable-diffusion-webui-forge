# Stable Diffusion WebUI Forge — personal fork

This fork keeps my Qwen Image 2.1, Krea 2, Anima and Illustrious workflows in one Forge installation. It adds model presets, an isolated Qwen runtime, native Outpaint, and an English touch interface for phone browsers.

Based on [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Model weights are stored separately from this repository.

[中文说明](README.zh-CN.md) · [Qwen setup and limitations](docs/QWEN_IMAGE_2_1.md) · [Outpaint](extensions-builtin/forge_outpaint/README.md)

## Start in Colab

[Open the English Forge notebook in Colab](https://colab.research.google.com/github/pepperedmutton/stable-diffusion-webui-forge/blob/main/colab/Forge_Colab.ipynb). Save a copy to Drive. Its single action menu separates first-time setup from daily launching, and **Run all** executes only the chosen action. The equivalent manual steps follow.

Select an L4 or A100 GPU runtime and mount Google Drive. Forge source, Python, both dependency environments, models and outputs are stored persistently under `MyDrive/ForgeColab`:

```text
MyDrive/ForgeColab/
├── forge/                  # persistent Git checkout and extensions
│   ├── .colab/python/      # complete persistent Python 3.10 distribution
│   ├── venv/              # main Forge dependencies
│   └── runtimes/qwen-image-2.1/  # separate Qwen dependencies
├── models/
│   ├── Stable-diffusion/
│   ├── text_encoder/
│   ├── VAE/
│   ├── Lora/
│   ├── ControlNet/
│   └── diffusers/
├── embeddings/             # textual inversion files
├── models-manifest.json    # optional integrity manifest
├── state/                  # persistent settings, created on first start
└── outputs/                # generated images, created on first start
```

Run this in one **Colab code cell** after Drive is mounted:

```python
from pathlib import Path
import os
import subprocess

storage = Path("/content/drive/MyDrive/ForgeColab")
if not os.path.ismount("/content/drive"):
    raise RuntimeError("Mount Google Drive before installing Forge.")
storage.mkdir(parents=True, exist_ok=True)
(storage / "models").mkdir(exist_ok=True)
repo = storage / "forge"
url = "https://github.com/pepperedmutton/stable-diffusion-webui-forge.git"
if repo.is_symlink():
    raise RuntimeError("Use a real forge directory inside your Drive storage.")
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", "main", "--single-branch", url, str(repo)], check=True)
if not (repo / ".git").is_dir():
    raise RuntimeError("This directory is not a Forge Git clone.")
origin = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
if origin.rstrip("/").removesuffix(".git") != url.removesuffix(".git"):
    raise RuntimeError("This checkout belongs to a different repository.")

%run /content/drive/MyDrive/ForgeColab/forge/colab/start.py --drive-root /content/drive/MyDrive/ForgeColab --install-only
```

This first step installs the persistent environments without requiring weights or opening a public server. The installer restores the bundled extension sources and applies the phone interface. On the first installation, a temporary Python bootstrap may be downloaded, then its complete interpreter and standard library are copied once to Drive. Both dependency environments are installed on Drive with copy mode; they are never copied into `/content` at launch. The notebook interpreter, required operating-system libraries and temporary installation files belong to the Colab VM.

Download the model collection in another Colab cell. Enter your Civitai API key in the hidden input; it is passed in memory, never placed in a download URL or saved in the notebook source:

```python
import getpass
import os

os.environ["CIVITAI_API_KEY"] = getpass.getpass("Civitai API key: ").strip()
try:
    %run /content/drive/MyDrive/ForgeColab/forge/colab/download_models.py --drive-root /content/drive/MyDrive/ForgeColab --download
finally:
    os.environ.pop("CIVITAI_API_KEY", None)
```

Omit `--download` to preview the current file list and storage requirement without downloading. The collection prefers exact Civitai matches and uses pinned official sources for public models absent there, including Qwen Image 2.1. It uses WAI Illustrious SDXL v150 in place of the local Novsw merge and skips the three private LoRAs (`csky_sora_nova_il_r32_a16`, `galkan`, `shadow_beast`). Model weights and embeddings go directly to their Drive folders. Interrupted transfers resume when the same command is rerun; completed files are checksum-verified. Download results are recorded in `ForgeColab/download-state/download-report.json`. Download completion does not prove successful generation.

After the BF16 Qwen bundle and the isolated Qwen environment are ready, create the NF4 and INT8 variants on a suitable GPU. Both commands read BF16 from Drive and write their separate, reusable directories back to Drive:

```python
from pathlib import Path
import subprocess

storage = Path("/content/drive/MyDrive/ForgeColab")
repo = storage / "forge"
for precision in ("nf4", "int8"):
    subprocess.run([
        str(repo / "runtimes/qwen-image-2.1/bin/python"),
        str(repo / "scripts/quantize_qwen21.py"), "--precision", precision,
        "--source", str(storage / "models/diffusers/Qwen-Image-2.1"),
        "--output", str(storage / ("models/diffusers/Qwen-Image-2.1-" + precision.upper())),
    ], check=True)
```

A failed conversion raises an error and stops this cell before the next variant. These conversions preserve the BF16 source and record checksums and reload checks in each output's `quantization-manifest.json`. An already verified conversion is reused. An incomplete or mismatched output is reported for review instead of being overwritten. Legacy split Qwen Image files are retained in `models/legacy-qwen-image`; they are not Qwen Image 2.1 profiles.

Once the model files are in the directories above, start Forge with:

```python
%run /content/drive/MyDrive/ForgeColab/forge/colab/start.py --drive-root /content/drive/MyDrive/ForgeColab
```

The launcher asks for a session password without echoing your input. If you leave it blank, it generates a password and shows it in the notebook output; keep that output private. Open the printed `https://…gradio.live` address in your phone browser and sign in as `forge` with that password. Keep the Colab cell running while using Forge.

The models are read from Drive; this command does **not** upload local models or download missing weights. An existing `models-manifest.json` is checked for missing files and wrong sizes. Add `--verify-sha256` to the `%run` line to verify the checksum of every manifest-listed file, which can take a long time on Drive. Files added later and absent from the manifest are reported as extra files and have no manifest checksum to verify.

The full set of presets requires native BF16 support. Use an L4 or A100 runtime when available; this launcher rejects T4 runtimes because they cannot run the complete model set as configured. Available GPU memory still limits model size and image resolution. The main environment uses PyTorch 2.7.0 and torchvision 0.22.0 with CUDA 12.6 wheels; see the [PyTorch package reference](https://pytorch.org/get-started/previous-versions/).

Colab restricts using web interfaces for content generation on free managed runtimes. Use an eligible paid account with a positive compute-unit balance, subject to the [Colab FAQ](https://research.google.com/colaboratory/faq.html). Runtime availability and duration vary. Stop the cell and disconnect/delete the runtime when finished.

After reconnecting, mount the same Drive and run the start command again. It executes the saved Python and dependencies directly from Drive, checking their locations and imports. It does not clone, copy or reinstall the complete environment on every new runtime. Missing native operating-system libraries may still need installation in the new VM. The checkout is not updated automatically, and local Git changes are preserved. Drive's many small-file reads can make startup slower; its mount-reported free space is not the same as your account storage quota.

Startup reports missing optional extension dependencies. The Linux installer includes `insightface==1.0.1`, `onnx==1.12.0` and `onnxruntime==1.23.2` for Forge's CPU FaceID preprocessing; these pins preserve the main environment's protobuf version and avoid a source compilation. Imports and CPU-provider availability are checked again when reconnecting. FaceID also needs the actual adapter, image encoder and face-analysis model files; a companion LoRA alone is insufficient. Add `--repair-environment` to the `%run` line to retry environment setup after an interrupted or failed installation; this does not supply missing model weights.

## Model presets

The `UI` selector switches the model family, defaults and relevant controls together. These are the filenames used by the presets; put them in the matching folders on Drive.

| Preset | Main model | Additional files |
| --- | --- | --- |
| Qwen Image 2.1 | `diffusers/Qwen-Image-2.1-NF4/`, `Qwen-Image-2.1-INT8/`, or `Qwen-Image-2.1/` | Complete Diffusers directory, including its matching text encoder, tokenizer and VAE |
| Krea 2 — CocoaMixZero v1.0 | `Stable-diffusion/krea2Cocoamixzero_v10.safetensors` | `text_encoder/qwen3vl_4b_fp8_scaled.safetensors`, `VAE/qwen_image_vae.safetensors` |
| MiaoMiao Harem — Illustrious v2.0 | `Stable-diffusion/miaomiaoHarem_v20.safetensors` | Optional `VAE/pppanimixVAE_il.safetensors` |
| MiaoMiao Harem — Anima 1.5 | `Stable-diffusion/miaomiaoHarem_anima15.safetensors` | `text_encoder/anima_text_encoder.safetensors`, `VAE/anima_vae.safetensors` |

Qwen Image 2.1 supports text-to-image and instruction-based image editing. BF16, INT8 and NF4 are selected in the UI and use a separate process so their newer dependencies do not replace Forge's dependencies. The BF16 directory is displayed as `Qwen-Image-2.1-BF16` in the checkpoint selector. Old split Qwen Image checkpoints are not substitutes for these directories.

Qwen mode hides controls it does not implement, including LoRA, ControlNet, traditional denoising strength, negative prompts, high-resolution fix, refiner and Outpaint. Switching to another preset restores that workflow's controls. See the [Qwen notes](docs/QWEN_IMAGE_2_1.md) for download, quantization and local validation details. Model licenses remain separate from the code license.

## Using Forge on a phone

The Colab launcher installs the touch layout automatically. The added controls are in English.

- Parameters start locked. Tap **Edit parameters**, then **Edit value** for a number. **Apply value** commits that number; **Cancel** discards its draft.
- Dropdowns and checkboxes take effect immediately while unlocked. **Finish editing** locks the controls again; it does not undo changes already applied.
- Scrolling across a slider does not change its value while locked. Model selectors and other guarded actions require an explicit action.
- Generation uses a confirmation step and blocks rapid repeat taps. For Outpaint, use **Edit outpaint**, followed by **Apply and lock** or **Discard and lock**.
- Larger targets, responsive panels and safe-area spacing support portrait and landscape use. The touch guards activate when the browser reports a coarse pointer; mouse-based desktop controls retain their normal behavior.

The original Forge canvas does not gain full pinch-to-zoom or two-finger panning. Use Outpaint's numeric controls for precise placement. Browser gesture emulation has been tested; physical Pixel 8 Pro testing is still required before claiming device-level coverage.

## Local Windows use

For an existing installation, run `webui-user.bat`. This fork's launcher uses the local virtual environment and selects an available port from 7861 through 7870.

For a fresh clone, restore the directories in `colab/vendor/extensions/` into `extensions/`. The bundled `huggingface_guess` snapshot also contains local model-detection changes; preserve it when setting up the dependency under `repositories/`. The Colab launcher handles both steps automatically. See [bundled source provenance](colab/vendor/README.md).

If you need to build Qwen weights locally, use the existing Forge Python environment:

```powershell
.\venv\Scripts\python.exe .\scripts\setup_qwen21_runtime.py
.\venv\Scripts\python.exe .\scripts\download_qwen21.py
.\runtimes\qwen-image-2.1\Scripts\python.exe .\scripts\quantize_qwen21.py --precision nf4
```

Skip download and quantization when you already have complete matching model directories. The Colab launcher uses those existing directories and does not repeat quantization.

## What this repository includes

- Local Forge changes for model selection, memory release between presets, metadata and control restoration.
- Qwen Image 2.1 adapter, worker, runtime setup, download and quantization tools.
- Native Outpaint with pixel-preserving composition and PNG output.
- Colab setup, persistent Drive state, and the phone touch guards.
- Source snapshots of Krea 2, Dynamic Prompts, Dynamic Thresholding, IP-Adapter and the locally modified model-detection dependency. Their original licenses and source hashes are retained.

Weights, generated images, private prompts/settings, virtual environments and caches are excluded. Existing upstream Forge features remain available where the selected backend supports them.

## Validation scope

The release check on a clean source export ran 168 Python tests: 165 passed and 3 skipped (two require a local checkpoint, one requires the separate CUDA quantization environment). Outpaint passed 21 tests; the JavaScript Krea checks passed. A separate Chromium touch simulation passed 22 interaction checks. The 1,817 bundled runtime source files matched their SHA-256 manifest. Linux/Python 3.10 dependency metadata resolution also passed for the main environment and its listed extensions.

The repository contains regression tests for preset selection, memory policy, Qwen routing, Outpaint, Colab preparation and mobile interaction safeguards. Setup and patching tests use temporary directories and mocked external installation steps. They do not demonstrate that a Google account obtained a particular GPU, that all packages installed in a live Colab session, or that every model fits that GPU.

The historical Qwen validation notes describe local GPU runs. They are not Colab performance measurements. A live Colab start and image generation remain the final environment check.

## License and attribution

Forge is distributed under [AGPL-3.0](LICENSE.txt). Bundled components retain their own licenses. Upstream project history and credits remain in this repository; the [upstream Forge README](https://github.com/lllyasviel/stable-diffusion-webui-forge#readme) documents the original project.
