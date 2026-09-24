"""Install and run this Forge fork from persistent Google Drive storage.

In a Colab Python cell, with Drive already mounted:
    %run /content/drive/MyDrive/ForgeColab/forge/colab/start.py --drive-root /content/drive/MyDrive/ForgeColab

No Android app, Google login automation, keepalive, or model download is used.
Stop the running cell to stop Forge; disconnect/delete the runtime when finished.
"""
from __future__ import annotations

import argparse
import codecs
from contextlib import contextmanager
import getpass
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import warnings

ROOT = Path(__file__).resolve().parents[1]
KIT = Path(__file__).resolve().parent / "kit"
REMOTE = "https://github.com/pepperedmutton/stable-diffusion-webui-forge"
UV_VERSION = "0.8.22"
TORCH = "2.7.0"
TORCHVISION = "0.22.0"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
GUESS_REVISION = "84826248b49bb7ca754c73293299c4d4e23a548d"
_SHARE_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.gradio\.live\Z")
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
_URL = re.compile(r"https://[^\s<>\"'`]+", re.IGNORECASE)


def helper(name):
    spec = importlib.util.spec_from_file_location("forge_colab_" + name, KIT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args, **kwargs):
    return subprocess.run([str(value) for value in args], check=True, **kwargs)


def expected_remote(value: str) -> bool:
    return value.rstrip("/").removesuffix(".git") in (
        REMOTE, "git@github.com:pepperedmutton/stable-diffusion-webui-forge",
    )


def validate_environment(root: Path, drive: Path):
    if sys.platform != "linux":
        raise ValueError("This launcher only runs inside Linux Google Colab; your local Forge is unchanged")
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise ValueError("Open the cell in Google Colab, with Drive mounted and a GPU selected") from exc
    mount = Path("/content/drive")
    if not os.path.ismount(mount) or not (mount / "MyDrive").is_dir():
        raise ValueError("Mount Google Drive at /content/drive in Colab before running this cell")
    if not drive.is_relative_to(mount) or drive == mount or not drive.is_dir():
        raise ValueError("--drive-root must be an existing directory on mounted Google Drive")
    if root != drive / "forge" or root.is_symlink() or drive.is_symlink():
        raise ValueError("Clone this fork into the forge/ directory inside --drive-root; source and environments must stay on Drive")
    if not (root / ".git").is_dir() or not (root / "launch.py").is_file():
        raise ValueError("The launcher must run from a complete Git clone of this fork")
    remote = subprocess.check_output(["git", "-C", str(root), "remote", "get-url", "origin"], text=True).strip()
    if not expected_remote(remote):
        raise ValueError("The checkout origin is not pepperedmutton/stable-diffusion-webui-forge; no files were changed")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], text=True)
    print("Attached GPU:", gpu.strip(), flush=True)
    first = gpu.splitlines()[0].split(",")
    if len(first) >= 2 and float(first[1].strip()) < 8.0:
        raise ValueError("This complete setup includes Qwen 2.1 and requires native BF16 (compute capability 8.0+). T4 is unsupported; select a suitable GPU such as L4 or A100")
    free = shutil.disk_usage(root).free / 2**30
    print(f"Drive mount-reported free space: {free:.1f} GiB (not your account storage quota). Forge, Python, dependencies and models stay on Drive.", flush=True)
    if free < 15:
        raise ValueError("At least 15 GiB of free Drive space is required for the isolated environments, in addition to model storage")


@contextmanager
def launch_lock(root: Path):
    import fcntl
    prepare = helper("colab_prepare")
    path = prepare.safe_target(root, ".colab/launcher.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This checkout is already running. Stop its previous launch cell before starting again") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def claim_runtime(root: Path, drive: Path):
    prepare = helper("colab_prepare")
    marker = prepare.safe_target(root, ".colab/runtime.json")
    if root != drive / "forge":
        raise ValueError("The persistent checkout must be Drive/forge")
    expected = {"format": 2, "storage": "drive", "root": str(root), "drive_root": str(drive)}
    if marker.exists():
        if prepare.read_json(marker) != expected:
            raise ValueError("This checkout belongs to different Colab storage. Use a fresh checkout to keep both environments intact")
    else:
        for relative in ("venv", "runtimes/qwen-image-2.1", ".colab/uv", ".colab/python"):
            path = prepare.safe_target(root, relative)
            if path.exists():
                raise ValueError(f"An unmanaged environment already exists: {path}. Use a fresh Drive/forge checkout")
        prepare.atomic_json(marker, expected)
    for relative in ("venv", "runtimes/qwen-image-2.1", ".colab/uv", ".colab/python"):
        prepare.safe_target(root, relative)
    if (root / "venv/Scripts/python.exe").exists():
        raise ValueError("A Windows environment must not be reused or modified in Colab")


def source_hashes(directory: Path):
    values = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Vendored source must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def verify_vendor_manifest(root: Path):
    """Check the shipped snapshot's exact file inventory, sizes, and digests."""
    vendor = root / "colab/vendor"
    document = json.loads((vendor / "manifest.json").read_text(encoding="utf-8-sig"))
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The vendored source manifest is empty or malformed")
    relative_path = helper("stage_models").relative_path
    indexed = set()
    for entry in entries:
        relative = relative_path(entry.get("path"))
        if relative in indexed:
            raise ValueError(f"Duplicate vendored source entry: {relative}")
        indexed.add(relative)
        path = vendor / relative
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(vendor.resolve()) or
                path.stat().st_size != entry.get("bytes") or
                hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256")):
            raise ValueError(f"Vendored source integrity check failed: {relative}")
    actual = set()
    for folder in (vendor / "extensions", vendor / "repositories"):
        for relative in source_hashes(folder):
            actual.add(folder.name + "/" + relative)
    if indexed != actual:
        raise ValueError("The vendored source manifest does not match its file inventory")
    print(f"Verified {len(indexed)} bundled source files.", flush=True)
    return len(indexed)


def install_extensions(root: Path):
    """Copy audited snapshots once; never overwrite an existing user's edits."""
    prepare = helper("colab_prepare")
    vendor = root / "colab/vendor/extensions"
    if not vendor.is_dir() or vendor.is_symlink():
        raise ValueError("colab/vendor/extensions is missing; clone the complete main branch of this fork")
    installed = []
    for source in sorted(vendor.iterdir()):
        if source.is_symlink():
            raise ValueError("Vendored extension directories must not be symlinks")
        if not source.is_dir():
            continue
        destination = prepare.safe_target(root, "extensions/" + source.name)
        marker = prepare.safe_target(root, ".colab/extensions/" + source.name + ".json")
        snapshot = source_hashes(source)
        if not snapshot:
            raise ValueError(f"Empty vendored extension: {source.name}")
        if destination.exists():
            if not marker.exists() and source_hashes(destination) == snapshot:
                # A previous cell may have stopped between the atomic directory
                # rename and marker write. Adopt only an exact, unedited copy.
                prepare.atomic_json(marker, snapshot)
            if not marker.exists() or prepare.read_json(marker) != snapshot:
                raise ValueError(f"Extension {source.name} is unmanaged or its packaged version changed. Use a fresh Colab checkout; existing files were not overwritten")
            for relative, digest in snapshot.items():
                path = prepare.safe_target(destination, relative)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Extension {source.name} has local changes. Use a fresh Colab checkout or keep those changes explicitly")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".colab-extension-", dir=destination.parent))
            try:
                shutil.copytree(source, temporary, dirs_exist_ok=True)
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            prepare.atomic_json(marker, snapshot)
        installed.append(source.name)
    if "sd-forge-krea2" not in installed:
        raise ValueError("The complete source snapshot must include the sd-forge-krea2 extension")
    print("Bundled extensions:", ", ".join(installed), flush=True)
    return installed


def installation_signature(root: Path, extension_names):
    paths = [root / "requirements_versions.txt", root / "scripts/setup_qwen21_runtime.py",
             root / "modules/launch_utils.py", Path(__file__)]
    digest = hashlib.sha256(f"python3.10 torch{TORCH} vision{TORCHVISION} uv{UV_VERSION}".encode())
    for path in paths:
        digest.update(path.read_bytes())
    for name in extension_names:
        digest.update(json.dumps(source_hashes(root / "colab/vendor/extensions" / name), sort_keys=True).encode())
    digest.update(json.dumps(source_hashes(root / "colab/vendor/repositories/huggingface_guess"), sort_keys=True).encode())
    return digest.hexdigest()


def install_dependency_overlay(root: Path):
    """Keep this fork's model detection changes over the pinned upstream clone.

    The original HEAD stays pinned. launch_utils.git_clone returns immediately
    at that HEAD, and the actual server starts with --skip-prepare-environment.
    """
    prepare = helper("colab_prepare")
    source = root / "colab/vendor/repositories/huggingface_guess"
    if not source.is_dir() or source.is_symlink():
        raise ValueError("The vendored huggingface_guess source is missing")
    destination = prepare.safe_target(root, "repositories/huggingface_guess")
    snapshot = source_hashes(source)
    if not snapshot or not (destination / ".git").is_dir():
        raise ValueError("The pinned huggingface_guess dependency was not cloned successfully")
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != GUESS_REVISION:
        raise ValueError("huggingface_guess is on an unexpected revision; use a fresh Colab checkout")
    marker = prepare.safe_target(root, ".colab/huggingface_guess.json")
    if marker.exists():
        if prepare.read_json(marker) != snapshot:
            raise ValueError("The packaged model-detection dependency changed; use a fresh Colab checkout")
        for relative, digest in snapshot.items():
            path = prepare.safe_target(destination, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Model-detection dependency has local changes: {relative}")
        return
    # Preflight all files: accept pristine upstream or our exact overlay. This
    # also lets an interrupted copy resume without losing unrelated changes.
    targets = []
    for relative, digest in snapshot.items():
        target = prepare.safe_target(destination, relative)
        if target.exists():
            if not target.is_file():
                raise ValueError(f"A directory blocks the dependency file: {relative}")
            actual = target.read_bytes()
            if hashlib.sha256(actual).hexdigest() != digest:
                original = subprocess.run(["git", "-C", str(destination), "show", "HEAD:" + relative], capture_output=True)
                if original.returncode != 0 or actual != original.stdout:
                    raise ValueError(f"An existing dependency file has user changes: {relative}")
        targets.append((source / relative, target))
    for original, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".colab-source-", dir=target.parent)
        os.close(fd)
        part = Path(temporary)
        try:
            shutil.copyfile(original, part)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    prepare.atomic_json(marker, snapshot)
    print("Preserved this fork's model-detection source over its pinned upstream dependency.", flush=True)


GPU_PROBE = """import json, torch
assert torch.cuda.is_available(), 'CUDA is unavailable; select a Colab GPU runtime'
assert torch.__version__.split('+')[0] == '2.7.0', 'Unexpected PyTorch version'
assert torch.cuda.is_bf16_supported(including_emulation=False), 'Native BF16 is required for Qwen; T4 is unsupported'
print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'vram_GiB': torch.cuda.get_device_properties(0).total_memory/2**30, 'torch': torch.__version__, 'native_bf16': True}))
"""
MAIN_PROBE = "import torch,gradio,transformers,diffusers,cv2; print('Main imports passed:',torch.__version__,gradio.__version__,transformers.__version__,diffusers.__version__)"
QWEN_PROBE = "import torch; from diffusers import QwenImage21Pipeline; import bitsandbytes; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(including_emulation=False); print('Qwen imports passed; bitsandbytes:',bitsandbytes.__version__)"
OPTIONAL_PROBE = """import importlib, json
modules = {'dynamicprompts': 'Dynamic Prompts', 'insightface': 'IP-Adapter FaceID',
           'onnxruntime': 'ONNX preprocessors', 'mediapipe': 'MediaPipe preprocessors',
           'handrefinerportable': 'HandRefiner', 'depth_anything': 'Depth Anything',
           'depth_anything_v2': 'Depth Anything V2'}
missing = {}
for module, feature in modules.items():
    try:
        imported = importlib.import_module(module)
        if module == 'mediapipe':
            assert imported.solutions.face_mesh, 'The legacy MediaPipe solutions API is required'
    except Exception as error:
        missing[feature] = type(error).__name__
print('COLAB_OPTIONAL_STATUS=' + json.dumps(missing))
"""


def install_uv(root: Path):
    # Colab's notebook interpreter may not ship ensurepip/pythonX.Y-venv.
    # --target keeps uv out of the notebook's and Forge's package environments.
    target = helper("colab_prepare").safe_target(root, ".colab/uv")
    for candidate in (target / "bin/uv", target / "uv/uv"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--upgrade", "--target", target, "uv==" + UV_VERSION])
    for candidate in (target / "bin/uv", target / "uv/uv"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise ValueError("The private uv installation is missing its Linux executable")


def persistent_python(root: Path, uv: Path, env):
    """Persist a complete managed CPython, including its standard library.

    A venv that points to /root/.local/share/uv would break after a runtime reset.
    Reuse that first-install bootstrap when available, then copy the interpreter
    distribution once to Drive. Dependency environments are never copied out.
    """
    prepare = helper("colab_prepare")
    destination = prepare.safe_target(root, ".colab/python")
    marker = destination / ".forge-python.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError("The persistent Python directory is unmanaged or incomplete; existing files were preserved")
        document = prepare.read_json(marker)
        if document.get("format") != 1 or document.get("version") != "3.10":
            raise ValueError("The persistent Python marker has an unsupported format")
        relative = helper("stage_models").relative_path(document.get("executable"))
        python = prepare.safe_target(destination, relative)
        if not python.is_file():
            raise ValueError("Persistent Python is incomplete; its interpreter is missing")
        return python
    try:
        source = subprocess.check_output([str(uv), "python", "find", "--managed-python", "3.10"], text=True, env=env).strip()
    except subprocess.CalledProcessError:
        run([uv, "python", "install", "3.10"], env=env)
        source = subprocess.check_output([str(uv), "python", "find", "--managed-python", "3.10"], text=True, env=env).strip()
    source = Path(source).resolve()
    source_root = Path(subprocess.check_output([str(source), "-c", "import sys;print(sys.base_prefix)"], text=True, env=env).strip()).resolve()
    if not source.is_relative_to(source_root / "bin") or not (source_root / "lib/python3.10").is_dir():
        raise ValueError("The Python bootstrap is not a complete managed CPython 3.10 distribution")
    relative = source.relative_to(source_root).as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".python-install-", dir=destination.parent))
    try:
        # copytree follows source links into regular copies; Drive hard links
        # are unsupported. No packages from Forge's venv are copied here.
        shutil.copytree(source_root, temporary, dirs_exist_ok=True, symlinks=False)
        prepare.atomic_json(temporary / ".forge-python.json", {"format": 1, "version": "3.10", "executable": relative})
        run([temporary / relative, "-c", "import sys;assert sys.version_info[:2] == (3,10);print('Persistent CPython 3.10 is executable')"], env=env)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / relative


def create_persistent_venv(base_python: Path, directory: Path, env):
    python = directory / "bin/python"
    if not python.is_file():
        # CPython creates lib64 -> lib even with --copies unless it exists.
        # An ordinary directory also works on Drive mounts without link support.
        (directory / "lib64").mkdir(parents=True, exist_ok=True)
        run([base_python, "-m", "venv", "--copies", "--without-pip", directory], env=env)
    return python


def report_optional_components(python: Path, env):
    checked = run([python, "-c", OPTIONAL_PROBE], env=env, text=True, encoding="utf-8", capture_output=True)
    lines = [line.removeprefix("COLAB_OPTIONAL_STATUS=") for line in checked.stdout.splitlines()
             if line.startswith("COLAB_OPTIONAL_STATUS=")]
    if len(lines) != 1:
        raise ValueError("The optional component check did not return a valid result")
    missing = json.loads(lines[0])
    if missing:
        print("OPTIONAL FEATURES UNAVAILABLE:", ", ".join(missing), flush=True)
        print("Main generation can still start. Missing feature libraries are listed in .colab/optional-components.json. "
              "Use --repair-environment to retry failed installers. Linux IP-Adapter FaceID additionally requires a compatible insightface installation.", flush=True)
    else:
        print("Optional component imports passed; model weights and actual preprocessing are not verified by this check.", flush=True)
    return missing


def install_environments(root: Path, drive: Path, extension_names, repair=False):
    prepare = helper("colab_prepare")
    python = root / "venv/bin/python"
    qpython = root / "runtimes/qwen-image-2.1/bin/python"
    constraints = prepare.safe_target(root, ".colab/constraints.txt")
    constraints.write_text((root / "requirements_versions.txt").read_text() +
                           f"\ntorch=={TORCH}\ntorchvision=={TORCHVISION}\n" +
                           "opencv-python==4.11.0.86\nopencv-python-headless==4.11.0.86\nopencv-contrib-python==4.11.0.86\nmediapipe==0.10.9\n", encoding="utf-8")
    env = os.environ.copy()
    # Do not inherit notebook launch arguments or a different Python environment.
    for name in ("COMMANDLINE_ARGS", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "TORCH_COMMAND", "REQS_FILE",
                 "HUGGINGFACE_GUESS_REPO", "HUGGINGFACE_GUESS_HASH"):
        env.pop(name, None)
    env.update({"PIP_CONSTRAINT": str(constraints), "GRADIO_ANALYTICS_ENABLED": "False",
                "GRADIO_TEMP_DIR": str(root / "tmp/gradio"), "PYTHONUNBUFFERED": "1",
                "UV_LINK_MODE": "copy", "PIP_NO_CACHE_DIR": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    qenv = dict(env)
    qenv.pop("PIP_CONSTRAINT", None)
    signature = installation_signature(root, extension_names)
    marker = prepare.safe_target(root, ".colab/installed.json")
    current = prepare.read_json(marker) if marker.exists() else {}
    if (root / ".colab/huggingface_guess.json").exists():
        install_dependency_overlay(root)
    if repair or current.get("signature") != signature or not python.is_file() or not qpython.is_file():
        print("Installing persistent Python 3.10 and CUDA libraries on Drive. Initial setup can take several minutes.", flush=True)
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "git", "build-essential", "libgl1", "libglib2.0-0", "libcairo2-dev", "pkg-config"])
        uv = install_uv(root)
        base_python = persistent_python(root, uv, env)
        create_persistent_venv(base_python, root / "venv", env)
        create_persistent_venv(base_python, root / "runtimes/qwen-image-2.1", qenv)
        version = subprocess.check_output([str(python), "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"], text=True).strip()
        if version != "3.10":
            raise ValueError("The managed environment is not Python 3.10; use a fresh Colab checkout")
        run([python, "-m", "ensurepip", "--upgrade"], env=env)
        # uv seeds newer setuptools; satisfy Forge's pin from PyPI before Triton
        # resolves its setuptools dependency against the CUDA-only wheel index.
        run([python, "-m", "pip", "install", "setuptools==69.5.1", "--index-url", "https://pypi.org/simple"], env=env)
        run([python, "-m", "pip", "install", f"torch=={TORCH}", f"torchvision=={TORCHVISION}", "--index-url", CUDA_INDEX], env=env)
        run([python, "-m", "pip", "install", "-r", root / "requirements_versions.txt", "sentencepiece==0.2.1", "opencv-python==4.11.0.86"], env=env)
        run([python, "-c", GPU_PROBE], env=env)
        run([python, "launch.py", "--exit", "--no-download-sd-model", "--models-dir", drive / "models",
             "--embeddings-dir", drive / "embeddings", "--ui-settings-file", drive / "state/config.json"], cwd=root, env=env)
        install_dependency_overlay(root)
        run([python, "scripts/setup_qwen21_runtime.py", "--uv", uv], cwd=root, env=qenv)
        run([python, "-c", MAIN_PROBE], env=env)
        run([qpython, "-c", QWEN_PROBE], cwd=root, env=qenv)
    else:
        print("Reusing Python and both dependency environments directly from Drive; no environment copy or package reinstall.", flush=True)
        # Native OS libraries belong to the fresh Colab VM, not to a Python venv.
        # Most Colab images include them; install only missing runtime libraries.
        missing_native = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True).stdout
        if any(name not in missing_native for name in ("libGL.so.1", "libglib-2.0.so.0", "libcairo.so.2")):
            run(["apt-get", "update", "-qq"])
            run(["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0", "libcairo2"])
        run([python, "-c", GPU_PROBE], env=env)
        run([python, "-c", MAIN_PROBE], env=env)
        run([qpython, "-c", QWEN_PROBE], cwd=root, env=qenv)
        install_dependency_overlay(root)
    for interpreter in (python, qpython):
        run([interpreter, "-c", "import sys,sysconfig;from pathlib import Path;"
             "expected=Path(sys.argv[1]).resolve();base=Path(sys.argv[2]).resolve();"
             "assert Path(sys.prefix).resolve()==expected,'Environment is not on Drive';"
             "assert Path(sys.base_prefix).resolve()==base,'Python base would disappear after a runtime reset';"
             "assert Path(sysconfig.get_path('purelib')).resolve().is_relative_to(expected),'Packages are outside Drive'",
             interpreter.parent.parent, root / ".colab/python"], env=env)
    print("Environment imports passed. This is not a guarantee that every optional extension or model can run on the allocated GPU.", flush=True)
    missing = report_optional_components(python, env)
    prepare.atomic_json(prepare.safe_target(root, ".colab/optional-components.json"), missing)
    prepare.atomic_json(marker, {"signature": signature})
    return python, env


def validate_share_url(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or not value.isascii():
        raise ValueError("Invalid share URL")
    if any(ord(c) <= 32 or ord(c) == 127 for c in value) or any(c in value for c in "\\%?#"):
        raise ValueError("Invalid share URL characters")
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (parsed.scheme.lower() != "https" or not _SHARE_HOST.fullmatch(host.lower()) or
            parsed.username is not None or parsed.password is not None or parsed.port is not None or
            parsed.path not in ("", "/") or parsed.netloc.lower() != host.lower()):
        raise ValueError("Expected an HTTPS gradio.live origin")
    return "https://" + host.lower() + "/"


class ShareOutputParser:
    """Parse complete log lines and redact credentials across chunk boundaries."""
    MAX_LINE = 65536

    def __init__(self, publish, log, private=()):
        self.publish, self.log = publish, log
        self.private = tuple(sorted((value for value in private if value), key=len, reverse=True))
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending, self.seen, self.last_url = "", set(), None

    def line(self, raw):
        value = _ANSI.sub("", raw)
        for match in _URL.finditer(value):
            candidate = match.group(0).rstrip(")]},;")
            try:
                url = validate_share_url(candidate)
            except ValueError:
                continue
            if url not in self.seen:
                self.seen.add(url)
                self.last_url = url
                self.publish(url)
        for secret in self.private:
            value = value.replace(secret, "[private]")
        self.log(value + "\n")

    def feed(self, block, final=False):
        self.pending += self.decoder.decode(block, final=final)
        while "\n" in self.pending or "\r" in self.pending:
            index = min(pos for pos in (self.pending.find("\n"), self.pending.find("\r")) if pos >= 0)
            raw, self.pending = self.pending[:index], self.pending[index + 1:]
            if len(raw) > self.MAX_LINE:
                raise ValueError("Forge emitted an excessive output line; stopped before publishing an incomplete URL")
            self.line(raw)
        if len(self.pending) > self.MAX_LINE:
            self.pending = ""
            raise ValueError("Forge emitted an excessive output line; stopped before publishing an incomplete URL")
        if final and self.pending:
            self.line(self.pending)
            self.pending = ""


def connection_html(url: str):
    safe = html.escape(validate_share_url(url), quote=True)
    return ("<div style='font:16px/1.6 system-ui;padding:18px;border:1px solid #92a8bd;"
            "border-radius:16px;max-width:640px'><strong style='font-size:20px'>Forge is ready</strong>"
            "<p>Open Forge in your phone browser. Sign in as <b>forge</b> with the password for this session.</p>"
            "<a href='" + safe + "' target='_blank' rel='noopener noreferrer' style='display:inline-block;"
            "padding:14px 22px;min-height:24px;background:#174c78;color:white;border-radius:24px'>Open Forge</a>"
            "<p style='overflow-wrap:anywhere'>" + safe + "</p>"
            "<p>Keep this cell running. Stop it to end Forge, then disconnect and delete the Colab runtime. "
            "A new session gets a new address.</p></div>")


def ensure_port_available(port=7860):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"Port {port} is already in use. Stop your earlier Forge cell before starting another") from exc


def write_auth(directory: Path, prompt=None):
    prompt = prompt or getpass.getpass
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        password = prompt("Forge password (12+ characters; leave blank to generate a strong password): ")
    generated = not password
    if generated:
        password = secrets.token_urlsafe(24)
    if (password != password.strip() or len(password) < 12 or len(password) > 256 or
            any(c in password for c in ",:\r\n") or any(ord(c) < 32 or ord(c) == 127 for c in password)):
        raise ValueError("Use 12-256 characters, with no commas, colons, line breaks, invisible controls, or surrounding spaces")
    fd, filename = tempfile.mkstemp(prefix=".forge-auth-", dir=directory)
    path = Path(filename)
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("forge:" + password + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print("Forge username: forge")
    if generated:
        print("Generated password (keep this notebook output private):", password)
    return path, password


def stop_process_group(process):
    """Stop only the group created by this launcher, including its Qwen worker."""
    for sig, seconds in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()  # Reap the immediate child before checking the group.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                process.wait()
                return
            time.sleep(0.1)
    process.wait(timeout=2)


def forge_arguments(python: Path, drive: Path, auth: Path):
    return [python, "launch.py", "--skip-prepare-environment", "--share", "--listen",
            "--port", "7860", "--no-download-sd-model", "--gradio-auth-path", auth,
            "--models-dir", drive / "models", "--embeddings-dir", drive / "embeddings",
            "--ui-settings-file", drive / "state/config.json",
            "--ui-config-file", drive / "state/ui-config.json",
            "--gradio-allowed-path", drive / "outputs"]


def run_forge(args, root: Path, env, auth: Path, password: str, publish=None, log=None):
    process = None
    try:
        if publish is None:
            from IPython.display import HTML, display
            publish = lambda url: display(HTML(connection_html(url)))
        if log is None:
            def log(value):
                sys.stdout.write(value)
                sys.stdout.flush()
        parser = ShareOutputParser(publish, log, private=("forge:" + password, password))
        ensure_port_available()
        process = subprocess.Popen([str(value) for value in args], cwd=root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=0, start_new_session=True)
        interrupted = False
        try:
            while True:
                block = os.read(process.stdout.fileno(), 4096)
                if not block:
                    break
                parser.feed(block)
            parser.feed(b"", final=True)
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            log("Stopping this Forge session...\n")
            code = None
        if not interrupted and code:
            raise RuntimeError(f"Forge exited with code {code}. Correct the error above, then rerun this cell")
        if not interrupted and not parser.last_url:
            raise RuntimeError("Forge exited without a valid share address. Check the startup errors and sharing service")
        log("Forge stopped. Your saved settings and images remain in Drive.\n")
    finally:
        try:
            if process is not None:
                try:
                    stop_process_group(process)
                except KeyboardInterrupt:
                    # A second Stop press still terminates the owned worker
                    # group instead of leaving a hidden generation running.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
        finally:
            auth.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/ForgeColab"))
    parser.add_argument("--verify-sha256", action="store_true", help="Read and verify all manifest-listed model bytes; can be slow")
    parser.add_argument("--install-only", action="store_true", help="Install persistent Python environments on Drive without requiring model weights or starting Forge")
    parser.add_argument("--repair-environment", action="store_true", help="Retry package and extension installers without replacing saved settings or model files")
    args = parser.parse_args(argv)
    root, drive = ROOT.resolve(), args.drive_root.expanduser().resolve()
    validate_environment(root, drive)
    print("This is a browser session. Colab requires manual sign-in/Drive approval and a permitted GPU runtime with sufficient resources.", flush=True)
    with launch_lock(root):
        ensure_port_available()
        claim_runtime(root, drive)
        verify_vendor_manifest(root)
        if args.install_only:
            helper("colab_prepare").safe_target(drive, "models").mkdir(parents=True, exist_ok=True)
        else:
            result = helper("stage_models").stage(root, drive, verify_sha256=args.verify_sha256)
            print("Persistent models:", json.dumps(result), flush=True)
        result = helper("colab_prepare").prepare(root, drive)
        print("Persistent settings:", result["settings_directory"], flush=True)
        extensions = install_extensions(root)
        run([sys.executable, KIT / "mobile_patch.py", "--root", root], cwd=root)
        python, env = install_environments(root, drive, extensions, args.repair_environment)
        if args.install_only:
            print("Installation complete. Python and both environments are stored on Drive. Rerun without --install-only after models are ready.", flush=True)
            return
        # The password file is outside the Forge/Drive paths served by Gradio.
        auth, password = write_auth(Path("/content"))
        try:
            run_forge(forge_arguments(python, drive, auth), root, env, auth, password)
        finally:
            auth.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Forge setup stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None
