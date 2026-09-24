"""Offline contracts for the Colab launcher; no GPU, installs, or model downloads."""
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("colab_launcher_test", "colab/start.py")
prepare = load("colab_prepare_test", "colab/kit/colab_prepare.py")
models = load("colab_models_test", "colab/kit/stage_models.py")
runtime_setup = load("colab_runtime_setup_test", "scripts/setup_qwen21_runtime.py")


class TemporaryTree(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.drive = self.base / "forge", self.base / "drive"
        self.root.mkdir()
        (self.root / "launch.py").write_text("# fixture\n")
        (self.drive / "models").mkdir(parents=True)

    def model(self, relative="Stable-diffusion/example.safetensors", content=b"weights"):
        path = self.drive / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def manifest(self, files=None):
        if files is None:
            files = [{"path": path.relative_to(self.drive / "models").as_posix(),
                      "bytes": path.stat().st_size, "sha256": models.sha256(path)}
                     for path in (self.drive / "models").rglob("*") if path.is_file()]
        (self.drive / "models-manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    def symlink(self, link, target, directory=False):
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            self.skipTest("Host cannot create symlinks: " + str(error))


class DriveLayoutTree(TemporaryTree):
    def setUp(self):
        super().setUp()
        self.root = self.drive / "forge"
        self.root.mkdir()
        (self.root / "launch.py").write_text("# persistent source fixture\n")


class SettingsTests(DriveLayoutTree):
    def test_fresh_clone_works_without_private_configs(self):
        result = prepare.prepare(self.root, self.drive)
        self.assertTrue(result["first_run_defaults_applied"])
        value = prepare.read_json(self.drive / "state/config.json")
        self.assertEqual(value["localization"], "None")
        self.assertEqual(value["outdir_img2img_samples"], (self.drive / "outputs/img2img-images").as_posix())
        self.assertNotIn("forge_preset", value)
        self.assertFalse((self.root / "config.json").exists())
        self.assertEqual(result["models_directory"], str(self.drive / "models"))
        self.assertEqual(result["embeddings_directory"], str(self.drive / "embeddings"))
        self.assertTrue((self.drive / "embeddings").is_dir())

    def test_wai_default_with_downloaded_weights(self):
        self.model("Stable-diffusion/" + prepare.DEFAULT_CHECKPOINT)
        self.model("diffusers/Qwen-Image-2.1-NF4/model_index.json", b"{}")
        prepare.prepare(self.root, self.drive)
        value = prepare.read_json(self.drive / "state/config.json")
        self.assertEqual(value["forge_preset"], "xl")
        self.assertEqual(value["sd_model_checkpoint"], prepare.DEFAULT_CHECKPOINT)
        self.assertEqual(value["forge_additional_modules"], [])

    def test_wai_default_after_install_created_state_before_download(self):
        prepare.prepare(self.root, self.drive)
        self.assertNotIn("sd_model_checkpoint", prepare.read_json(self.drive / "state/config.json"))
        self.model("Stable-diffusion/" + prepare.DEFAULT_CHECKPOINT)
        result = prepare.prepare(self.root, self.drive)
        self.assertFalse(result["first_run_defaults_applied"])
        value = prepare.read_json(self.drive / "state/config.json")
        self.assertEqual(value["sd_model_checkpoint"], prepare.DEFAULT_CHECKPOINT)
        self.assertEqual(value["forge_preset"], "xl")

    def test_migrated_novsw_is_replaced_by_wai_only(self):
        self.model("Stable-diffusion/" + prepare.DEFAULT_CHECKPOINT)
        config = {"sd_model_checkpoint": "D:/Forge/stable-diffusion-webui-forge/models/Stable-diffusion/Novsw.safetensors [abcd1234]", "forge_additional_modules": ["old-vae"], "forge_preset": "xl"}
        source = self.root / "config.json"
        source.write_text(json.dumps(config))
        original = source.read_bytes()
        prepare.prepare(self.root, self.drive)
        value = prepare.read_json(self.drive / "state/config.json")
        self.assertEqual(value["sd_model_checkpoint"], prepare.DEFAULT_CHECKPOINT)
        self.assertEqual(value["forge_checkpoint_xl"], prepare.DEFAULT_CHECKPOINT)
        self.assertEqual(value["forge_additional_modules"], [])
        self.assertFalse(value["forge_additional_modules_xl_configured"])
        self.assertEqual(source.read_bytes(), original)

    def test_other_migrated_selection_is_preserved(self):
        self.model("Stable-diffusion/" + prepare.DEFAULT_CHECKPOINT)
        config = {"sd_model_checkpoint": "Qwen-Image-2.1-INT8", "forge_preset": "qwen", "forge_qwen21_precision": "int8", "forge_checkpoint_xl": "Novsw.safetensors"}
        (self.root / "config.json").write_text(json.dumps(config))
        prepare.prepare(self.root, self.drive)
        value = prepare.read_json(self.drive / "state/config.json")
        self.assertEqual(value["sd_model_checkpoint"], config["sd_model_checkpoint"])
        self.assertEqual(value["forge_preset"], "qwen")
        self.assertEqual(value["forge_qwen21_precision"], "int8")
        self.assertEqual(value["forge_checkpoint_xl"], prepare.DEFAULT_CHECKPOINT)

    def test_missing_wai_does_not_persist_invalid_novsw_selection(self):
        (self.root / "config.json").write_text(json.dumps({"sd_model_checkpoint": "Novsw.safetensors"}))
        with self.assertRaisesRegex(ValueError, "Download WAI"):
            prepare.prepare(self.root, self.drive)
        self.assertFalse((self.drive / "state").exists())

    def test_saved_values_and_prompts_survive_a_restart(self):
        prepare.prepare(self.root, self.drive)
        path = self.drive / "state/config.json"
        value = prepare.read_json(path)
        value.update({"forge_preset": "xl", "sd_model_checkpoint": "mine.safetensors", "prompt": r"D:\Forge\literal\prompt", "qwen21_t2i_steps": 33})
        prepare.atomic_json(path, value)
        ui = {"txt2img/Width/value": 768, "txt2img/Prompt/value": "/content/forge/a picture"}
        prepare.atomic_json(self.drive / "state/ui-config.json", ui)
        prepare.prepare(self.root, self.drive)
        value = prepare.read_json(path)
        self.assertEqual(value["forge_preset"], "xl")
        self.assertEqual(value["prompt"], r"D:\Forge\literal\prompt")
        self.assertEqual(value["qwen21_t2i_steps"], 33)
        self.assertEqual(prepare.read_json(self.drive / "state/ui-config.json"), ui)

    def test_old_and_windows_paths_remap_without_prefix_false_positive(self):
        for old in ("D:\\Forge\\stable-diffusion-webui-forge\\models\\a", "/content/forge/models/a", "/content/forge-web/models/a"):
            with self.subTest(old=old):
                self.assertEqual(prepare.remap_path(old, self.root, [], "model"), (self.drive / "models/a").as_posix())
        self.assertEqual(prepare.remap_path("/content/forgery/a", self.root, [], "model"), "/content/forgery/a")

    def test_relative_and_current_source_models_use_sibling_storage(self):
        for value in ("models/text_encoder/a.safetensors", (self.root / "models/text_encoder/a.safetensors").as_posix()):
            self.assertEqual(prepare.remap_path(value, self.root, [], "module"), (self.drive / "models/text_encoder/a.safetensors").as_posix())
        value = (self.drive / "models/text_encoder/a.safetensors").as_posix()
        self.assertEqual(prepare.remap_path(value, self.root, [], "module"), value)
        self.assertEqual(prepare.remap_path("D:/Forge/stable-diffusion-webui-forge/embeddings/a.pt", self.root, [], "input"), (self.drive / "embeddings/a.pt").as_posix())
        self.assertEqual(prepare.remap_path("/content/forge/extensions/x", self.root, [], "input"), (self.root / "extensions/x").as_posix())

    def test_module_lists_and_batch_paths_migrate_but_prompt_text_does_not(self):
        config = {"forge_additional_modules_anima": ["D:/Forge/stable-diffusion-webui-forge/models/text_encoder/a.safetensors"], "styles_file": "/content/forge/styles.csv"}
        ui = {"img2img/Input directory/value": "D:/Forge/stable-diffusion-webui-forge/embeddings", "txt2img/Prompt/value": "models/leave-this-text"}
        normalized, normalized_ui, _, warnings = prepare.normalize_settings(config, ui, self.root, self.drive, False)
        self.assertEqual(normalized["forge_additional_modules_anima"], [(self.drive / "models/text_encoder/a.safetensors").as_posix()])
        self.assertEqual(normalized["styles_file"], (self.root / "styles.csv").as_posix())
        self.assertEqual(normalized_ui["img2img/Input directory/value"], (self.drive / "embeddings").as_posix())
        self.assertEqual(normalized_ui["txt2img/Prompt/value"], ui["txt2img/Prompt/value"])
        self.assertFalse(warnings)

    def test_unmapped_paths_fail_before_writing_state(self):
        (self.root / "config.json").write_text(json.dumps({"sd_model_checkpoint": "Z:/private/model"}))
        with self.assertRaisesRegex(ValueError, "Unmapped Windows path"):
            prepare.prepare(self.root, self.drive)
        self.assertFalse((self.drive / "state").exists())

    def test_config_input_is_untouched_gpu_budget_removed(self):
        value = {"forge_inference_memory": 8192, "forge_XL_GPU_MB": 12000, "forge_preset": "xl"}
        path = self.root / "config.json"
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        prepare.prepare(self.root, self.drive)
        self.assertEqual(path.read_bytes(), before)
        persisted = prepare.read_json(self.drive / "state/config.json")
        self.assertNotIn("forge_inference_memory", persisted)
        self.assertNotIn("forge_XL_GPU_MB", persisted)

    def test_invalid_json_does_not_replace_valid_state(self):
        (self.drive / "state").mkdir()
        valid = '{"forge_preset":"xl"}'
        (self.drive / "state/config.json").write_text(valid)
        (self.drive / "state/ui-config.json").write_text("broken")
        with self.assertRaises(ValueError):
            prepare.prepare(self.root, self.drive)
        self.assertEqual((self.drive / "state/config.json").read_text(), valid)

    def test_settings_symlink_escape_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.symlink(self.drive / "state", outside, directory=True)
        with self.assertRaises(ValueError):
            prepare.prepare(self.root, self.drive)
        self.assertEqual(list(outside.iterdir()), [])

    def test_path_overlap_rejected(self):
        with self.assertRaises(ValueError):
            prepare.prepare(self.root, self.root / "nested")

    def test_source_must_be_named_forge_directly_inside_storage(self):
        for root in (self.base / "forge", self.drive, self.drive / "nested/forge"):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "directly inside"):
                    prepare.prepare(root, self.drive)

    def test_embedding_symlink_rejected_before_state_writes(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.symlink(self.drive / "embeddings", outside, directory=True)
        with self.assertRaises(ValueError):
            prepare.prepare(self.root, self.drive)
        self.assertFalse((self.drive / "state").exists())


class ModelTests(DriveLayoutTree):
    def test_default_reads_direct_drive_paths_without_link_copy_or_hash(self):
        source = self.model()
        before = source.read_bytes()
        with mock.patch.object(Path, "symlink_to", side_effect=AssertionError("Never create model links")), mock.patch.object(models, "sha256", side_effect=AssertionError("Default only checks model sizes")):
            result = models.stage(self.root, self.drive)
            models.stage(self.root, self.drive)
        self.assertFalse((self.root / "models").exists())
        self.assertEqual(result["models_directory"], str(self.drive / "models"))
        self.assertEqual(result["embeddings_directory"], str(self.drive / "embeddings"))
        self.assertEqual(result["runtime_model_copies"], 0)
        self.assertEqual(result["model_links_created"], 0)
        self.assertEqual(result["models_downloaded"], 0)
        self.assertEqual(source.read_bytes(), before)

    def test_manifest_size_mismatch_leaves_source_tree_untouched(self):
        self.model()
        self.manifest([{"path": "Stable-diffusion/example.safetensors", "bytes": 900}])
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            models.stage(self.root, self.drive)
        self.assertFalse((self.root / "models").exists())

    def test_hash_verification_rejects_same_size_corruption(self):
        path = self.model()
        self.manifest()
        path.write_bytes(b"WEIGHTS")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            models.stage(self.root, self.drive, verify_sha256=True)

    def test_hash_flag_requires_manifest_and_hash(self):
        self.model()
        with self.assertRaisesRegex(ValueError, "manifest"):
            models.stage(self.root, self.drive, verify_sha256=True)
        self.manifest([{"path": "Stable-diffusion/example.safetensors", "bytes": 7}])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            models.stage(self.root, self.drive, verify_sha256=True)

    def test_extra_uploaded_models_remain_available(self):
        self.model("a.bin")
        self.manifest()
        self.model("b.bin")
        result = models.stage(self.root, self.drive)
        self.assertEqual(result["extra_files"], 1)
        self.assertTrue((self.drive / "models/b.bin").is_file())
        self.assertFalse((self.root / "models").exists())

    def test_manifest_traversal_duplicate_and_invalid_size_rejected(self):
        self.model()
        for entry in ({"path": "../secret", "bytes": 0}, {"path": "C:/secret", "bytes": 0},
                      {"path": "a\\b", "bytes": 0}, {"path": "a", "bytes": True},
                      {"path": "a", "bytes": -1}):
            with self.subTest(entry=entry):
                self.manifest([entry])
                with self.assertRaises(ValueError):
                    models.stage(self.root, self.drive)
        entry = {"path": "Stable-diffusion/example.safetensors", "bytes": 7}
        self.manifest([entry, entry])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            models.stage(self.root, self.drive)

    def test_source_checkout_model_placeholders_are_not_used_or_changed(self):
        self.model("a.bin")
        target = self.root / "models/a.bin"
        target.parent.mkdir()
        target.write_bytes(b"user changes")
        result = models.stage(self.root, self.drive)
        self.assertEqual(result["models_directory"], str(self.drive / "models"))
        self.assertEqual(target.read_bytes(), b"user changes")

    def test_identical_tracked_helper_kept(self):
        self.model("helper.txt", b"helper")
        (self.root / "models").mkdir()
        (self.root / "models/helper.txt").write_bytes(b"helper")
        models.stage(self.root, self.drive)
        self.assertFalse((self.root / "models/helper.txt").is_symlink())

    def test_legacy_cache_request_is_rejected_without_copying(self):
        source = self.model("a.bin")
        with self.assertRaisesRegex(ValueError, "caching is disabled"):
            models.stage(self.root, self.drive, cache=["a.bin"])
        self.assertFalse((self.root / "models").exists())
        self.assertEqual(source.read_bytes(), b"weights")

    def test_stage_is_read_only_when_embeddings_are_absent(self):
        self.model("a.bin")
        before = sorted(str(path.relative_to(self.drive)) for path in self.drive.rglob("*"))
        result = models.stage(self.root, self.drive)
        self.assertEqual(result["embedding_files"], 0)
        self.assertEqual(before, sorted(str(path.relative_to(self.drive)) for path in self.drive.rglob("*")))

    def test_empty_collection_and_wrong_source_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            models.stage(self.root, self.drive)
        self.model("a.bin")
        with self.assertRaisesRegex(ValueError, "directly inside"):
            models.stage(self.base / "forge", self.drive)

    def test_verify_only_makes_no_model_directory(self):
        self.model()
        self.manifest()
        result = models.stage(self.root, self.drive, verify_sha256=True, verify_only=True)
        self.assertTrue(result["verification_only"])
        self.assertFalse((self.root / "models").exists())

    def test_source_escape_rejected(self):
        outside = self.base / "secret"
        outside.write_bytes(b"outside")
        self.symlink(self.drive / "models/secret", outside)
        with self.assertRaises(ValueError):
            models.stage(self.root, self.drive)

    def test_unused_source_model_symlink_is_not_followed(self):
        self.model("a.bin")
        outside = self.base / "outside"
        outside.mkdir()
        self.symlink(self.root / "models", outside, directory=True)
        result = models.stage(self.root, self.drive)
        self.assertEqual(result["models_directory"], str(self.drive / "models"))
        self.assertEqual(list(outside.iterdir()), [])

    def test_unrelated_source_model_link_preserved(self):
        self.model("a.bin")
        outside = self.base / "other.bin"
        outside.write_bytes(b"other")
        self.symlink(self.root / "models/a.bin", outside)
        models.stage(self.root, self.drive)
        self.assertEqual(outside.read_bytes(), b"other")

    def test_embeddings_are_counted_without_copy_and_escapes_rejected(self):
        self.model()
        embedding = self.drive / "embeddings" / "example.pt"
        embedding.parent.mkdir()
        embedding.write_bytes(b"embedding")
        result = models.stage(self.root, self.drive)
        self.assertEqual(result["embedding_files"], 1)
        self.assertEqual(embedding.read_bytes(), b"embedding")
        self.assertFalse((self.root / "embeddings").exists())
        outside = self.base / "other.pt"
        outside.write_bytes(b"outside")
        self.symlink(embedding.parent / "escape.pt", outside)
        with self.assertRaisesRegex(ValueError, "Embedding files"):
            models.stage(self.root, self.drive)


class AuthenticationAndShareTests(TemporaryTree):
    def test_url_acceptance_and_attack_rejection(self):
        self.assertEqual(launcher.validate_share_url("HTTPS://A1-B2.GRADIO.LIVE"), "https://a1-b2.gradio.live/")
        for value in ("http://a.gradio.live", "https://gradio.live", "https://a.b.gradio.live",
                      "https://a.gradio.live.evil.test", "https://a.gradio.live@evil.test",
                      "https://user:pass@a.gradio.live", "https://a.gradio.live:443",
                      "https://a.gradio.live/path", "https://a.gradio.live?auth=1", "https://a.gradio.live#x",
                      "https://a.gradio.live\\evil", "https://a%2egradio.live", "https://a.gradio.live\x00", " https://a.gradio.live"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    launcher.validate_share_url(value)

    def test_split_chunks_redact_password_and_defer_full_url(self):
        urls, logs = [], []
        parser = launcher.ShareOutputParser(urls.append, logs.append, private=("top-secret-pass",))
        parser.feed(b"running at https://abc.gradio.live")
        self.assertEqual(urls, [])
        parser.feed(b".evil.test\npassword=top-sec")
        parser.feed(b"ret-pass\n\x1b[32mhttps://good.gradio.live/\x1b[0m\n")
        parser.feed(b"https://good.gradio.live/\n")
        self.assertEqual(urls, ["https://good.gradio.live/"])
        self.assertNotIn("top-secret-pass", "".join(logs))
        self.assertIn("[private]", "".join(logs))

    def test_long_output_is_not_published(self):
        urls = []
        parser = launcher.ShareOutputParser(urls.append, lambda _: None)
        with self.assertRaises(ValueError):
            parser.feed(b"x" * 65537 + b"\n")
        self.assertEqual(urls, [])

    def test_auth_uses_getpass_and_an_ephemeral_file(self):
        with mock.patch.object(launcher.getpass, "getpass", return_value="a-strong-password"), mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            path, password = launcher.write_auth(self.base)
        self.assertEqual(path.read_text(), "forge:a-strong-password\n")
        self.assertEqual(password, "a-strong-password")
        self.assertNotIn(password, stdout.getvalue())
        self.assertNotIn(str(self.drive), str(path))
        path.unlink()

    def test_generated_password_is_strong_and_shown_once(self):
        with mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            path, password = launcher.write_auth(self.base, prompt=lambda _: "")
        self.assertGreaterEqual(len(password), 32)
        self.assertEqual(stdout.getvalue().count(password), 1)
        path.unlink()

    def test_invalid_passwords_create_no_secret_files(self):
        for password in ("short", "long,unsafe-password", "long:unsafe-password", " has-surrounding-space", "hidden\x00character", "newline\npassword"):
            with self.subTest(password=password):
                with self.assertRaises(ValueError):
                    launcher.write_auth(self.base, prompt=lambda _: password)
        self.assertEqual(list(self.base.glob(".forge-auth-*")), [])

    def test_password_never_in_process_arguments(self):
        args = launcher.forge_arguments(Path("python"), self.drive, self.base / ".auth")
        self.assertIn("--gradio-auth-path", args)
        self.assertNotIn("--gradio-auth", args)
        self.assertIn("--skip-prepare-environment", args)
        self.assertIn("--no-download-sd-model", args)
        self.assertNotIn("--enable-insecure-extension-access", args)
        self.assertEqual(args[args.index("--clip-models-path") + 1], self.drive / "models/CLIP")

    def test_failed_spawn_removes_password_file(self):
        auth = self.base / "auth"
        auth.write_text("forge:a-strong-password")
        with mock.patch.object(launcher, "ensure_port_available"), mock.patch.object(launcher.subprocess, "Popen", side_effect=OSError("spawn failed")):
            with self.assertRaises(OSError):
                launcher.run_forge(["python"], self.root, {}, auth, "a-strong-password", publish=lambda _: None, log=lambda _: None)
        self.assertFalse(auth.exists())

    def test_early_display_setup_failure_removes_password_file(self):
        auth = self.base / "auth"
        auth.write_text("forge:a-strong-password")
        with mock.patch.dict(launcher.sys.modules, {"IPython.display": None}):
            with self.assertRaises(ImportError):
                launcher.run_forge(["python"], self.root, {}, auth, "a-strong-password")
        self.assertFalse(auth.exists())

    def test_interrupt_stops_owned_group_and_removes_auth(self):
        auth = self.base / "auth"
        auth.write_text("forge:a-strong-password")
        process = mock.Mock()
        process.stdout.fileno.return_value = 17
        with mock.patch.object(launcher, "ensure_port_available"), mock.patch.object(launcher.subprocess, "Popen", return_value=process) as popen, mock.patch.object(launcher.os, "read", side_effect=KeyboardInterrupt), mock.patch.object(launcher, "stop_process_group") as stop:
            launcher.run_forge(["python"], self.root, {}, auth, "a-strong-password", publish=lambda _: None, log=lambda _: None)
        stop.assert_called_once_with(process)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertFalse(auth.exists())

    def test_html_has_only_browser_link(self):
        value = launcher.connection_html("https://safe.gradio.live")
        self.assertIn("Open Forge", value)
        self.assertIn("noopener noreferrer", value)
        self.assertNotIn("intent:", value)
        self.assertNotIn("forgepocket", value.lower())


class RuntimeAndVendorTests(DriveLayoutTree):
    def extension(self):
        path = self.root / "colab/vendor/extensions/sd-forge-krea2"
        path.mkdir(parents=True)
        (path / "LICENSE").write_bytes(b"license")
        (path / "module.py").write_bytes(b"answer = 42\n")
        return path

    def test_runtime_claim_is_idempotent_and_storage_bound(self):
        launcher.claim_runtime(self.root, self.drive)
        launcher.claim_runtime(self.root, self.drive)
        with self.assertRaisesRegex(ValueError, "persistent checkout"):
            launcher.claim_runtime(self.root, self.base / "another")

    def test_unmanaged_environment_never_overwritten(self):
        (self.root / "venv").mkdir()
        (self.root / "venv/user.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            launcher.claim_runtime(self.root, self.drive)
        self.assertEqual((self.root / "venv/user.txt").read_text(), "keep")
        self.assertFalse((self.root / ".colab/runtime.json").exists())

    def test_extensions_copy_and_resume_but_preserve_user_changes(self):
        source = self.extension()
        launcher.install_extensions(self.root)
        launcher.install_extensions(self.root)
        installed = self.root / "extensions/sd-forge-krea2/module.py"
        installed.write_bytes(b"user changes")
        with self.assertRaisesRegex(ValueError, "local changes"):
            launcher.install_extensions(self.root)
        self.assertEqual(installed.read_bytes(), b"user changes")
        self.assertEqual((source / "module.py").read_bytes(), b"answer = 42\n")

    def test_existing_unmanaged_extension_is_preserved(self):
        self.extension()
        installed = self.root / "extensions/sd-forge-krea2"
        installed.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            launcher.install_extensions(self.root)
        self.assertEqual(list(installed.iterdir()), [])

    def test_interrupted_extension_marker_write_can_resume_without_recopy(self):
        source = self.extension()
        installed = self.root / "extensions/sd-forge-krea2"
        import shutil
        shutil.copytree(source, installed)
        with mock.patch.object(launcher.shutil, "copytree", side_effect=AssertionError("Exact copy must be reused")):
            launcher.install_extensions(self.root)
        self.assertTrue((self.root / ".colab/extensions/sd-forge-krea2.json").is_file())

    def test_vendor_integrity_detects_modified_and_extra_files(self):
        source = self.extension()
        vendor = self.root / "colab/vendor"
        (vendor / "repositories").mkdir()
        files = [{"path": path.relative_to(vendor).as_posix(), "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in source.iterdir()]
        (vendor / "manifest.json").write_text(json.dumps({"files": files}))
        self.assertEqual(launcher.verify_vendor_manifest(self.root), 2)
        (source / "extra.py").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            launcher.verify_vendor_manifest(self.root)
        (source / "extra.py").unlink()
        (source / "module.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            launcher.verify_vendor_manifest(self.root)

    def test_dependency_overlay_preserves_pin_and_detects_edits(self):
        source = self.root / "colab/vendor/repositories/huggingface_guess"
        source.mkdir(parents=True)
        (source / "model.py").write_bytes(b"custom model")
        destination = self.root / "repositories/huggingface_guess"
        (destination / ".git").mkdir(parents=True)
        (destination / "model.py").write_bytes(b"upstream model")
        with mock.patch.object(launcher.subprocess, "check_output", return_value=launcher.GUESS_REVISION), mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"upstream model")) as git:
            launcher.install_dependency_overlay(self.root)
            launcher.install_dependency_overlay(self.root)
            self.assertEqual(git.call_count, 1)
            self.assertIn("show", git.call_args.args[0])
            (destination / "model.py").write_bytes(b"user edit")
            with self.assertRaisesRegex(ValueError, "local changes"):
                launcher.install_dependency_overlay(self.root)
        self.assertEqual((destination / "model.py").read_bytes(), b"user edit")

    def test_origin_allowlist(self):
        for value in (launcher.REMOTE, launcher.REMOTE + ".git", "git@github.com:pepperedmutton/stable-diffusion-webui-forge.git"):
            self.assertTrue(launcher.expected_remote(value))
        for value in (launcher.REMOTE + "-evil", "https://evil.test/pepperedmutton/stable-diffusion-webui-forge", launcher.REMOTE + "?token=secret"):
            self.assertFalse(launcher.expected_remote(value))

    def test_launcher_rejects_windows_before_any_install(self):
        with mock.patch.object(launcher.sys, "platform", "win32"), mock.patch.object(launcher.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Linux Google Colab"):
                launcher.validate_environment(self.root, self.drive)
        run.assert_not_called()

    def test_qwen_site_packages_comes_from_selected_interpreter(self):
        with mock.patch.object(runtime_setup.subprocess, "check_output", return_value="/runtime/lib/python3.10/site-packages\n") as query:
            result = runtime_setup.site_packages(Path("/runtime/bin/python"))
        self.assertEqual(result.as_posix(), "/runtime/lib/python3.10/site-packages")
        self.assertEqual(query.call_args.args[0][0], str(Path("/runtime/bin/python")))

    def test_uv_bootstrap_does_not_require_host_venv_or_modify_host_packages(self):
        binary = self.root / ".colab/uv/bin/uv"

        def installed(command):
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fixture executable")

        with mock.patch.object(launcher, "run", side_effect=installed) as run, mock.patch.object(launcher.os, "access", return_value=True):
            self.assertEqual(launcher.install_uv(self.root), binary)
        command = run.call_args.args[0]
        self.assertIn("--target", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--no-compile", command)
        self.assertIn("uv==0.8.22", command)
        self.assertNotIn("venv", command)
        with mock.patch.object(launcher, "run") as run, mock.patch.object(launcher.os, "access", return_value=True):
            self.assertEqual(launcher.install_uv(self.root), binary)
        run.assert_not_called()

    def test_t4_is_rejected_before_package_installation(self):
        google, colab = ModuleType("google"), ModuleType("google.colab")
        google.colab = colab
        with mock.patch.dict(launcher.sys.modules, {"google": google, "google.colab": colab}), mock.patch.object(launcher.sys, "platform", "linux"), mock.patch.object(Path, "is_dir", return_value=True), mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(Path, "is_symlink", return_value=False), mock.patch.object(launcher.os.path, "ismount", return_value=True), mock.patch.object(launcher.subprocess, "check_output", side_effect=[launcher.REMOTE, "Tesla T4, 7.5, 15360 MiB\n"]), mock.patch.object(launcher.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "T4 is unsupported"):
                launcher.validate_environment(Path("/content/drive/MyDrive/ForgeColab/forge"), Path("/content/drive/MyDrive/ForgeColab"))
        run.assert_not_called()

    def test_modern_pip_and_setuptools_are_ready_before_cuda_only_torch_install(self):
        (self.root / ".colab").mkdir()
        (self.root / "requirements_versions.txt").write_text("setuptools==69.5.1\n")
        python = self.root / "venv/bin/python"
        uv = self.root / ".colab/uv/bin/uv"
        for existing_environment in (False, True):
            with self.subTest(existing_environment=existing_environment):
                if existing_environment:
                    python.parent.mkdir(parents=True, exist_ok=True)
                    python.write_bytes(b"managed interpreter fixture")
                commands = []
                setuptools_version = "84.0.0"
                pip_version = "bundled-old-pip"

                def simulated_run(command, **kwargs):
                    nonlocal setuptools_version, pip_version
                    command = list(map(str, command))
                    commands.append(command)
                    if "env" in kwargs:
                        self.assertNotIn("CIVITAI_API_KEY", kwargs["env"])
                        self.assertEqual(kwargs["env"]["PIP_COMPILE"], "0")
                        self.assertEqual(kwargs["env"]["PIP_NO_COMPILE"], "0")
                        self.assertEqual(kwargs["env"]["UV_COMPILE_BYTECODE"], "false")
                    if command[1:4] == ["-m", "pip", "install"]:
                        self.assertEqual(command[0], str(python))
                        self.assertIn("--no-compile", command)
                        constraints = Path(kwargs["env"]["PIP_CONSTRAINT"])
                        self.assertIn("setuptools==69.5.1", constraints.read_text())
                        if "pip==26.2.1" in command:
                            self.assertEqual(command[command.index("--index-url") + 1], "https://pypi.org/simple")
                            self.assertIn("--only-binary=:all:", command)
                            pip_version = "26.2.1"
                        if "setuptools==69.5.1" in command:
                            self.assertEqual(pip_version, "26.2.1")
                            self.assertEqual(command[command.index("--index-url") + 1], "https://pypi.org/simple")
                            setuptools_version = "69.5.1"
                        if f"torch=={launcher.TORCH}" in command:
                            self.assertEqual(pip_version, "26.2.1", "Bundled pip mishandles current normalized wheel metadata names")
                            self.assertEqual(command[command.index("--index-url") + 1], launcher.CUDA_INDEX)
                            self.assertEqual(setuptools_version, "69.5.1", "Triton cannot obtain Forge's setuptools pin from the CUDA-only index")
                    return SimpleNamespace(returncode=0, stdout="")

                with mock.patch.dict(launcher.os.environ, {"PIP_COMPILE": "1", "PIP_NO_COMPILE": "1", "UV_COMPILE_BYTECODE": "true", "CIVITAI_API_KEY": "fixture-do-not-inherit"}), mock.patch.object(launcher, "run", side_effect=simulated_run), mock.patch.object(launcher, "install_uv", return_value=uv), mock.patch.object(launcher, "persistent_python", return_value=self.root / ".colab/python/bin/python3.10"), mock.patch.object(launcher, "installation_signature", return_value="fixture"), mock.patch.object(launcher.subprocess, "check_output", return_value="3.10\n"), mock.patch.object(launcher, "install_dependency_overlay"), mock.patch.object(launcher, "report_optional_components", return_value={}):
                    result, _ = launcher.install_environments(self.root, self.drive, [], repair=existing_environment)
                self.assertEqual(result, python)
                pip_installs = [command for command in commands if command[1:4] == ["-m", "pip", "install"]]
                self.assertIn("pip==26.2.1", pip_installs[0])
                self.assertIn("setuptools==69.5.1", pip_installs[1])
                self.assertIn(f"torch=={launcher.TORCH}", pip_installs[2])
                prepare_command = next(command for command in commands if "launch.py" in command)
                self.assertEqual(prepare_command[prepare_command.index("--clip-models-path") + 1], str(self.drive / "models/CLIP"))
                faceid_install = next(command for command in pip_installs if "insightface==1.0.1" in command)
                self.assertIn("--only-binary=:all:", faceid_install)
                self.assertIn("onnx==1.12.0", faceid_install)
                self.assertIn("onnxruntime==1.23.2", faceid_install)
                self.assertFalse(any("onnxruntime-gpu" in value for value in faceid_install))
                self.assertLess(commands.index(faceid_install), commands.index(prepare_command))
                self.assertEqual(json.loads((self.root / ".colab/installed.json").read_text())["signature"], "fixture")

    def test_persistent_python_survives_bootstrap_removal_and_uses_no_hardlinks(self):
        bootstrap = self.base / "temporary-python"
        source = bootstrap / "bin/python3.10"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"ELF fixture")
        (bootstrap / "lib/python3.10").mkdir(parents=True)
        (bootstrap / "lib/python3.10/os.py").write_text("# standard library fixture")
        with mock.patch.object(launcher.subprocess, "check_output", side_effect=[str(source), str(bootstrap)]), mock.patch.object(launcher, "run"), mock.patch.object(launcher.os, "link", side_effect=AssertionError("Drive hard links are unsupported")):
            persistent = launcher.persistent_python(self.root, Path("uv"), {"UV_LINK_MODE": "copy"})
        launcher.shutil.rmtree(bootstrap)
        self.assertEqual(persistent, self.root / ".colab/python/bin/python3.10")
        self.assertEqual(persistent.read_bytes(), b"ELF fixture")
        self.assertTrue((self.root / ".colab/python/lib/python3.10/os.py").is_file())
        with mock.patch.object(launcher.subprocess, "check_output") as query, mock.patch.object(launcher, "run") as run:
            self.assertEqual(launcher.persistent_python(self.root, Path("uv"), {}), persistent)
        query.assert_not_called()
        run.assert_not_called()

    def test_incomplete_persistent_python_is_never_overwritten(self):
        directory = self.root / ".colab/python"
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("user data")
        with self.assertRaisesRegex(ValueError, "unmanaged or incomplete"):
            launcher.persistent_python(self.root, Path("uv"), {})
        self.assertEqual((directory / "keep.txt").read_text(), "user data")

    def test_venv_uses_persistent_base_and_copies(self):
        base = self.root / ".colab/python/bin/python3.10"
        destination = self.root / "venv"
        with mock.patch.object(launcher, "run") as run:
            result = launcher.create_persistent_venv(base, destination, {"UV_LINK_MODE": "copy"})
        self.assertEqual(result, destination / "bin/python")
        self.assertTrue((destination / "lib64").is_dir())
        self.assertFalse((destination / "lib64").is_symlink())
        self.assertEqual(run.call_args.args[0], [base, "-m", "venv", "--copies", "--without-pip", destination])

    def test_reconnect_reuses_drive_environments_without_installing_or_copying(self):
        (self.root / ".colab").mkdir()
        (self.root / "requirements_versions.txt").write_text("setuptools==69.5.1\n")
        for relative in ("venv/bin/python", "runtimes/qwen-image-2.1/bin/python"):
            python = self.root / relative
            python.parent.mkdir(parents=True)
            python.write_bytes(b"persistent interpreter fixture")
        (self.root / ".colab/installed.json").write_text(json.dumps({"signature": "fixture"}))
        with mock.patch.object(launcher, "installation_signature", return_value="fixture"), mock.patch.object(launcher, "run") as run, mock.patch.object(launcher, "install_uv") as bootstrap, mock.patch.object(launcher, "persistent_python") as copy_python, mock.patch.object(launcher, "install_dependency_overlay"), mock.patch.object(launcher, "report_optional_components", return_value={}), mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout="libGL.so.1 libglib-2.0.so.0 libcairo.so.2")):
            _, env = launcher.install_environments(self.root, self.drive, [])
        bootstrap.assert_not_called()
        copy_python.assert_not_called()
        self.assertEqual(env["UV_LINK_MODE"], "copy")
        self.assertEqual(env["PIP_COMPILE"], "0")
        self.assertEqual(env["PIP_NO_COMPILE"], "0")
        self.assertEqual(env["UV_COMPILE_BYTECODE"], "false")
        self.assertTrue(any(launcher.FACEID_PROBE in call.args[0] for call in run.call_args_list))
        for call in run.call_args_list:
            self.assertEqual(call.args[0][1], "-c")
            self.assertTrue(call.args[0][0].is_relative_to(self.root))

    def test_faceid_probe_requires_the_cpu_provider_used_by_forge(self):
        app, utils, runtime = ModuleType("insightface.app"), ModuleType("insightface.utils"), ModuleType("onnxruntime")
        app.FaceAnalysis = lambda: None
        utils.face_align = SimpleNamespace(norm_crop=lambda: None)
        versions = {"insightface": "1.0.1", "onnx": "1.12.0", "onnxruntime": "1.23.2"}
        for providers, accepted in ((["CPUExecutionProvider"], True), (["CUDAExecutionProvider"], False)):
            with self.subTest(providers=providers):
                runtime.get_available_providers = lambda: providers
                with mock.patch.dict(launcher.sys.modules, {"insightface.app": app, "insightface.utils": utils, "onnxruntime": runtime}), mock.patch("importlib.metadata.version", side_effect=versions.__getitem__):
                    if accepted:
                        exec(launcher.FACEID_PROBE, {})
                    else:
                        with self.assertRaises(AssertionError):
                            exec(launcher.FACEID_PROBE, {})

    def test_install_only_does_not_require_weights_or_start_public_server(self):
        with mock.patch.object(launcher, "ROOT", self.root), mock.patch.object(launcher, "validate_environment"), mock.patch.object(launcher, "launch_lock"), mock.patch.object(launcher, "ensure_port_available"), mock.patch.object(launcher, "verify_vendor_manifest"), mock.patch.object(launcher, "install_extensions", return_value=[]), mock.patch.object(launcher, "run"), mock.patch.object(launcher, "install_environments", return_value=(self.root / "venv/bin/python", {})), mock.patch.object(launcher, "write_auth") as auth, mock.patch.object(launcher, "run_forge") as forge:
            launcher.main(["--drive-root", str(self.drive), "--install-only"])
        self.assertEqual(list((self.drive / "models").iterdir()), [])
        auth.assert_not_called()
        forge.assert_not_called()

    def test_optional_missing_features_are_reported_without_claiming_success(self):
        output = SimpleNamespace(stdout='COLAB_OPTIONAL_STATUS={"IP-Adapter FaceID":"ModuleNotFoundError"}\n')
        with mock.patch.object(launcher, "run", return_value=output), mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            missing = launcher.report_optional_components(Path("python"), {})
        self.assertIn("IP-Adapter FaceID", missing)
        self.assertIn("OPTIONAL FEATURES UNAVAILABLE", stdout.getvalue())
        self.assertIn("--repair-environment", stdout.getvalue())

    def test_legacy_mediapipe_api_is_actually_probed(self):
        self.assertIn("imported.solutions.face_mesh", launcher.OPTIONAL_PROBE)


if __name__ == "__main__":
    unittest.main()
