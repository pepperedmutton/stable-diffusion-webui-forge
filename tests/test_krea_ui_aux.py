import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
# Fresh clones keep extension fixtures in vendor until Colab installs them.
EXTENSION_ROOT = REPO_ROOT if (REPO_ROOT / "extensions/sd-forge-krea2/hf_config").is_dir() else REPO_ROOT / "colab/vendor"


def load_fake_initial_model(opts):
    source_path = REPO_ROOT / "modules" / "sd_models.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FakeInitialModel"
    )
    namespace = {
        "math": math,
        "os": os,
        "opts": opts,
        "paths": SimpleNamespace(script_path=str(EXTENSION_ROOT)),
        "threading": threading,
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["FakeInitialModel"]


def load_prompt_length_selector(opts, loaded_model):
    source_path = REPO_ROOT / "modules" / "sd_models.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected_nodes = [
        node for node in tree.body
        if (
            isinstance(node, ast.ClassDef) and node.name == "FakeInitialModel"
        ) or (
            isinstance(node, ast.FunctionDef) and node.name == "get_prompt_length_counter_for_ui"
        )
    ]
    namespace = {
        "math": math,
        "model_data": SimpleNamespace(sd_model=loaded_model),
        "os": os,
        "opts": opts,
        "paths": SimpleNamespace(script_path=str(EXTENSION_ROOT)),
        "threading": threading,
    }
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["get_prompt_length_counter_for_ui"]


class FakeInitialModelTests(unittest.TestCase):
    def test_krea_counter_uses_bundled_qwen_tokenizer_and_512_limit(self):
        from transformers import Qwen2Tokenizer

        fake_model_class = load_fake_initial_model(SimpleNamespace(forge_preset="krea"))
        model = fake_model_class()
        tokenizer_path = EXTENSION_ROOT / "extensions" / "sd-forge-krea2" / "hf_config" / "Krea2" / "tokenizer"
        tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        suffix_count = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])

        for prompt in ("", "a red cat", "cinematic full-body portrait, detailed armor"):
            expected = len(tokenizer(prompt.strip(), add_special_tokens=False)["input_ids"]) + suffix_count
            self.assertEqual(model.get_prompt_lengths_on_ui(prompt), (expected, 512))

    def test_krea_fallback_is_a_conservative_utf8_byte_count(self):
        fake_model_class = load_fake_initial_model(SimpleNamespace(forge_preset="krea"))
        fake_model_class._krea_tokenizer_failed = True
        model = fake_model_class()
        prompt = "red cat 月"

        count, limit = model.get_prompt_lengths_on_ui(prompt)

        self.assertEqual(count, len(prompt.encode("utf-8")) + 5)
        self.assertEqual(limit, 512)

    def test_non_krea_counter_keeps_legacy_chunk_size(self):
        fake_model_class = load_fake_initial_model(SimpleNamespace(forge_preset="xl"))

        self.assertEqual(fake_model_class().get_prompt_lengths_on_ui("red cat"), (2, 75))

    def test_krea_counter_ignores_a_stale_loaded_clip_model(self):
        class StaleClipModel:
            def __init__(self):
                self.calls = 0

            def get_prompt_lengths_on_ui(self, _prompt):
                self.calls += 1
                return 99, 75

        stale_model = StaleClipModel()
        select_counter = load_prompt_length_selector(
            SimpleNamespace(forge_preset="krea"),
            stale_model,
        )

        self.assertEqual(select_counter()("a red cat"), (8, 512))
        self.assertEqual(stale_model.calls, 0)

    def test_non_krea_selector_still_uses_the_loaded_model(self):
        class LoadedModel:
            def get_prompt_lengths_on_ui(self, _prompt):
                return 17, 225

        select_counter = load_prompt_length_selector(
            SimpleNamespace(forge_preset="xl"),
            LoadedModel(),
        )

        self.assertEqual(select_counter()("a red cat"), (17, 225))

    def test_non_krea_selector_keeps_the_missing_model_fallback(self):
        select_counter = load_prompt_length_selector(
            SimpleNamespace(forge_preset="xl"),
            None,
        )

        with self.assertRaises(AttributeError):
            select_counter()


class HiddenLuminaSettingsTests(unittest.TestCase):
    def test_lumina_defaults_are_hidden_states(self):
        source_path = REPO_ROOT / "modules" / "shared_options.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        expected = {
            "lumina_max_sequence_length",
            "lumina_sampler_shift",
            "lumina_cfg_normalization",
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.args[0], ast.Call) or len(node.args[0].args) < 2:
                continue
            section_call = node.args[0]
            if not isinstance(section_call.args[1], ast.Dict):
                continue
            keys = {
                key.value for key in section_call.args[1].keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if keys != expected:
                continue

            section = section_call.args[0]
            self.assertIsInstance(section, ast.Tuple)
            self.assertIsNone(section.elts[0].value)
            for value in section_call.args[1].values:
                self.assertIsInstance(value, ast.Call)
                self.assertGreaterEqual(len(value.args), 3)
                component = value.args[2]
                self.assertIsInstance(component, ast.Attribute)
                self.assertEqual((component.value.id, component.attr), ("gr", "State"))
            break
        else:
            self.fail("Lumina legacy defaults were not found")


class KreaLoraMetadataTests(unittest.TestCase):
    @staticmethod
    def load_network_classes():
        import enum
        import re

        source_path = REPO_ROOT / "extensions-builtin" / "sd_forge_lora" / "network.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        class_nodes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in {"SdVersion", "NetworkOnDisk"}
        ]
        namespace = {"enum": enum, "re": re}
        exec(compile(ast.Module(body=class_nodes, type_ignores=[]), str(source_path), "exec"), namespace)
        return namespace["SdVersion"], namespace["NetworkOnDisk"]

    def detect(self, name, metadata):
        sd_version, network_on_disk = self.load_network_classes()
        item = object.__new__(network_on_disk)
        item.name = name
        item.metadata = metadata
        return sd_version, item.detect_version()

    def test_explicit_incompatible_metadata_wins_over_krea_filename(self):
        sd_version, detected = self.detect(
            "portrait_krea2_v1",
            {"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"},
        )

        self.assertIs(detected, sd_version.SDXL)

    def test_krea_metadata_and_filename_fallback_are_detected(self):
        sd_version, detected_metadata = self.detect(
            "portrait",
            {"modelspec.architecture": "krea-2/lora"},
        )
        _, detected_name = self.detect("CocoaMixZero_detail", {})

        self.assertIs(detected_metadata, sd_version.Krea)
        self.assertEqual(detected_name.name, "Krea")

    def test_unidentified_lora_stays_unknown(self):
        sd_version, detected = self.detect("portrait", {})

        self.assertIs(detected, sd_version.Unknown)

    def test_metadata_editor_offers_krea(self):
        source = (
            REPO_ROOT / "extensions-builtin" / "sd_forge_lora" / "ui_edit_user_metadata.py"
        ).read_text(encoding="utf-8")

        self.assertIn("gr.Radio(['Krea', 'SD1', 'SD2', 'SDXL', 'Flux', 'Unknown']", source)


if __name__ == "__main__":
    unittest.main()
