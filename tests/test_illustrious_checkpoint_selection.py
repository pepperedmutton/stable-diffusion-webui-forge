import ast
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from modules_forge.checkpoint_inspection import inspect_checkpoint_file


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ENTRY_PATH = REPO_ROOT / "modules_forge" / "main_entry.py"
SHARED_OPTIONS_PATH = REPO_ROOT / "modules_forge" / "shared_options.py"
SYSINFO_PATH = REPO_ROOT / "modules" / "sysinfo.py"
MODEL_TYPE_KEY = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
BUILTIN_VAE_KEY = "first_stage_model.decoder.conv_in.weight"


def write_checkpoint_header(path, cross_attention_dim=None, include_vae=False, metadata=None):
    header = {}
    if cross_attention_dim is not None:
        header[MODEL_TYPE_KEY] = {
            "dtype": "F32",
            "shape": [640, cross_attention_dim],
            "data_offsets": [0, 0],
        }
    if include_vae:
        header[BUILTIN_VAE_KEY] = {
            "dtype": "F32",
            "shape": [512, 4, 3, 3],
            "data_offsets": [0, 0],
        }
    if metadata is not None:
        header["__metadata__"] = metadata

    encoded_header = json.dumps(header).encode("utf-8")
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(encoded_header)))
        file.write(encoded_header)


def get_function(name):
    tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def load_function(name, namespace):
    node = get_function(name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(MAIN_ENTRY_PATH), "exec"), namespace)
    return namespace[name]


def load_functions(names, namespace):
    tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(MAIN_ENTRY_PATH), "exec"), namespace)
    return namespace


class FakeOptions:
    def __init__(self, data):
        self.data = data
        self.saved = 0

    def set(self, key, value):
        self.data[key] = value

    def save(self, _filename):
        self.saved += 1


class IllustriousCheckpointInspectionTests(unittest.TestCase):
    def test_backend_signature_identifies_sdxl_and_builtin_vae(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unknown-name.safetensors"
            write_checkpoint_header(checkpoint, cross_attention_dim=2048, include_vae=True)
            self.assertEqual(inspect_checkpoint_file(checkpoint), ("xl", True))

    def test_known_legacy_non_xl_signatures_are_not_sdxl(self):
        with tempfile.TemporaryDirectory() as directory:
            for dimension in (768, 1024, 1280):
                with self.subTest(dimension=dimension):
                    checkpoint = Path(directory) / f"model-{dimension}.safetensors"
                    write_checkpoint_header(checkpoint, cross_attention_dim=dimension)
                    self.assertEqual(inspect_checkpoint_file(checkpoint), ("non_xl", False))

    def test_standard_metadata_can_identify_sdxl(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "metadata-only.safetensors"
            write_checkpoint_header(
                checkpoint,
                metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
            )
            self.assertEqual(inspect_checkpoint_file(checkpoint), ("xl", False))

    def test_file_change_invalidates_cached_result(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "changing.safetensors"
            write_checkpoint_header(checkpoint, cross_attention_dim=2048, include_vae=True)
            self.assertEqual(inspect_checkpoint_file(checkpoint), ("xl", True))

            write_checkpoint_header(
                checkpoint,
                cross_attention_dim=768,
                metadata={"modelspec.implementation": "changed-size"},
            )
            self.assertEqual(inspect_checkpoint_file(checkpoint), ("non_xl", False))

    def test_local_novsw_is_sdxl_with_builtin_vae(self):
        checkpoint = REPO_ROOT / "models" / "Stable-diffusion" / "Novsw.safetensors"
        if not checkpoint.exists():
            self.skipTest("Novsw.safetensors is not installed")
        self.assertEqual(inspect_checkpoint_file(checkpoint), ("xl", True))


class IllustriousFrontendStateTests(unittest.TestCase):
    def load_checkpoint_detection_functions(self):
        namespace = {
            "os": os,
            "inspect_checkpoint_file": inspect_checkpoint_file,
            "XL_DEFAULT_CHECKPOINT": "miaomiaoHarem_v20.safetensors",
        }
        for name in (
            "is_lumina_checkpoint_name",
            "normalize_checkpoint_text",
            "resolve_checkpoint_file",
            "checkpoint_structure_family",
            "is_qwen_checkpoint_name",
            "is_krea_checkpoint_name",
            "is_anima_checkpoint_name",
            "infer_module_profile_from_checkpoint",
            "is_xl_checkpoint_name",
            "infer_forge_preset_from_checkpoint",
        ):
            load_function(name, namespace)
        return namespace

    def test_unnamed_sdxl_file_routes_to_xl_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "Novsw.safetensors"
            write_checkpoint_header(checkpoint, cross_attention_dim=2048, include_vae=True)
            functions = self.load_checkpoint_detection_functions()
            self.assertEqual(functions["infer_forge_preset_from_checkpoint"](checkpoint), "xl")

    def test_structure_rejects_a_misleading_xl_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "not-really-xl.safetensors"
            write_checkpoint_header(checkpoint, cross_attention_dim=768)
            functions = self.load_checkpoint_detection_functions()
            self.assertIsNone(functions["infer_forge_preset_from_checkpoint"](checkpoint))

    def test_sdxl_structure_wins_over_an_ambiguous_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "anima-style-illustrious.safetensors"
            write_checkpoint_header(checkpoint, cross_attention_dim=2048, include_vae=True)
            functions = self.load_checkpoint_detection_functions()
            self.assertEqual(functions["infer_forge_preset_from_checkpoint"](checkpoint), "xl")

    def test_xl_module_resolver_keeps_automatic_and_manual_modes_separate(self):
        options = FakeOptions({
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "shared": SimpleNamespace(opts=options),
            "normalize_forge_preset": lambda value: value,
            "infer_module_profile_from_checkpoint": lambda _value: None,
            "infer_forge_preset_from_checkpoint": lambda _value: "xl",
            "get_checkpoint_preferred_modules": lambda _value: None,
            "resolve_existing_modules": lambda values: list(values),
            "get_default_qwen_modules": lambda: [],
            "get_default_krea_modules": lambda: [],
            "get_default_anima_modules": lambda: [],
            "get_default_xl_modules": lambda _checkpoint: ["automatic-vae"],
            "is_lumina_checkpoint_name": lambda _value: False,
        }
        resolver = load_function("get_preset_additional_modules", namespace)
        self.assertEqual(resolver("xl", "Novsw.safetensors"), ["automatic-vae"])

        options.data["forge_additional_modules_xl"] = ["manual-vae"]
        options.data["forge_additional_modules_xl_configured"] = True
        self.assertEqual(resolver("xl", "Novsw.safetensors"), ["manual-vae"])

        options.data["forge_additional_modules_xl"] = []
        self.assertEqual(resolver("xl", "Novsw.safetensors"), [])

    def test_checkpoint_preferred_vae_overrides_global_xl_choice(self):
        options = FakeOptions({
            "forge_additional_modules_xl": ["global-vae"],
            "forge_additional_modules_xl_configured": True,
        })
        namespace = {
            "shared": SimpleNamespace(opts=options),
            "normalize_forge_preset": lambda value: value,
            "infer_module_profile_from_checkpoint": lambda _value: None,
            "infer_forge_preset_from_checkpoint": lambda _value: "xl",
            "get_checkpoint_preferred_modules": lambda _value: ["model-vae"],
            "resolve_existing_modules": lambda values: list(values),
            "get_default_qwen_modules": lambda: [],
            "get_default_krea_modules": lambda: [],
            "get_default_anima_modules": lambda: [],
            "get_default_xl_modules": lambda _checkpoint: [],
            "is_lumina_checkpoint_name": lambda _value: False,
        }
        resolver = load_function("get_preset_additional_modules", namespace)
        self.assertEqual(resolver("xl", "custom.safetensors"), ["model-vae"])

    def test_real_novsw_uses_builtin_vae_in_production_resolver(self):
        checkpoint = REPO_ROOT / "models" / "Stable-diffusion" / "Novsw.safetensors"
        if not checkpoint.exists():
            self.skipTest("Novsw.safetensors is not installed")

        options = FakeOptions({
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "os": os,
            "inspect_checkpoint_file": inspect_checkpoint_file,
            "shared": SimpleNamespace(opts=options),
            "module_list": {},
            "XL_DEFAULT_MODULES": [],
            "XL_FALLBACK_MODULES": [],
            "XL_DEFAULT_CHECKPOINT": "miaomiaoHarem_v20.safetensors",
        }
        load_functions(
            {
                "get_default_qwen_modules",
                "get_default_krea_modules",
                "get_default_xl_modules",
                "get_default_anima_modules",
                "resolve_existing_modules",
                "get_checkpoint_preferred_modules",
                "get_preset_additional_modules",
                "normalize_forge_preset",
                "is_lumina_checkpoint_name",
                "normalize_checkpoint_text",
                "resolve_checkpoint_file",
                "checkpoint_structure_family",
                "is_qwen_checkpoint_name",
                "is_krea_checkpoint_name",
                "is_anima_checkpoint_name",
                "infer_module_profile_from_checkpoint",
                "infer_forge_preset_from_checkpoint",
                "is_xl_checkpoint_name",
            },
            namespace,
        )
        self.assertEqual(namespace["get_preset_additional_modules"]("xl", checkpoint), [])

    def test_selecting_builtin_vae_is_saved_even_when_global_list_is_empty(self):
        options = FakeOptions({
            "forge_additional_modules": [],
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "os": os,
            "module_list": {},
            "shared": SimpleNamespace(opts=options, config_filename="config.json"),
            "refresh_model_loading_parameters": lambda: None,
        }
        change_modules = load_function("modules_change", namespace)
        self.assertTrue(change_modules([], preset="xl", save=False, refresh=False))
        self.assertTrue(options.data["forge_additional_modules_xl_configured"])
        self.assertEqual(options.data["forge_additional_modules_xl"], [])

    def test_automatic_xl_update_does_not_become_a_manual_override(self):
        options = FakeOptions({
            "forge_additional_modules": [],
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "os": os,
            "module_list": {},
            "shared": SimpleNamespace(opts=options, config_filename="config.json"),
            "refresh_model_loading_parameters": lambda: None,
        }
        change_modules = load_function("modules_change", namespace)
        self.assertFalse(change_modules([], preset="xl", save=False, refresh=False, persist_profile=False))
        self.assertFalse(options.data["forge_additional_modules_xl_configured"])

    def test_legacy_xl_modules_are_migrated_once(self):
        options = FakeOptions({
            "forge_additional_modules": ["legacy-vae"],
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "shared": SimpleNamespace(opts=options),
            "infer_forge_preset_from_checkpoint": lambda _value: "xl",
            "resolve_existing_modules": lambda values: list(values),
        }
        migrate = load_function("migrate_xl_module_state", namespace)
        self.assertTrue(migrate("Novsw.safetensors"))
        self.assertEqual(options.data["forge_additional_modules_xl"], ["legacy-vae"])
        self.assertTrue(options.data["forge_additional_modules_xl_configured"])
        self.assertFalse(migrate("Novsw.safetensors"))

    def test_non_xl_legacy_modules_are_not_migrated(self):
        options = FakeOptions({
            "forge_additional_modules": ["qwen-vae"],
            "forge_additional_modules_xl": [],
            "forge_additional_modules_xl_configured": False,
        })
        namespace = {
            "shared": SimpleNamespace(opts=options),
            "infer_forge_preset_from_checkpoint": lambda _value: "qwen",
            "resolve_existing_modules": lambda values: list(values),
        }
        migrate = load_function("migrate_xl_module_state", namespace)
        self.assertFalse(migrate("qwen.safetensors"))
        self.assertFalse(options.data["forge_additional_modules_xl_configured"])

    def test_xl_branch_restores_saved_checkpoint_and_keeps_dimensions(self):
        handler = get_function("on_preset_change")
        xl_branch = next(
            node
            for node in handler.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "preset"
            and any(isinstance(item, ast.Constant) and item.value == "xl" for item in node.test.comparators)
        )
        branch_source = ast.unparse(xl_branch)
        self.assertIn("forge_checkpoint_xl", branch_source)
        self.assertIn("get_preset_additional_modules('xl', xl_checkpoint)", branch_source)
        self.assertIn("checkpoint_change(xl_checkpoint, preset='xl'", branch_source)
        self.assertIn("persist_profile=False", branch_source)

    def test_xl_state_options_are_registered(self):
        source = SHARED_OPTIONS_PATH.read_text(encoding="utf-8")
        self.assertIn('"forge_checkpoint_xl"', source)
        self.assertIn('"forge_additional_modules_xl"', source)
        self.assertIn('"forge_additional_modules_xl_configured"', source)

    def test_xl_detection_uses_checkpoint_structure(self):
        function_source = ast.unparse(get_function("is_xl_checkpoint_name"))
        self.assertIn("checkpoint_structure_family(value)", function_source)
        self.assertIn("return structure_family == 'xl'", function_source)

    def test_only_user_input_saves_manual_xl_vae_state(self):
        source = MAIN_ENTRY_PATH.read_text(encoding="utf-8")
        self.assertIn("ui_vae.input(modules_change", source)
        self.assertNotIn("ui_vae.change(modules_change", source)

    def test_api_checkpoint_change_also_sets_the_matching_preset(self):
        source = SYSINFO_PATH.read_text(encoding="utf-8")
        self.assertIn("target_preset = main_entry.infer_forge_preset_from_checkpoint(v)", source)
        self.assertIn("shared.opts.set('forge_preset', target_preset", source)
        self.assertIn("run_callbacks=False", source)
        self.assertIn("checkpoint_change(v, preset=target_preset", source)


if __name__ == "__main__":
    unittest.main()
