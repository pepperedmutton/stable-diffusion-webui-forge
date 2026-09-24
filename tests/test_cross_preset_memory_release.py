import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_MODELS_PATH = REPO_ROOT / "modules" / "sd_models.py"
PROCESSING_PATH = REPO_ROOT / "modules" / "processing.py"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class CrossPresetMemoryReleaseTests(unittest.TestCase):
    def test_switch_cache_clear_drops_both_ram_caches(self):
        calls = []
        model_cache = {"old-model": object()}
        checkpoint_cache = {"old-checkpoint": object()}
        namespace = {
            "model_data": SimpleNamespace(forge_cpu_cache=model_cache),
            "checkpoints_loaded": checkpoint_cache,
            "gc": SimpleNamespace(collect=lambda: calls.append("collect")),
        }
        clear_caches = load_function(
            SD_MODELS_PATH,
            "clear_forge_model_caches_for_switch",
            namespace,
        )

        clear_caches()

        self.assertEqual(model_cache, {})
        self.assertEqual(checkpoint_cache, {})
        self.assertEqual(calls, ["collect"])

    def test_cross_preset_switch_uses_the_resident_model_family(self):
        calls = []
        old_model = SimpleNamespace(forge_preset="krea")
        new_model = SimpleNamespace(forge_preset="xl")
        task = SimpleNamespace(
            sd_model=old_model,
            clear_prompt_cache=lambda: calls.append("prompt-cache"),
        )

        def clear_caches():
            calls.append("model-caches")

        def reload_model(force_full_unload=False):
            calls.append(("reload", force_full_unload))
            return new_model, True

        namespace = {
            "StableDiffusionProcessing": object,
            "sd_models": SimpleNamespace(
                model_data=SimpleNamespace(get_sd_model=lambda: old_model),
                clear_forge_model_caches_for_switch=clear_caches,
            ),
            "opts": SimpleNamespace(forge_preset="xl"),
            "forge_model_reload": reload_model,
            "memory_management": SimpleNamespace(
                unload_all_models=lambda: calls.append("global-unload")
            ),
            "need_global_unload": True,
        }
        manage_cache = load_function(
            PROCESSING_PATH,
            "manage_model_and_prompt_cache",
            namespace,
        )

        manage_cache(task)

        self.assertIs(task.sd_model, new_model)
        self.assertEqual(
            calls,
            ["model-caches", ("reload", True), "prompt-cache"],
        )
        self.assertIs(namespace["need_global_unload"], False)

    def test_same_preset_keeps_normal_reload_path(self):
        calls = []
        old_model = SimpleNamespace(forge_preset="xl")
        new_model = SimpleNamespace(forge_preset="xl")
        task = SimpleNamespace(
            sd_model=None,
            clear_prompt_cache=lambda: calls.append("prompt-cache"),
        )

        namespace = {
            "StableDiffusionProcessing": object,
            "sd_models": SimpleNamespace(
                model_data=SimpleNamespace(get_sd_model=lambda: old_model),
                clear_forge_model_caches_for_switch=lambda: calls.append("model-caches"),
            ),
            "opts": SimpleNamespace(forge_preset="xl"),
            "forge_model_reload": lambda force_full_unload=False: (
                calls.append(("reload", force_full_unload)) or new_model,
                False,
            ),
            "memory_management": SimpleNamespace(
                unload_all_models=lambda: calls.append("global-unload")
            ),
            "need_global_unload": True,
        }
        manage_cache = load_function(
            PROCESSING_PATH,
            "manage_model_and_prompt_cache",
            namespace,
        )

        manage_cache(task)

        self.assertIs(task.sd_model, new_model)
        self.assertEqual(
            calls,
            [("reload", False), "global-unload", "prompt-cache"],
        )

    def test_forced_reload_does_not_keep_the_previous_model(self):
        source = SD_MODELS_PATH.read_text(encoding="utf-8")

        self.assertIn("def forge_model_reload(force_full_unload=False):", source)
        self.assertIn(
            "previous_sd_model = None if force_full_unload else model_data.sd_model",
            source,
        )
        self.assertIn("if not force_full_unload:\n            remember_forge_model_in_cpu_cache", source)
        self.assertIn("sd_model.forge_preset = str(getattr(opts, 'forge_preset', '')", source)
        self.assertIn('shared.opts.data["sd_checkpoint_hash"] = \'\'', source)


if __name__ == "__main__":
    unittest.main()
