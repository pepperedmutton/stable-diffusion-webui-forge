import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ENTRY_PATH = REPO_ROOT / "modules_forge" / "main_entry.py"


def load_xl_memory_policy(total_vram, saved_model_memory=None):
    tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "XL_INFERENCE_MEMORY_MB"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "get_xl_model_memory")
    ]
    opts = SimpleNamespace()
    if saved_model_memory is not None:
        opts.xl_GPU_MB = saved_model_memory
    namespace = {"shared": SimpleNamespace(opts=opts), "total_vram": total_vram}
    exec(
        compile(ast.Module(body=selected_nodes, type_ignores=[]), str(MAIN_ENTRY_PATH), "exec"),
        namespace,
    )
    return namespace


class IllustriousMemoryTests(unittest.TestCase):
    def test_xl_preset_uses_the_memory_policy(self):
        tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
        preset_handler = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "on_preset_change"
        )
        xl_branch = next(
            node
            for node in ast.walk(preset_handler)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "preset"
            and any(isinstance(item, ast.Constant) and item.value == "xl" for item in node.test.comparators)
        )
        model_memory_assignment = next(
            node
            for node in xl_branch.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "model_mem" for target in node.targets)
        )

        self.assertIsInstance(model_memory_assignment.value, ast.Call)
        self.assertEqual(model_memory_assignment.value.func.id, "get_xl_model_memory")

    def test_xl_preset_reserves_4096_mb_by_default(self):
        namespace = load_xl_memory_policy(total_vram=24576)

        self.assertEqual(namespace["get_xl_model_memory"](), 24576 - 4096)

    def test_xl_preset_rejects_old_1024_mb_reserve(self):
        namespace = load_xl_memory_policy(
            total_vram=24576,
            saved_model_memory=24576 - 1024,
        )

        self.assertEqual(namespace["get_xl_model_memory"](), 24576 - 4096)

    def test_krea_keeps_4096_mb_reserve(self):
        source = MAIN_ENTRY_PATH.read_text(encoding="utf-8")

        self.assertIn('opt("krea_GPU_MB", total_vram - 4096)', source)
        self.assertIn('shared.OptionInfo(total_vram - 4096, "GPU Weights (MB)"', source)

    def test_cache_defaults_limit_resident_checkpoints(self):
        # A clean checkout has no user config.json. Verify the registered
        # defaults without importing Forge or initializing its model backend.
        path = REPO_ROOT / "modules" / "shared_options.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defaults = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or key.value not in {
                    "sd_checkpoints_limit", "sd_checkpoints_keep_in_cpu"
                }:
                    continue
                # OptionInfo(...).info(...) retains the constructor default.
                while isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                    value = value.func.value
                self.assertIsInstance(value, ast.Call)
                self.assertIsInstance(value.func, ast.Name)
                self.assertEqual(value.func.id, "OptionInfo")
                defaults[key.value] = ast.literal_eval(value.args[0])

        self.assertEqual(defaults["sd_checkpoints_limit"], 1)
        self.assertIs(defaults["sd_checkpoints_keep_in_cpu"], True)


if __name__ == "__main__":
    unittest.main()
