import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ENTRY_PATH = REPO_ROOT / "modules_forge" / "main_entry.py"
DIMENSION_NAMES = (
    "current_t2i_width",
    "current_i2i_width",
    "current_t2i_height",
    "current_i2i_height",
)
DIMENSION_UPDATE_INDEXES = (9, 10, 11, 12)


def get_preset_handler():
    tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_preset_change"
    )


def get_function(name):
    tree = ast.parse(MAIN_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(MAIN_ENTRY_PATH))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def get_preset_branch(handler, preset):
    return next(
        node
        for node in handler.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "preset"
        and any(
            isinstance(item, ast.Constant) and item.value == preset
            for item in node.test.comparators
        )
    )


def get_update_list(branch):
    return_statement = next(node for node in branch.body if isinstance(node, ast.Return))
    return return_statement.value.args[1]


class PresetDimensionPreservationTests(unittest.TestCase):
    def test_each_model_preset_keeps_current_dimensions(self):
        handler = get_preset_handler()

        for preset in ("qwen", "anima", "krea", "xl", "flux"):
            with self.subTest(preset=preset):
                updates = get_update_list(get_preset_branch(handler, preset))
                for index, expected_name in zip(DIMENSION_UPDATE_INDEXES, DIMENSION_NAMES):
                    update = updates.elts[index]
                    value = next(
                        keyword.value
                        for keyword in update.keywords
                        if keyword.arg == "value"
                    )
                    self.assertIsInstance(value, ast.Name)
                    self.assertEqual(value.id, expected_name)

    def test_unknown_preset_keeps_current_dimensions(self):
        handler = get_preset_handler()
        updates = get_update_list(handler)

        for index, expected_name in zip(DIMENSION_UPDATE_INDEXES, DIMENSION_NAMES):
            update = updates.elts[index]
            value = next(
                keyword.value
                for keyword in update.keywords
                if keyword.arg == "value"
            )
            self.assertIsInstance(value, ast.Name)
            self.assertEqual(value.id, expected_name)

    def test_checkpoint_driven_preset_change_forwards_current_dimensions(self):
        handler = get_function("on_checkpoint_sampler_ui_sync")
        preset_call = next(
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "on_preset_change"
        )

        forwarded_names = [
            argument.id
            for argument in preset_call.args[1:]
            if isinstance(argument, ast.Name)
        ]
        self.assertEqual(forwarded_names, list(DIMENSION_NAMES))


if __name__ == "__main__":
    unittest.main()
