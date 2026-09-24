"""Legacy Qwen PNG parsing and paste overrides without loading Forge or Torch."""

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "modules" / "infotext_utils.py"


def load_parser(current_checkpoint="Qwen-Image-2.1-NF4"):
    options = SimpleNamespace(
        sd_model_checkpoint=current_checkpoint,
        infotext_skip_pasting=[], infotext_styles="Ignore",
        use_old_hires_fix_width_height=False, forge_additional_modules=[],
        disable_weights_auto_swap=False,
        data_labels={"sd_model_checkpoint": SimpleNamespace(infotext="Model")},
        cast_value=lambda _key, value: value,
    )
    namespace = {
        "json": json, "os": os, "re": re,
        "shared": SimpleNamespace(opts=options),
        "main_entry": SimpleNamespace(module_list={}),
        "prompt_parser": SimpleNamespace(parse_prompt_attention=lambda text: [[text, 1.0]]),
        "infotext_versions": SimpleNamespace(backcompat=lambda _params: None),
    }
    functions = {"unquote", "restore_old_hires_fix_params", "parse_generation_parameters", "get_override_settings"}
    assignments = {"re_param_code", "re_param", "re_imagesize", "infotext_to_setting_name_mapping"}
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in functions
             or isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id in assignments]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, options


def infotext(model="Qwen-Image-2.1", runtime="Diffusers BF16 / CPU offload"):
    text = f"A paper crane\nSteps: 40, Sampler: Euler, Seed: 42, Model: {model}"
    return text + (f", Qwen runtime: {runtime}" if runtime is not None else "")


class LegacyQwenInfotextTests(unittest.TestCase):
    def test_legacy_bf16_png_overrides_each_other_active_precision(self):
        for current in ("Qwen-Image-2.1-NF4", "Qwen-Image-2.1-INT8", "ordinary-xl"):
            with self.subTest(current=current):
                ns, _ = load_parser(current)
                parsed = ns["parse_generation_parameters"](infotext())
                self.assertEqual(parsed["Model"], "Qwen-Image-2.1-BF16")
                self.assertEqual(parsed["Prompt"], "A paper crane")
                self.assertEqual(parsed["Seed"], "42")
                self.assertEqual(ns["get_override_settings"](parsed), [
                    ("Model", "sd_model_checkpoint", "Qwen-Image-2.1-BF16")
                ])

    def test_matching_bf16_checkpoint_needs_no_paste_override(self):
        ns, _ = load_parser("Qwen-Image-2.1-BF16")
        parsed = ns["parse_generation_parameters"](infotext())
        self.assertNotIn("Model", parsed)
        self.assertEqual(ns["get_override_settings"](parsed), [])

    def test_other_models_and_quantized_runtimes_are_not_reclassified(self):
        cases = [
            ("Qwen-Image", "Diffusers BF16 / CPU offload"),
            ("Qwen-Image-1.0", "Diffusers BF16 / CPU offload"),
            ("Qwen-Image-2.1-NF4", "Diffusers BF16 / CPU offload"),
            ("Qwen-Image-2.1-INT8", "DiT + TE INT8 / VAE BF16 / CPU offload"),
            ("Qwen-Image-2.1", "DiT + TE NF4 (double quantization) / VAE BF16 / CPU offload"),
            ("Qwen-Image-2.1", None),
        ]
        ns, _ = load_parser("ordinary-xl")
        for model, runtime in cases:
            with self.subTest(model=model, runtime=runtime):
                parsed = ns["parse_generation_parameters"](infotext(model, runtime))
                self.assertEqual(parsed["Model"], model)

    def test_user_skip_model_preference_is_preserved(self):
        ns, _ = load_parser()
        parsed = ns["parse_generation_parameters"](infotext(), skip_fields=["Model"])
        self.assertNotIn("Model", parsed)
        self.assertEqual(ns["get_override_settings"](parsed), [])

    def test_disabling_automatic_model_swap_is_preserved(self):
        ns, options = load_parser()
        options.disable_weights_auto_swap = True
        parsed = ns["parse_generation_parameters"](infotext())
        self.assertEqual(parsed["Model"], "Qwen-Image-2.1-BF16")
        self.assertEqual(ns["get_override_settings"](parsed), [])


if __name__ == "__main__":
    unittest.main()
