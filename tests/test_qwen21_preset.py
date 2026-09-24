"""Preset migration and switching tests without loading diffusion weights."""
import ast
import os
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules_forge" / "main_entry.py"


class Options:
    def __init__(self, data):
        self.data = dict(data)
        self.saves = 0

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def set(self, key, value):
        self.data[key] = value

    def save(self, _filename):
        self.saves += 1


def load_functions(options=None):
    options = options or Options({})
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "migrate_qwen21_options", "qwen21_dimension", "qwen21_ui_state_updates",
        "build_preset_ui_updates", "on_preset_change", "normalize_forge_preset",
        "get_preset_additional_modules", "get_default_qwen_modules",
        "on_checkpoint_sampler_ui_sync",
        "qwen21_clean_prompt",
        "forge_main_entry",
        "on_preset_page_load",
        "qwen21_precision_from_checkpoint", "get_qwen21_precision", "remember_qwen21_checkpoint",
        "sync_qwen21_precision_ui", "on_qwen21_precision_change", "checkpoint_change",
    }
    namespace = {
        "os": os,
        "re": re,
        "gr": SimpleNamespace(update=lambda **kwargs: dict(kwargs)),
        "shared": SimpleNamespace(opts=options, config_filename="unused.json"),
        "total_vram": 24576,
        "cmd_opts": SimpleNamespace(ui_config_file="unused.json"),
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith(("QWEN", "KREA_DEFAULT_", "ANIMA_DEFAULT_", "XL_DEFAULT_", "PRESET_")):
                try:
                    namespace[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    namespace["PRESET_OUTPUT_COUNT"] = 31
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class Qwen21MigrationTests(unittest.TestCase):
    def test_old_qwen_selection_and_modules_migrate_together(self):
        opts = Options({"sd_model_checkpoint": "qwen_image_fp8_e4m3fn.safetensors",
                        "forge_additional_modules": ["old-te", "old-vae"],
                        "forge_additional_modules_qwen": ["old-te", "old-vae"],
                        "qwen_t2i_steps": 8, "qwen_t2i_sampler": "LCM"})
        ns = load_functions(opts)
        self.assertTrue(ns["migrate_qwen21_options"]())
        self.assertEqual(opts.sd_model_checkpoint, "Qwen-Image-2.1-NF4")
        self.assertEqual(opts.forge_checkpoint_qwen, "Qwen-Image-2.1-NF4")
        self.assertEqual(opts.forge_qwen21_precision, "nf4")
        self.assertEqual(opts.forge_qwen21_profile_version, 2)
        self.assertEqual(opts.forge_additional_modules, [])
        self.assertEqual(opts.forge_additional_modules_qwen, [])
        self.assertFalse(ns["migrate_qwen21_options"]())
        self.assertEqual(opts.saves, 1)

    def test_migration_preserves_active_other_model_and_its_modules(self):
        opts = Options({"sd_model_checkpoint": "my-xl.safetensors",
                        "forge_preset": "xl", "forge_additional_modules": ["custom-vae"],
                        "forge_checkpoint_krea": "custom-krea.safetensors"})
        load_functions(opts)["migrate_qwen21_options"]()
        self.assertEqual(opts.sd_model_checkpoint, "my-xl.safetensors")
        self.assertEqual(opts.forge_additional_modules, ["custom-vae"])
        self.assertEqual(opts.forge_checkpoint_krea, "custom-krea.safetensors")
        self.assertEqual(opts.forge_preset, "xl")

    def test_explicit_precision_survives_migration_and_repeated_page_loads(self):
        for precision in ("nf4", "int8", "bf16"):
            with self.subTest(precision=precision):
                checkpoint = f"Qwen-Image-2.1-{precision.upper()}"
                opts = Options({"sd_model_checkpoint": checkpoint, "forge_preset": "qwen",
                                "forge_qwen21_profile_version": 1,
                                "forge_qwen21_precision": "nf4", "qwen21_t2i_steps": 32})
                ns = load_functions(opts)
                ns["migrate_qwen21_options"]()
                self.assertEqual(opts.sd_model_checkpoint, checkpoint)
                self.assertEqual(opts.forge_checkpoint_qwen, checkpoint)
                self.assertEqual(opts.forge_qwen21_precision, precision)
                self.assertEqual(opts.qwen21_t2i_steps, 32)
                self.assertFalse(ns["migrate_qwen21_options"]())

    def test_other_active_family_retains_saved_qwen_precision(self):
        opts = Options({"sd_model_checkpoint": "custom-anima.safetensors", "forge_preset": "anima",
                        "forge_checkpoint_qwen": "Qwen-Image-2.1-INT8",
                        "forge_additional_modules": ["anima-text-encoder"]})
        load_functions(opts)["migrate_qwen21_options"]()
        self.assertEqual(opts.forge_qwen21_precision, "int8")
        self.assertEqual(opts.forge_checkpoint_qwen, "Qwen-Image-2.1-INT8")
        self.assertEqual(opts.sd_model_checkpoint, "custom-anima.safetensors")
        self.assertEqual(opts.forge_additional_modules, ["anima-text-encoder"])

    def test_qwen_modules_never_reuse_a_stale_saved_encoder(self):
        ns = load_functions(Options({"forge_additional_modules_qwen": ["wrong-encoder"]}))
        ns["infer_module_profile_from_checkpoint"] = lambda value: "qwen"
        self.assertEqual(ns["get_preset_additional_modules"]("qwen", "Qwen-Image-2.1"), [])

    def test_dimensions_preserved_or_aligned(self):
        align = load_functions()["qwen21_dimension"]
        for value in (896, 1024, 1152, 1344):
            self.assertEqual(align(value), value)
        self.assertEqual(align(1000) % 32, 0)
        self.assertEqual(align(1328), 1312)
        self.assertEqual(align(64), 256)
        self.assertEqual(align(None), 1024)


class Qwen21SwitchTests(unittest.TestCase):
    def test_token_refresh_is_chained_after_each_backend_switch_event(self):
        chained = []
        bound_events = []

        class Event:
            def __init__(self, backend):
                self.backend = backend

            def then(self, **kwargs):
                chained.append((self.backend, kwargs))
                return self

        class Component:
            def input(self, fn=None, **kwargs):
                bound_events.append((fn, kwargs))
                return Event(fn)

            change = input
            load = input

        ns = load_functions()
        ns["gr"].State = lambda **kwargs: Component()
        ns["get_a1111_ui_component"] = lambda *args: Component()
        ns["Context"] = SimpleNamespace(root_block=Component())
        ns["refresh_model_loading_parameters"] = lambda: None
        for name in ("ui_checkpoint", "ui_vae", "ui_clip_skip", "ui_forge_preset",
                     "ui_forge_unet_storage_dtype_options", "ui_forge_async_loading",
                     "ui_forge_pin_shared_memory", "ui_forge_inference_memory", "ui_qwen21_precision"):
            ns[name] = Component()
        ns["forge_main_entry"]()
        token_refreshes = [(backend, kwargs) for backend, kwargs in chained if "js" in kwargs]
        precision_refreshes = [(backend, kwargs) for backend, kwargs in chained if "js" not in kwargs]
        expected_handlers = ["on_preset_change", "on_qwen21_precision_change", "on_checkpoint_sampler_ui_sync", "on_preset_page_load"]
        self.assertEqual([backend.__name__ for backend, _ in token_refreshes], expected_handlers)
        self.assertEqual([backend.__name__ for backend, _ in precision_refreshes], expected_handlers)
        self.assertTrue(all(kwargs["js"] == "refreshForgePresetTokenCounters" for _, kwargs in token_refreshes))
        self.assertTrue(all(kwargs["fn"] is ns["sync_qwen21_precision_ui"] for _, kwargs in precision_refreshes))
        self.assertTrue(all(kwargs["outputs"] == [ns["ui_qwen21_precision"]] for _, kwargs in precision_refreshes))
        page_load = next(kwargs for callback, kwargs in bound_events if callback is ns["on_preset_page_load"])
        self.assertNotIn(ns["ui_forge_preset"], page_load["inputs"])
        self.assertIs(page_load["outputs"][0], ns["ui_forge_preset"])
        self.assertEqual(len(page_load["outputs"]), 32)

    def make_runtime(self):
        opts = Options({"forge_preset": "xl", "sd_model_checkpoint": "custom-xl.safetensors",
                        "forge_additional_modules": ["custom-xl-vae"],
                        "forge_checkpoint_xl": "custom-xl.safetensors",
                        "forge_checkpoint_anima": "custom-anima.safetensors",
                        "forge_checkpoint_krea": "custom-krea.safetensors",
                        "qwen_t2i_steps": 8, "qwen_i2i_steps": 8,
                        "qwen_t2i_sampler": "LCM", "qwen_t2i_scheduler": "Normal"})
        ns = load_functions(opts)
        fake_modules = ModuleType("modules")
        fake_modules.ui_loadsave = SimpleNamespace(UiLoadsave=lambda path: SimpleNamespace(ui_settings={}))
        fake_modules.sd_models = SimpleNamespace(get_closet_checkpoint_match=lambda name: SimpleNamespace(name=name) if name else None)
        ns.update({
            "activate_preset_storage_dtype": lambda *args: ("Automatic", False),
            "sync_anima_live_preview_options": lambda *args: None,
            "refresh_model_loading_parameters": lambda: None,
            "get_preset_additional_modules": lambda preset, name="": [] if preset == "qwen" else [preset + "-module"],
            "checkpoint_change": lambda name, **kwargs: opts.set("sd_model_checkpoint", name),
            "modules_change": lambda values, **kwargs: opts.set("forge_additional_modules", values),
            "is_krea_checkpoint_name": lambda name: "krea" in name.lower(),
            "is_xl_checkpoint_name": lambda name: "xl" in name.lower(),
            "sync_xl_sampler_defaults": lambda **kwargs: None,
            "get_xl_model_memory": lambda: 20480,
        })
        return ns, opts, fake_modules

    def test_selecting_qwen_replaces_legacy_parameters_and_hides_manual_runtime_controls(self):
        ns, opts, fake_modules = self.make_runtime()
        with patch.dict(sys.modules, {"modules": fake_modules}):
            updates = ns["on_preset_change"]("qwen", 896, 1024, 1152, 1024,
                                             "negative", "edit negative", True, True, True)
        self.assertEqual(len(updates), 31)
        self.assertEqual(opts.sd_model_checkpoint, "Qwen-Image-2.1-NF4")
        self.assertEqual(opts.forge_additional_modules, [])
        self.assertEqual([updates[i]["value"] for i in (7, 8)], [40, 40])
        self.assertEqual([updates[i]["value"] for i in (13, 14)], [1.0, 1.0])
        self.assertEqual([updates[i]["value"] for i in (17, 18, 19, 20)], ["Euler", "Euler", "Simple", "Simple"])
        for index in range(1, 7):
            self.assertFalse(updates[index]["visible"])
        for index in (13, 14, 17, 18, 19, 20):
            self.assertFalse(updates[index]["interactive"])
        self.assertEqual(updates[28]["t2i_negative"], "negative")
        self.assertEqual([updates[i]["value"] for i in range(23, 28)], ["", "", False, False, False])

    def test_manual_precision_survives_leaving_returning_and_browser_refresh(self):
        for precision in ("int8", "bf16", "nf4"):
            with self.subTest(precision=precision):
                ns, opts, fake_modules = self.make_runtime()
                checkpoint = f"Qwen-Image-2.1-{precision.upper()}"
                with patch.dict(sys.modules, {"modules": fake_modules}):
                    changed = ns["on_qwen21_precision_change"](precision, 896, 1024, 1152, 1024)
                    self.assertEqual(changed[0]["value"], checkpoint)
                    self.assertEqual(opts.forge_qwen21_precision, precision)
                    self.assertEqual(ns["sync_qwen21_precision_ui"](), {"value": precision, "visible": True})
                    ns["on_preset_change"]("xl")
                    self.assertEqual(ns["sync_qwen21_precision_ui"](), {"value": precision, "visible": False})
                    returned = ns["on_preset_change"]("qwen", 896, 1024, 1152, 1024)
                    self.assertEqual(returned[0]["value"], checkpoint)
                    reloaded = ns["on_preset_page_load"](896, 1024, 1152, 1024)
                    self.assertEqual(reloaded[1]["value"], checkpoint)
                    self.assertEqual([returned[i]["value"] for i in (9, 10, 11, 12)], [896, 1024, 1152, 1024])
                    self.assertEqual(opts.sd_model_checkpoint, checkpoint)

    def test_unavailable_precision_does_not_change_saved_selection(self):
        ns, opts, fake_modules = self.make_runtime()
        ns["gr"].Error = ValueError
        before = dict(opts.data)
        fake_modules.sd_models.get_closet_checkpoint_match = lambda name: None
        with patch.dict(sys.modules, {"modules": fake_modules}):
            with self.assertRaisesRegex(ValueError, "not installed"):
                ns["on_qwen21_precision_change"]("int8")
        self.assertEqual(opts.data, before)

    def test_checkpoint_selection_updates_precision_even_if_checkpoint_is_unchanged(self):
        for unchanged in (False, True):
            with self.subTest(unchanged=unchanged):
                target = SimpleNamespace(name="Qwen-Image-2.1-INT8")
                old = target if unchanged else SimpleNamespace(name="Qwen-Image-2.1-NF4")
                opts = Options({"sd_model_checkpoint": old.name, "forge_qwen21_precision": "nf4"})
                ns = load_functions(opts)
                ns.update({"is_xl_checkpoint_name": lambda name: False,
                           "infer_module_profile_from_checkpoint": lambda name: "qwen",
                           "infer_forge_preset_from_checkpoint": lambda name: "qwen",
                           "refresh_model_loading_parameters": lambda: None})
                fake_modules = ModuleType("modules")
                fake_modules.sd_models = SimpleNamespace(get_closet_checkpoint_match=lambda name: target if name == target.name else old)
                with patch.dict(sys.modules, {"modules": fake_modules}):
                    ns["checkpoint_change"](target.name, preset="qwen")
                self.assertEqual(opts.forge_qwen21_precision, "int8")
                self.assertEqual(opts.forge_checkpoint_qwen, target.name)
                self.assertEqual(opts.sd_model_checkpoint, target.name)
                self.assertEqual(opts.saves, 1)

    def test_switching_back_restores_controls_and_selects_other_family_modules(self):
        for family in ("xl", "anima", "krea"):
            with self.subTest(family=family):
                ns, opts, fake_modules = self.make_runtime()
                with patch.dict(sys.modules, {"modules": fake_modules}):
                    entered = ns["on_preset_change"]("qwen", 896, 1024, 1152, 1024,
                                                     "negative", "edit negative", True, True, True)
                    repeated = ns["on_preset_change"]("qwen", 896, 1024, 1152, 1024,
                                                      "", "", False, False, False, entered[28])
                    self.assertEqual(repeated[28], entered[28])
                    restored = ns["on_preset_change"](family, 896, 1024, 1152, 1024,
                                                      "", "", False, False, False, entered[28])
                self.assertEqual(opts.forge_preset, family)
                self.assertEqual(opts.forge_additional_modules, [family + "-module"])
                self.assertIn(family, opts.sd_model_checkpoint)
                self.assertEqual([restored[i]["value"] for i in (9, 10, 11, 12)], [896, 1024, 1152, 1024])
                for index in (13, 14, 17, 18, 19, 20):
                    self.assertTrue(restored[index]["interactive"])
                self.assertIsNone(restored[28])
                self.assertEqual([restored[i]["visible"] for i in range(23, 28)],
                                 [True, True, False, False, False])
                if family == "krea":
                    self.assertEqual([restored[i]["value"] for i in range(23, 28)], ["", "", False, False, False])
                else:
                    self.assertEqual([restored[i]["value"] for i in range(23, 28)], ["negative", "edit negative", True, True, True])

    def test_checkpoint_dropdown_switch_restores_same_browser_state(self):
        ns, opts, fake_modules = self.make_runtime()
        ns["infer_forge_preset_from_checkpoint"] = lambda name: "xl"
        with patch.dict(sys.modules, {"modules": fake_modules}):
            entered = ns["on_preset_change"]("qwen", 896, 1024, 1152, 1024,
                                             "negative", "edit negative", True, True, True)
            restored = ns["on_checkpoint_sampler_ui_sync"](
                "custom-xl.safetensors", "qwen", 896, 1024, 1152, 1024,
                "", "", False, False, False, entered[28],
            )
        self.assertEqual(restored[0], {"value": "xl"})
        self.assertEqual(restored[23]["value"], "negative")
        self.assertTrue(restored[25]["value"])
        self.assertIsNone(restored[28])
        self.assertEqual(opts.sd_model_checkpoint, "custom-xl.safetensors")

    def test_lora_tags_are_removed_and_restored_without_modifying_prose(self):
        ns, opts, fake_modules = self.make_runtime()
        t2i_prompt = "  a cat <lora:cat-face:0.8>, (red scarf:1.2)\n<hypernet:detail:1>"
        i2i_prompt = "Change the color. <LYCO:color:1> Keep <ordinary:text> unchanged."
        cleaned_t2i = "  a cat , (red scarf:1.2)\n"
        cleaned_i2i = "Change the color.  Keep <ordinary:text> unchanged."
        with patch.dict(sys.modules, {"modules": fake_modules}):
            entered = ns["on_preset_change"]("qwen", current_t2i_prompt=t2i_prompt,
                                             current_i2i_prompt=i2i_prompt)
            self.assertEqual(entered[29]["value"], cleaned_t2i)
            self.assertEqual(entered[30]["value"], cleaned_i2i)
            restored = ns["on_preset_change"]("xl", qwen_ui_backup=entered[28],
                                               current_t2i_prompt=cleaned_t2i,
                                               current_i2i_prompt=cleaned_i2i)
        self.assertEqual(restored[29]["value"], t2i_prompt)
        self.assertEqual(restored[30]["value"], i2i_prompt)

    def test_prompt_edits_made_in_qwen_are_kept_on_exit(self):
        ns, opts, fake_modules = self.make_runtime()
        with patch.dict(sys.modules, {"modules": fake_modules}):
            entered = ns["on_preset_change"]("qwen", current_t2i_prompt="cat <lora:cat:1>")
            restored = ns["on_preset_change"]("xl", qwen_ui_backup=entered[28],
                                               current_t2i_prompt="a new dog prompt")
        self.assertNotIn("value", restored[29])

    def test_qwen_xl_krea_anima_sequence_reenables_negative_prompts(self):
        ns, opts, fake_modules = self.make_runtime()
        backup = None
        visible_state = [{"value": "negative", "interactive": True},
                         {"value": "edit negative", "interactive": True},
                         {"value": True, "visible": False},
                         {"value": True, "visible": False},
                         {"value": True, "visible": False}]
        with patch.dict(sys.modules, {"modules": fake_modules}):
            for family, negative_enabled in (("qwen", False), ("xl", True),
                                             ("krea", False), ("anima", True)):
                with self.subTest(family=family):
                    result = ns["on_preset_change"](
                        family, 896, 1024, 1152, 1024,
                        *(item["value"] for item in visible_state), backup,
                    )
                    for current, update in zip(visible_state, result[23:28]):
                        current.update(update)
                    if result[28] != {}:
                        backup = result[28]
                    self.assertEqual([item["interactive"] for item in visible_state[:2]],
                                     [negative_enabled, negative_enabled])
                    self.assertEqual([item["visible"] for item in visible_state[2:]],
                                     [False, False, False])

    def test_missing_backup_still_resets_stale_disabled_and_hidden_controls(self):
        for family in ("xl", "anima", "krea"):
            with self.subTest(family=family):
                ns, opts, fake_modules = self.make_runtime()
                with patch.dict(sys.modules, {"modules": fake_modules}):
                    result = ns["on_preset_change"](family, qwen_ui_backup=None)
                self.assertEqual([update["visible"] for update in result[23:28]],
                                 [True, True, False, False, False])
                self.assertEqual([update["interactive"] for update in result[23:28]],
                                 [family != "krea", family != "krea", True, True, True])
                if family != "krea":
                    self.assertTrue(all("value" not in update for update in result[23:28]))

    def test_each_negative_field_tracks_its_own_target_cfg(self):
        ns, opts, fake_modules = self.make_runtime()
        opts.set("xl_t2i_cfg", 1.0)
        opts.set("xl_i2i_cfg", 6.0)
        with patch.dict(sys.modules, {"modules": fake_modules}):
            result = ns["on_preset_change"]("xl")
        self.assertFalse(result[23]["interactive"])
        self.assertTrue(result[24]["interactive"])

    def test_page_reload_uses_live_preset_and_updates_radio(self):
        ns, opts, fake_modules = self.make_runtime()
        # The UI was constructed with XL; a later selection was persisted as Qwen.
        opts.set("forge_preset", "qwen")
        opts.set("sd_model_checkpoint", "Qwen-Image-2.1")
        with patch.dict(sys.modules, {"modules": fake_modules}):
            result = ns["on_preset_page_load"](1024, 1024, 1024, 1024)
        self.assertEqual(len(result), 32)
        self.assertEqual(result[0], {"value": "qwen"})
        self.assertEqual(opts.forge_preset, "qwen")
        self.assertEqual(opts.sd_model_checkpoint, "Qwen-Image-2.1-NF4")
        self.assertEqual(result[1]["value"], "Qwen-Image-2.1-NF4")
        self.assertFalse(result[24]["visible"])


if __name__ == "__main__":
    unittest.main()
