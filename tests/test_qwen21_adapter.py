"""Qwen bridge contracts without loading Forge, Torch, or model weights."""

import ast
from collections import deque
from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qwen21_adapter_under_test", ROOT / "modules_forge" / "qwen21.py")
adapter = importlib.util.module_from_spec(SPEC)
with mock.patch("atexit.register"):
    SPEC.loader.exec_module(adapter)


def load_mask_function():
    tree = ast.parse((ROOT / "modules" / "processing.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_binary_mask")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "processing-mask", "exec"), namespace)
    return namespace["create_binary_mask"]


def load_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ModelPrecisionTests(unittest.TestCase):
    def write_model(self, directory, precision, sharded=(False, False)):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
        quantization = {
            "quant_method": "bitsandbytes", "load_in_4bit": precision == "nf4",
            "load_in_8bit": precision == "int8", "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        for index, (component, prefix) in enumerate((("transformer", "diffusion_pytorch_model"),
                                                    ("text_encoder", "model"),
                                                    ("vae", "diffusion_pytorch_model"))):
            folder = directory / component
            folder.mkdir()
            config = {"quantization_config": quantization} if precision != "bf16" and component != "vae" else {}
            (folder / "config.json").write_text(json.dumps(config))
            if index < 2 and sharded[index]:
                files = [f"{prefix}-00001-of-00002.safetensors", f"{prefix}-00002-of-00002.safetensors"]
                (folder / f"{prefix}.safetensors.index.json").write_text(json.dumps({"weight_map": dict(zip(("weight1", "weight2"), files))}))
            else:
                files = [f"{prefix}.safetensors"]
            for filename in files:
                (folder / filename).write_bytes(b"test weights")

    def test_each_precision_accepts_single_files_and_all_shard_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            for precision in ("nf4", "int8", "bf16"):
                for sharded in ((False, False), (True, True), (False, True), (True, False)):
                    with self.subTest(precision=precision, sharded=sharded):
                        directory = Path(temporary) / f"{precision}-{sharded}"
                        self.write_model(directory, precision, sharded)
                        self.assertTrue(adapter.model_files_ready(directory, precision))
                        for wrong_precision in {"nf4", "int8", "bf16"} - {precision}:
                            self.assertFalse(adapter.model_files_ready(directory, wrong_precision))

    def test_missing_shard_and_empty_index_are_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "model"
            self.write_model(directory, "nf4", (True, True))
            (directory / "text_encoder" / "model-00002-of-00002.safetensors").unlink()
            self.assertFalse(adapter.model_files_ready(directory, "nf4"))
            (directory / "text_encoder" / "model.safetensors.index.json").write_text('{"weight_map":{}}')
            self.assertFalse(adapter.model_files_ready(directory, "nf4"))

    def test_nf4_requires_double_quantization_for_both_components(self):
        for component in ("transformer", "text_encoder"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "model"
                self.write_model(directory, "nf4")
                path = directory / component / "config.json"
                config = json.loads(path.read_text())
                config["quantization_config"]["bnb_4bit_use_double_quant"] = False
                path.write_text(json.dumps(config))
                self.assertFalse(adapter.model_files_ready(directory, "nf4"))

    def test_checkpoint_names_aliases_and_paths_have_distinct_precisions(self):
        aliases = {}
        registered = []

        class CheckpointInfo:
            def __init__(self, filename):
                self.filename = filename

            def register(self):
                registered.append(self)
                aliases.update(dict.fromkeys(self.ids, self))

        modules = ModuleType("modules")
        modules.sd_models = SimpleNamespace(CheckpointInfo=CheckpointInfo)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(adapter, "ROOT", Path(temporary)):
            for precision, profile in adapter.MODEL_PROFILES.items():
                self.write_model(Path(temporary) / "models" / "diffusers" / profile["directory"], precision)
            with mock.patch.dict(sys.modules, {"modules": modules}):
                adapter.register_checkpoint()
            self.assertEqual({info.qwen21_precision for info in registered}, {"nf4", "int8", "bf16"})
            self.assertEqual(aliases["Qwen-Image-2.1"].name, "Qwen-Image-2.1-NF4")
            self.assertEqual(aliases["Qwen Image 2.1"].qwen21_precision, "nf4")
            bf16 = aliases["Qwen-Image-2.1-BF16"]
            self.assertEqual(Path(bf16.model_dir).name, "Qwen-Image-2.1")
            self.assertIs(aliases[bf16.filename], bf16)
            for info in registered:
                self.assertEqual(info.metadata["precision"], info.qwen21_precision)

    def test_bf16_weights_cannot_be_registered_under_nf4_name(self):
        modules = ModuleType("modules")
        modules.sd_models = SimpleNamespace(CheckpointInfo=mock.Mock())
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(adapter, "ROOT", Path(temporary)):
            self.write_model(Path(temporary) / "models" / "diffusers" / "Qwen-Image-2.1-NF4", "bf16")
            with mock.patch.dict(sys.modules, {"modules": modules}):
                adapter.register_checkpoint()
            modules.sd_models.CheckpointInfo.assert_not_called()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.actions = deque()
        self.calls = []
        self.saved = []
        self.workers = []
        self.jobs_finished = 0
        self.state = SimpleNamespace(
            interrupted=False, skipped=False, stopping_generation=False,
            sampling_step=0, sampling_steps=0, id_live_preview=0,
            nextjob=self.next_job, assign_current_image=self.assign_image,
        )
        self.shared = SimpleNamespace(
            sd_model=SimpleNamespace(qwen21=True), state=self.state,
            opts=SimpleNamespace(outdir_txt2img_samples="txt2img", outdir_img2img_samples="img2img"),
        )
        self.processing = SimpleNamespace(
            need_global_unload=True, get_fixed_seed=lambda seed: 123 if seed == -1 else seed,
            create_infotext=lambda p, prompts, seeds, subseeds, index: f"{prompts[index]}\nSeed: {seeds[index]}, Model: {p.sd_model_name}",
            create_binary_mask=load_mask_function(),
            Processed=lambda p, images, **kw: SimpleNamespace(images=images, **kw),
        )
        modules = ModuleType("modules")
        modules.images = SimpleNamespace(save_image=lambda *args, **kw: self.saved.append((args, kw)))
        modules.processing = self.processing
        modules.shared = self.shared
        self.sd_models = SimpleNamespace(model_data=SimpleNamespace(forge_loading_parameters={"checkpoint_info": SimpleNamespace(qwen21=True)}))
        modules.sd_models = self.sd_models
        self.patches = [
            mock.patch.dict(sys.modules, {"modules": modules}),
            mock.patch.object(adapter, "WORK_DIR", self.work),
            mock.patch.object(adapter, "_worker", None),
            mock.patch.object(adapter, "Worker", self.make_worker),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def next_job(self):
        self.jobs_finished += 1

    def assign_image(self, image):
        self.state.current_image = image
        self.state.id_live_preview += 1

    def make_worker(self):
        owner = self

        class StubWorker:
            stopped = False

            @property
            def process(self):
                return SimpleNamespace(poll=lambda: 0 if self.stopped else None)

            def stop(self):
                self.stopped = True

            def generate(self, payload, state):
                captured = dict(payload)
                captured["references"] = []
                for path in payload["input_images"]:
                    with Image.open(path) as source:
                        captured["references"].append(source.copy())
                owner.calls.append(captured)
                action = owner.actions.popleft() if owner.actions else "result"
                if action in ("skip", "interrupt"):
                    state.skipped = action == "skip"
                    state.interrupted = action == "interrupt"
                    adapter.stop_worker()
                    return None
                if action == "error":
                    adapter.stop_worker()
                    raise RuntimeError("GPU allocation failed")
                Image.new("RGBA", (payload["width"], payload["height"]), (20, 40, 60, 128)).save(payload["output_path"])
                return {"event": "result", "output_path": payload["output_path"]}

        worker = StubWorker()
        self.workers.append(worker)
        return worker

    def task(self, **changes):
        p = SimpleNamespace(
            prompt="a blue cat", negative_prompt="stale negative", batch_size=1, n_iter=1,
            seed=42, subseed=55, subseed_strength=0, seed_resize_from_w=0, seed_resize_from_h=0,
            width=512, height=512, steps=8, extra_generation_params={},
            outpath_samples="samples", init_images=[], image_mask=None,
            inpainting_mask_invert=0, mask_round=True, mask_blur_x=0, mask_blur_y=0,
            enable_hr=True, restore_faces=True, tiling=True,
            clear_prompt_cache=lambda: None, fill_fields_from_opts=lambda: None,
            save_samples=lambda: True,
        )
        p.__dict__.update(changes)

        def setup_prompts():
            p.all_prompts = list(p.prompt) if isinstance(p.prompt, list) else [p.prompt] * (p.batch_size * p.n_iter)
            p.all_negative_prompts = [p.negative_prompt] * len(p.all_prompts)
        p.setup_prompts = setup_prompts
        return p

    def test_rgba_gallery_metadata_and_saved_png_use_the_actual_seed(self):
        result = adapter.process_images(self.task(batch_size=2))
        self.assertEqual(len(result.images), 2)
        self.assertEqual(result.all_seeds, [42, 43])
        self.assertEqual([item.mode for item in result.images], ["RGBA", "RGBA"])
        self.assertEqual(result.images[0].getpixel((0, 0)), (20, 40, 60, 128))
        self.assertIn("Seed: 43", result.images[1].info["parameters"])
        self.assertEqual([args[5] for args, _ in self.saved], ["png", "png"])
        self.assertTrue(all(kwargs.get("skip_stealth_pnginfo") for _, kwargs in self.saved))
        self.assertEqual(self.jobs_finished, 2)
        self.assertEqual(self.state.id_live_preview, 2)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_nonstandard_canvas_is_aligned_to_worker_patch_size(self):
        adapter.process_images(self.task(width=1040, height=1000))
        self.assertEqual(self.calls[0]["width"] % 32, 0)
        self.assertEqual(self.calls[0]["height"] % 32, 0)

    def test_selected_checkpoint_controls_worker_path_and_precision_metadata(self):
        for precision, profile in adapter.MODEL_PROFILES.items():
            with self.subTest(precision=precision):
                directory = self.work / profile["directory"]
                info = SimpleNamespace(qwen21=True, qwen21_precision=precision, model_dir=str(directory))
                self.sd_models.model_data.forge_loading_parameters["checkpoint_info"] = info
                self.shared.sd_model = SimpleNamespace(qwen21=True, qwen21_model_dir=str(directory))
                p = self.task()
                result = adapter.process_images(p)
                self.assertEqual(self.calls[-1]["model_dir"], str(directory))
                self.assertEqual(self.calls[-1]["precision"], precision)
                self.assertEqual(p.sd_model_name, profile["name"])
                self.assertEqual(p.extra_generation_params["Qwen precision"], precision.upper())
                self.assertEqual(p.extra_generation_params["Qwen runtime"], profile["runtime"])
                self.assertIn(profile["name"], result.info)

    def test_changing_qwen_precision_replaces_the_loaded_proxy(self):
        directory = self.work / "Qwen-Image-2.1-INT8"
        info = SimpleNamespace(qwen21=True, qwen21_precision="int8", model_dir=str(directory))
        self.sd_models.model_data.forge_loading_parameters["checkpoint_info"] = info
        self.shared.sd_model = SimpleNamespace(qwen21=True, qwen21_model_dir="old-nf4-directory")
        with mock.patch.object(adapter, "load_forge_model") as load:
            adapter.process_images(self.task())
        load.assert_called_once_with(info)

    def test_stale_diffusion_features_are_reset_before_generation(self):
        p = self.task(subseed_strength=0.8, seed_resize_from_w=512, seed_resize_from_h=512)
        adapter.process_images(p)
        self.assertFalse(p.enable_hr)
        self.assertFalse(p.restore_faces)
        self.assertFalse(p.tiling)
        self.assertEqual((p.sampler_name, p.scheduler, p.cfg_scale), ("Euler", "Simple", 1.0))
        self.assertEqual(p.negative_prompt, "")
        self.assertEqual((p.subseed_strength, p.seed_resize_from_w, p.seed_resize_from_h), (0, 0, 0))

    def test_old_lora_is_removed_and_adjustment_recorded(self):
        p = self.task(prompt="cat <lora:old:0.8>")
        result = adapter.process_images(p)
        self.assertEqual(self.calls[0]["prompt"], "cat")
        self.assertEqual(result.all_prompts, ["cat"])
        self.assertIn("Qwen prompt adjustment", p.extra_generation_params)

    def test_error_cleans_job_and_next_generation_can_restart(self):
        self.actions.append("error")
        with self.assertRaisesRegex(RuntimeError, "allocation"):
            adapter.process_images(self.task())
        self.assertIsNone(adapter._worker)
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertEqual(len(adapter.process_images(self.task()).images), 1)
        self.assertEqual(len(self.workers), 2)

    def test_interrupt_retains_completed_images_and_cleans_job(self):
        self.actions.extend(("result", "interrupt"))
        result = adapter.process_images(self.task(batch_size=3))
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.all_seeds, [42])
        self.assertIsNone(adapter._worker)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_skip_continues_with_next_requested_image_and_its_seed(self):
        self.actions.extend(("skip", "result"))
        result = adapter.process_images(self.task(prompt=["first", "second"]))
        self.assertEqual(len(result.images), 1)
        self.assertEqual([call["seed"] for call in self.calls], [42, 43])
        self.assertEqual(result.all_seeds, [43])
        self.assertIn("second", result.infotexts[0])
        self.assertEqual(len(self.workers), 2)

    def test_gradio_rgba_mask_uses_alpha_channel(self):
        source = Image.new("RGB", (512, 512), "red")
        mask = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        mask.paste((0, 0, 0, 255), (256, 0, 512, 512))
        adapter.process_images(self.task(init_images=[source], image_mask=mask))
        captured_mask = self.calls[0]["references"][1]
        self.assertEqual(captured_mask.convert("L").getpixel((0, 0)), 0)
        self.assertEqual(captured_mask.convert("L").getpixel((511, 0)), 255)

    def test_inpaint_keeps_pixels_outside_mask(self):
        source = Image.new("RGBA", (512, 512), (200, 100, 50, 255))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (256, 0, 512, 512))
        result = adapter.process_images(self.task(init_images=[source], image_mask=mask))
        self.assertEqual(result.images[0].getpixel((0, 0)), source.getpixel((0, 0)))
        self.assertEqual(result.images[0].getpixel((511, 0)), (20, 40, 60, 128))

    def test_switch_away_stops_worker_and_same_checkpoint_keeps_it(self):
        worker = self.make_worker()
        adapter._worker = worker
        adapter.release_if_inactive(SimpleNamespace(qwen21=True))
        self.assertIs(adapter._worker, worker)
        adapter.release_if_inactive(SimpleNamespace(qwen21=False))
        self.assertTrue(worker.stopped)
        self.assertIsNone(adapter._worker)


class WorkerProtocolTests(unittest.TestCase):
    def make_worker(self):
        worker = adapter.Worker.__new__(adapter.Worker)
        worker.process = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        worker.events = queue.Queue()
        worker.stop = mock.Mock()
        return worker

    def test_request_id_filters_stale_messages_and_progress_updates(self):
        worker = self.make_worker()
        for event in (
            {"event": "result", "request_id": "stale"},
            {"event": "progress", "request_id": "current", "step": 2, "total": 8},
            {"event": "result", "request_id": "current", "output_path": "ok.png"},
        ):
            worker.events.put(event)
        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter.uuid, "uuid4", return_value=SimpleNamespace(hex="current")):
            result = worker.generate({"seed": 42}, state)
        self.assertEqual(result["output_path"], "ok.png")
        self.assertEqual((state.sampling_step, state.sampling_steps), (2, 8))
        self.assertEqual(json.loads(worker.process.stdin.getvalue())["request_id"], "current")

    def test_worker_error_resets_session_for_next_request(self):
        worker = self.make_worker()
        worker.events.put({"event": "error", "request_id": "current", "message": "out of memory"})
        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter, "_worker", worker), mock.patch.object(adapter.uuid, "uuid4", return_value=SimpleNamespace(hex="current")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                worker.generate({}, state)
            self.assertIsNone(adapter._worker)
        worker.stop.assert_called_once()

    def test_cancellation_terminates_subprocess(self):
        worker = self.make_worker()
        state = SimpleNamespace(interrupted=True, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter, "_worker", worker):
            self.assertIsNone(worker.generate({}, state))
            self.assertIsNone(adapter._worker)
        worker.stop.assert_called_once()

    def test_unexpected_exit_closes_session_immediately(self):
        worker = self.make_worker()
        worker.events.put({"event": "exit"})
        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter, "_worker", worker):
            with self.assertRaisesRegex(RuntimeError, "runtime exited"):
                worker.generate({}, state)
            self.assertIsNone(adapter._worker)
        worker.stop.assert_called_once()

    def test_broken_pipe_closes_session_immediately(self):
        worker = self.make_worker()
        worker.process.stdin = SimpleNamespace(write=mock.Mock(side_effect=BrokenPipeError("worker died")))
        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter, "_worker", worker):
            with self.assertRaises((RuntimeError, BrokenPipeError)):
                worker.generate({}, state)
            self.assertIsNone(adapter._worker)
        worker.stop.assert_called_once()

    def test_stalled_worker_is_terminated(self):
        worker = self.make_worker()
        worker.events = SimpleNamespace(get=mock.Mock(side_effect=queue.Empty))
        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with mock.patch.object(adapter, "_worker", worker), mock.patch.object(adapter.time, "monotonic", side_effect=[0, 1801]):
            with self.assertRaisesRegex(RuntimeError, "30 minutes"):
                worker.generate({}, state)
            self.assertIsNone(adapter._worker)
        worker.stop.assert_called_once()

    def test_launch_failure_closes_log(self):
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(adapter, "WORK_DIR", Path(directory)), mock.patch("builtins.open", return_value=log), mock.patch.object(adapter.subprocess, "Popen", side_effect=OSError("missing runtime")):
                with self.assertRaisesRegex(OSError, "missing runtime"):
                    adapter.Worker()
        self.assertTrue(log.closed)

    def test_reader_logs_non_protocol_output_and_reports_eof(self):
        worker = self.make_worker()
        worker.log = io.StringIO()
        worker.process.stdout = io.StringIO('startup warning\n{"event": "loading", "request_id": "current"}\n')
        worker._read()
        self.assertEqual(worker.log.getvalue(), "startup warning\n")
        self.assertEqual(worker.events.get_nowait(), {"event": "loading", "request_id": "current"})
        self.assertEqual(worker.events.get_nowait(), {"event": "exit"})

    def test_real_subprocess_roundtrip_then_stop_closes_reader_and_streams(self):
        real_popen = adapter.subprocess.Popen
        child = (
            "import json, sys\n"
            "print('runtime initialized', flush=True)\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    print(json.dumps({'event':'progress', 'request_id':request['request_id'], 'step':1, 'total':1}), flush=True)\n"
            "    print(json.dumps({'event':'result', 'request_id':request['request_id'], 'seed':request['seed']}), flush=True)\n"
        )

        def start_stub(args, **kwargs):
            return real_popen([sys.executable, "-u", "-c", child], **kwargs)

        state = SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(adapter, "WORK_DIR", Path(directory)), mock.patch.object(adapter.subprocess, "Popen", side_effect=start_stub), mock.patch.object(adapter, "_worker", None):
                worker = adapter.Worker()
                adapter._worker = worker
                try:
                    self.assertEqual(worker.generate({"seed": 42}, state)["seed"], 42)
                finally:
                    adapter.stop_worker()
                self.assertIsNotNone(worker.process.poll())
                self.assertFalse(worker.reader_thread.is_alive())
                self.assertTrue(worker.process.stdin.closed)
                self.assertTrue(worker.process.stdout.closed)
                self.assertTrue(worker.log.closed)
                self.assertIn("runtime initialized", (Path(directory) / "worker.stderr.log").read_text())


class DispatchHookTests(unittest.TestCase):
    def test_prompt_counter_follows_selected_preset_before_native_reload(self):
        class FakeInitialModel:
            def get_prompt_lengths_on_ui(self, prompt):
                return len(prompt), 75

            @staticmethod
            def get_krea_prompt_lengths_on_ui(prompt):
                return len(prompt), 512

        qwen_module = ModuleType("modules_forge.qwen21")
        qwen_module.prompt_lengths = lambda prompt: (len(prompt), 1024)
        proxy = SimpleNamespace(qwen21=True, get_prompt_lengths_on_ui=mock.Mock(side_effect=AssertionError("stale Qwen proxy counter used")))
        native_counter = mock.Mock(return_value=(3, 225))
        native = SimpleNamespace(get_prompt_lengths_on_ui=native_counter)
        opts = SimpleNamespace(forge_preset="qwen")
        model_data = SimpleNamespace(sd_model=proxy)
        counter_for_ui = load_function("modules/sd_models.py", "get_prompt_length_counter_for_ui", {
            "opts": opts, "model_data": model_data, "FakeInitialModel": FakeInitialModel,
        })
        with mock.patch.dict(sys.modules, {"modules_forge.qwen21": qwen_module}):
            for resident, preset, expected_limit in (
                (proxy, "qwen", 1024), (proxy, "krea", 512),
                (proxy, "xl", 75), (proxy, "anima", 75),
                (native, "xl", 225), (native, "anima", 225),
                (native, "qwen", 1024), (native, "krea", 512),
            ):
                with self.subTest(preset=preset, qwen_resident=resident is proxy):
                    opts.forge_preset = preset
                    model_data.sd_model = resident
                    counter = counter_for_ui()
                    self.assertEqual(counter("cat"), (3, expected_limit))
                    if resident is native and preset in ("xl", "anima"):
                        self.assertIs(counter, native_counter)
        proxy.get_prompt_lengths_on_ui.assert_not_called()
        self.assertEqual(native_counter.call_count, 2)

    def test_qwen_failure_restores_temporary_settings_without_native_processing(self):
        calls = []
        bridge = SimpleNamespace(is_checkpoint=lambda info: info.qwen21, process_images=mock.Mock(side_effect=RuntimeError("worker failed")))
        package = ModuleType("modules_forge")
        package.qwen21 = bridge
        p = SimpleNamespace(scripts=None, override_settings={"temporary": 20}, override_settings_restore_afterwards=True)
        namespace = {
            "StableDiffusionProcessing": object, "Processed": object,
            "opts": SimpleNamespace(data={"temporary": 10}),
            "sd_models": SimpleNamespace(checkpoint_aliases={}, model_data=SimpleNamespace(forge_loading_parameters={"checkpoint_info": SimpleNamespace(qwen21=True)})),
            "set_config": lambda values, **kwargs: calls.append(dict(values)),
            "manage_model_and_prompt_cache": mock.Mock(side_effect=AssertionError("native sampler invoked")),
        }
        process = load_function("modules/processing.py", "process_images", namespace)
        with mock.patch.dict(sys.modules, {"modules_forge": package}):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                process(p)
        self.assertEqual(calls, [{"temporary": 20}, {"temporary": 10}])
        bridge.process_images.assert_called_once_with(p)

    def test_non_qwen_checkpoint_uses_original_sampler(self):
        calls = []
        bridge = SimpleNamespace(is_checkpoint=lambda info: False, process_images=mock.Mock(side_effect=AssertionError("Qwen intercepted ordinary model")))
        package = ModuleType("modules_forge")
        package.qwen21 = bridge
        p = SimpleNamespace(scripts=None, override_settings={}, override_settings_restore_afterwards=True)
        expected = object()
        namespace = {
            "StableDiffusionProcessing": object, "Processed": object,
            "opts": SimpleNamespace(data={}),
            "sd_models": SimpleNamespace(checkpoint_aliases={}, model_data=SimpleNamespace(forge_loading_parameters={"checkpoint_info": object()})),
            "set_config": lambda *args, **kwargs: None,
            "manage_model_and_prompt_cache": lambda p: calls.append("load"),
            "sd_samplers": SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda p: calls.append("sampler")),
            "profiling": SimpleNamespace(Profiler=nullcontext),
            "process_images_inner": lambda p: expected,
        }
        process = load_function("modules/processing.py", "process_images", namespace)
        with mock.patch.dict(sys.modules, {"modules_forge": package}):
            self.assertIs(process(p), expected)
        self.assertEqual(calls, ["load", "sampler"])

    def test_reload_reuses_qwen_proxy_unless_forced(self):
        target = SimpleNamespace(qwen21=True)
        current = SimpleNamespace(qwen21=True)
        replacement = object()
        loading = {"checkpoint_info": target}
        bridge = SimpleNamespace(
            is_checkpoint=lambda info: info.qwen21,
            load_forge_model=mock.Mock(return_value=(replacement, True)),
            release_if_inactive=mock.Mock(),
        )
        package = ModuleType("modules_forge")
        package.qwen21 = bridge
        namespace = {"model_data": SimpleNamespace(forge_loading_parameters=loading, forge_hash=str(loading), sd_model=current)}
        reload_model = load_function("modules/sd_models.py", "forge_model_reload", namespace)
        with mock.patch.dict(sys.modules, {"modules_forge": package}):
            self.assertEqual(reload_model(), (current, False))
            bridge.load_forge_model.assert_not_called()
            self.assertEqual(reload_model(force_full_unload=True), (replacement, True))
        bridge.load_forge_model.assert_called_once_with(target)

if __name__ == "__main__":
    unittest.main()
