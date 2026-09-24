"""Protocol/input checks that do not load model weights or require a GPU."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


WORKER = Path(__file__).resolve().parents[1] / "modules_forge" / "qwen21_worker.py"
SPEC = importlib.util.spec_from_file_location("qwen21_worker", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}')
        self.request = {"model_dir": str(self.root), "output_path": str(self.root / "result.png")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_legacy_qwen_weights(self):
        (self.root / "model_index.json").write_text('{"_class_name":"QwenImagePipeline"}')
        with self.assertRaisesRegex(ValueError, "not a Qwen-Image 2.1"):
            worker.validate_generate(self.request)

    def test_dimensions_are_not_silently_rounded(self):
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            worker.validate_generate({**self.request, "width": 1000})

    def write_precision(self, component, quant):
        directory = self.root / component
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps({"quantization_config": quant}))

    def test_precision_is_read_from_both_components_not_directory_name(self):
        self.assertEqual(worker.quantization_profile(self.root), "BF16")
        nf4 = {"quant_method": "bitsandbytes", "load_in_4bit": True,
               "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
               "bnb_4bit_compute_dtype": "bfloat16"}
        int8 = {"quant_method": "bitsandbytes", "load_in_8bit": True}
        for quant, expected in ((nf4, "NF4"), (int8, "INT8")):
            for component in ("transformer", "text_encoder"):
                self.write_precision(component, quant)
            self.assertEqual(worker.quantization_profile(self.root), expected)
            worker.validate_generate({**self.request, "precision": expected.lower()})
        self.write_precision("text_encoder", nf4)
        with self.assertRaisesRegex(ValueError, "same precision"):
            worker.quantization_profile(self.root)

    def test_precision_mismatch_and_partial_quantization_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            worker.validate_generate({**self.request, "precision": "nf4"})
        self.write_precision("transformer", {"quant_method": "bitsandbytes", "load_in_8bit": True})
        with self.assertRaisesRegex(ValueError, "text_encoder"):
            worker.quantization_profile(self.root)

    def test_rejects_reference_overwrite_and_excess_references(self):
        image = self.root / "input.png"
        image.touch()
        with self.assertRaisesRegex(ValueError, "differ"):
            worker.validate_generate({**self.request, "input_images": [str(image)], "output_path": str(image)})
        with self.assertRaisesRegex(ValueError, "at most 10"):
            worker.validate_generate({**self.request, "input_images": [str(image)] * 11})

    def test_protocol_recovers_after_invalid_request(self):
        commands = '\n'.join([
            '{"command":"unknown","request_id":"bad"}',
            '{"command":"unload","request_id":"ok"}',
            '{"command":"shutdown","request_id":"end"}',
        ]) + '\n'
        result = subprocess.run([sys.executable, "-u", str(WORKER)], input=commands,
                                capture_output=True, text=True, check=True)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([event["event"] for event in events], ["ready", "error", "unloaded", "shutdown"])
        self.assertEqual(events[1]["request_id"], "bad")
        self.assertEqual(events[2]["request_id"], "ok")
        self.assertIn("ValueError", result.stderr)

    def test_generation_preserves_rgba_and_forwards_native_arguments(self):
        from PIL import Image

        reference = self.root / "reference.png"
        Image.new("RGBA", (64, 64), (12, 34, 56, 0)).save(reference)
        recorded = {}
        events = []

        class Pipeline:
            def __call__(self, **kwargs):
                recorded.update(kwargs)
                kwargs["callback_on_step_end"](self, 1, 0, {})
                return SimpleNamespace(images=[Image.new("RGBA", (64, 64), (12, 34, 56, 78))])

        instance = worker.Qwen21Worker(lambda event, **kwargs: events.append((event, kwargs)))
        instance._load = lambda path: None
        instance.pipe = Pipeline()
        torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed))
        with patch.dict(sys.modules, {"torch": torch}):
            result = instance.generate({**self.request, "prompt": "edit", "negative_prompt": "ignored",
                                        "width": 64, "height": 64, "steps": 2, "seed": 42,
                                        "input_images": [str(reference)]})
        self.assertNotIn("negative_prompt", recorded)
        self.assertEqual(recorded["true_cfg_scale"], 1.0)
        self.assertEqual(recorded["image"][0].mode, "RGBA")
        self.assertEqual(events[-1], ("progress", {"step": 2, "total": 2}))
        self.assertEqual(result["mode"], "RGBA")
        with Image.open(result["output_path"]) as generated:
            self.assertEqual(generated.getpixel((0, 0)), (12, 34, 56, 78))

    def test_model_load_preserves_untiled_vae_to_avoid_periodic_seams(self):
        pipe = Mock()
        pipeline_class = SimpleNamespace(from_pretrained=Mock(return_value=pipe))
        torch = SimpleNamespace(
            bfloat16="bf16", cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True,
                get_device_name=lambda index: "test GPU",
            ),
        )
        instance = worker.Qwen21Worker(Mock())
        instance.unload = Mock()
        with patch.dict(sys.modules, {"torch": torch, "diffusers": SimpleNamespace(QwenImage21Pipeline=pipeline_class)}):
            instance._load(str(self.root))
        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=0)
        pipe.vae.enable_tiling.assert_not_called()
        self.assertEqual(pipeline_class.from_pretrained.call_args.kwargs["dtype"], "bf16")


if __name__ == "__main__":
    unittest.main()
