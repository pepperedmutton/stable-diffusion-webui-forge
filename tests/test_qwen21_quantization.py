"""GPU regression for INT8 caches surviving parent-level CPU offload."""

import importlib.util
from pathlib import Path
import unittest


class Int8OffloadTests(unittest.TestCase):
    def test_parent_offload_after_forward_and_hook_rebuild(self):
        try:
            import torch
            import bitsandbytes as bnb
            from accelerate import cpu_offload_with_hook
            from accelerate.hooks import remove_hook_from_module
        except ImportError as error:
            self.skipTest(str(error))
        if not torch.cuda.is_available() or bnb.__version__ != "0.50.2":
            self.skipTest("Requires isolated Qwen runtime with CUDA and bitsandbytes 0.50.2")
        path = Path(__file__).resolve().parents[1] / "modules_forge" / "qwen21_quantization.py"
        spec = importlib.util.spec_from_file_location("qwen21_quantization_tested", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

        layer = bnb.nn.Linear8bitLt(128, 64, has_fp16_weights=False, threshold=6.0).to("cuda")
        model = torch.nn.Sequential(layer).eval()
        self.assertEqual(helper.patch_int8_cpu_offload(model), 1)
        apply_function = model._apply
        layout_hook = layer._qwen21_int8_contiguous_hook_handle
        self.assertEqual(helper.patch_int8_cpu_offload(model), 1)
        self.assertIs(model._apply, apply_function)
        self.assertIs(layer._qwen21_int8_contiguous_hook_handle, layout_hook)
        x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = model(x)
            for _ in range(2):
                model, hook = cpu_offload_with_hook(model, "cuda:0")
                actual = model(x)
                self.assertTrue(torch.equal(actual, expected))
                hook.offload()
                self.assertEqual(layer.weight.device.type, "cpu")
                for container in (layer.weight, layer.state):
                    for field in ("CB", "SCB"):
                        value = getattr(container, field)
                        if value is not None:
                            self.assertEqual(value.device.type, "cpu")
                    if container.CB is not None:
                        self.assertEqual(container.CB.data_ptr(), layer.weight.data_ptr())
                remove_hook_from_module(model, recurse=True)
                self.assertIs(model._apply, apply_function)
                self.assertIs(layer._qwen21_int8_contiguous_hook_handle, layout_hook)
            model.to("cuda")
            self.assertTrue(torch.equal(model(x), expected))
            # A transpose keeps the values but changes their storage order;
            # the CUDA INT8 quantization kernel otherwise reads that incorrectly.
            strided = torch.randn(128, 16, device="cuda", dtype=torch.bfloat16).T
            self.assertFalse(strided.is_contiguous())
            contiguous_result = model(strided.contiguous())
            self.assertTrue(torch.equal(model(strided), contiguous_result))
            self.assertTrue(torch.equal(layer(x=strided), contiguous_result))


if __name__ == "__main__":
    unittest.main()
