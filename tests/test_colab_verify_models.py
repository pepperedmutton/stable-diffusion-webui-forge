"""Offline QA-harness contracts; these tests do not run or emulate inference."""
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_verify_models_test", ROOT / "colab/verify_models.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class QAPlanTests(unittest.TestCase):
    def test_default_manifest_contains_eleven_unique_models_and_twenty_two_cases(self):
        profiles = qa.load_profiles(qa.DEFAULT_MANIFEST)
        self.assertEqual(len(profiles), 11)
        self.assertEqual(sum(len(profile["modes"]) for profile in profiles), 22)
        self.assertEqual(len({profile["model"] for profile in profiles}), 11)
        self.assertFalse(any(qa.canonical(profile["model"]) == "novsw" for profile in profiles))
        self.assertEqual(sum(profile["model"] == "waiIllustriousSDXL_v150.safetensors" for profile in profiles), 1)

    def test_default_is_offline_and_does_not_create_artifacts(self):
        output = io.StringIO()
        with mock.patch.object(qa, "API", side_effect=AssertionError("No API in plan mode")), \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("No directories in plan mode")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(qa.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["expected_cases"], 22)
        self.assertEqual(qa.QA_ROOT.as_posix(), "/content/drive/MyDrive/ForgeColab/qa")

    def test_model_aliases_match_basename_and_hash(self):
        self.assertEqual(qa.canonical("/models/Stable-diffusion/waiIllustriousSDXL_v150.safetensors [123abc]"), "waiillustrioussdxl_v150")
        profile = {"model": "waiIllustriousSDXL_v150.safetensors"}
        model = {"title": "waiIllustriousSDXL_v150.safetensors [123abc]"}
        self.assertEqual(qa.resolve_model(profile, [model]), model)
        with self.assertRaises(ValueError):
            qa.resolve_model(profile, [model, model])

    def test_api_accepts_only_private_loopback_origins(self):
        for origin in ("http://127.0.0.1:7860", "http://localhost:7860", "http://[::1]:7860"):
            qa.API(origin, 1)
        for origin in ("https://example.com", "http://10.0.2.2:7860", "http://localhost.evil.test", "http://u:p@localhost", "http://localhost/path", "http://localhost?x=1"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                qa.API(origin, 1)

    def test_manifest_rejects_duplicate_ids_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "profiles.json"
            profile = {"id": "same", "preset": "xl", "model": "example"}
            manifest.write_text(json.dumps({"profiles": [profile, profile]}))
            with self.assertRaisesRegex(ValueError, "unique"):
                qa.load_profiles(manifest)

    def test_missing_inventory_is_not_ready_without_settings_or_generation(self):
        class InventoryAPI:
            def __init__(self, *args):
                pass

            def idle(self, seconds):
                return True

            def call(self, method, endpoint, *args, **kwargs):
                if method != "GET" or endpoint not in {"sd-models", "sd-modules", "script-info"}:
                    raise AssertionError("Inventory mode must not change settings or generate images")
                return []

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with mock.patch.object(qa, "API", InventoryAPI), \
                    mock.patch.object(qa.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(qa.main(["--inventory-only", "--output", str(output)]), 1)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["status"], "inventory_only_not_generated")
            self.assertEqual(len(report["cases"]), 22)
            self.assertTrue(all(case["status"] == "not_ready" and not case["generation_attempted"] for case in report["cases"]))
            self.assertFalse(list(output.glob("*.png")))


class QAImageProofTests(unittest.TestCase):
    def image(self, uniform=False):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (512, 512), "white")
        if not uniform:
            ImageDraw.Draw(image).rectangle((100, 100, 410, 410), fill="green")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def test_blank_image_rejected_and_dimensions_hash_recorded(self):
        with self.assertRaisesRegex(ValueError, "blank"):
            qa.image_proof(self.image(uniform=True))
        proof = qa.image_proof(self.image())
        self.assertEqual((proof["width"], proof["height"]), (512, 512))
        self.assertEqual(len(proof["pixel_sha256"]), 64)

    def test_response_must_prove_model_and_seed_and_changed_reference(self):
        data = self.image()
        info = {"sd_model_name": "model", "seed": 17, "all_seeds": [17], "steps": 20, "width": 512, "height": 512}
        response = {"images": [base64.b64encode(data).decode()], "info": json.dumps(info)}
        request = {"seed": 17, "steps": 20}
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "test.png"
            with self.assertRaisesRegex(ValueError, "Wrong model"):
                qa.verify_response({"model": "different"}, response, image, request)
            with self.assertRaisesRegex(ValueError, "fixed seed"):
                qa.verify_response({"model": "model"}, response, image, {**request, "seed": 18})
            with self.assertRaisesRegex(ValueError, "unchanged"):
                qa.verify_response({"model": "model"}, response, image, request, reference=data)
            proof = qa.verify_response({"model": "model"}, response, image, request)
            self.assertEqual(proof["actual_model"], "model")


if __name__ == "__main__":
    unittest.main()
