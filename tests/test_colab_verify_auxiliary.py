"""Offline auxiliary-harness contracts. No Forge, model loads or GPU inference."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_verify_auxiliary_test", ROOT / "colab/verify_auxiliary.py")
aux = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux)


class AuxiliaryPlanTests(unittest.TestCase):
    def test_default_is_offline_with_six_cases_and_explicit_limits(self):
        output = io.StringIO()
        with mock.patch.object(aux, "AuxiliaryAPI", side_effect=AssertionError("No network in plan")), contextlib.redirect_stdout(output):
            self.assertEqual(aux.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan["cases"]), 6)
        self.assertIn("faceid-plusv2", plan["cases"])
        self.assertEqual({value["id"] for value in plan["limitations"]}, {"legacy-qwen-image", "vae-preview-networks", "private-loras"})

    def test_control_case_includes_detection_and_comparison_baseline(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            aux.main(["--cases", "openpose-control"])
        self.assertEqual(list(json.loads(output.getvalue())["cases"]), ["openpose-full", "sdxl-baseline", "openpose-control"])

    def test_landmarks_ignore_missing_points_and_preserve_each_detector(self):
        frame = {"people": [{"pose_keypoints_2d": [1, 2, 1, 0, 0, 0, 3, 4, 1], "hand_left_keypoints_2d": [2, 3, 1], "face_keypoints_2d": None}]}
        self.assertEqual(aux.pose_counts([json.dumps(frame)]), {"people": 1, "body": 2, "left_hand": 1, "right_hand": 0, "face": 0})

    def test_control_units_are_disabled_except_the_one_requested(self):
        infos = [{"name": "ControlNet", "is_alwayson": True, "is_img2img": False, "args": [{}, {}, {}]}]
        scripts = aux.unit_scripts(infos)
        self.assertEqual(scripts["ControlNet"]["args"], [{"enabled": False}] * 3)
        unit = {"enabled": True, "module": "None", "model": "openpose"}
        scripts = aux.unit_scripts(infos, unit)
        self.assertEqual(scripts["ControlNet"]["args"], [unit, {"enabled": False}, {"enabled": False}])
        with self.assertRaises(aux.NotReady):
            aux.unit_scripts([])

    def test_missing_dependency_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(aux.NotReady, "missing"):
                aux.require_files(Path(temporary), ["models/example.pth"])


class AuxiliaryEvidenceTests(unittest.TestCase):
    def fixture(self, size=(128, 128)):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", size, "white")
        ImageDraw.Draw(image).ellipse((10, 10, size[0] - 10, size[1] - 10), fill="green")
        return image

    def test_interpolation_fallback_is_never_neural_pass(self):
        from PIL import Image, ImageDraw
        source = self.fixture()
        data = aux.png_bytes(source)
        for mode in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS):
            result = source.resize((512, 512), mode)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "interpolation-only"):
                aux.require_neural_upscale(data, aux.png_bytes(result))
        ImageDraw.Draw(result).rectangle((0, 0, 20, 20), fill="blue")
        aux.require_neural_upscale(data, aux.png_bytes(result))

    def test_transport_timeout_aborts_before_later_requests(self):
        class TimeoutAPI:
            def __init__(self, *args):
                pass

            def idle(self, seconds):
                return True

            def call(self, method, endpoint, *args):
                if method == "GET" and endpoint in {"sd-models", "script-info"}:
                    return []
                raise AssertionError("No further API calls after uncertain timeout")

            def control(self, method, endpoint, *args):
                if endpoint == "module_list":
                    return {"module_list": ["openpose_full"]}
                raise TimeoutError("Offline harness timeout fixture")

        with tempfile.TemporaryDirectory() as temporary:
            drive = Path(temporary)
            input_image = drive / "input.png"
            input_image.write_bytes(aux.png_bytes(self.fixture((512, 512))))
            for name in ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"):
                path = drive / "models/ControlNetPreprocessor/openpose" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"offline fixture only")
            output = drive / "results"
            with mock.patch.object(aux, "AuxiliaryAPI", TimeoutAPI), contextlib.redirect_stdout(io.StringIO()):
                result = aux.main(["--run", "--drive-root", str(drive), "--input-image", str(input_image), "--output", str(output), "--cases", "openpose-full", "realesrgan-anime6b"])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(report["status"], "aborted")
            self.assertEqual(len(report["cases"]), 1)
            self.assertEqual(report["cases"][0]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
