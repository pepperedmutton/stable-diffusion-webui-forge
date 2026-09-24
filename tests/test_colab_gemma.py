"""Offline harness safety/config tests; no model weights, Torch or GPU required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'colab/verify_gemma.py'
spec = importlib.util.spec_from_file_location('gemma_component_qa', SCRIPT)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class GemmaHarnessTests(unittest.TestCase):
    def test_publisher_config_integrity_and_architecture(self):
        qa.validate_config()
        self.assertIn('/resolve/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298/', qa.CONFIG_SOURCE)
        self.assertEqual(qa.TEXT_CONFIG['num_hidden_layers'], 34)
        self.assertEqual(qa.TEXT_CONFIG['hidden_size'], 2560)

    def test_modified_config_rejected(self):
        with mock.patch.dict(qa.TEXT_CONFIG, {'hidden_size': 64}):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                qa.validate_config()

    def test_plan_does_not_touch_weights_outputs_or_import_torch(self):
        before = 'torch' in sys.modules
        capture = io.StringIO()
        with mock.patch.object(qa, 'persistent_paths', side_effect=AssertionError('No disk operation')), mock.patch.object(qa, 'run_component', side_effect=AssertionError('No inference')), contextlib.redirect_stdout(capture):
            self.assertEqual(qa.main([]), 0)
        plan = json.loads(capture.getvalue())
        self.assertEqual(plan['mode'], 'plan_only_no_inference')
        self.assertIn('no NewBie image pipeline', plan['provenance']['proof_scope'])
        self.assertEqual('torch' in sys.modules, before)

    def state(self):
        return {**{'model.tensor_' + str(i): i for i in range(444)}, 'spiece_model': b'tokenizer'}

    def test_normalization_removes_only_documented_prefix_and_tokenizer(self):
        actual = qa.normalize_keys(self.state())
        self.assertEqual(len(actual), 444)
        self.assertEqual(actual['tensor_443'], 443)

    def test_missing_extra_and_unknown_keys_rejected(self):
        missing = self.state()
        missing.pop('spiece_model')
        extra = self.state()
        extra['other'] = None
        unknown = self.state()
        unknown['unknown'] = unknown.pop('model.tensor_0')
        for state in (missing, extra, unknown):
            with self.subTest(size=len(state)), self.assertRaises(ValueError):
                qa.normalize_keys(state)

    def test_token_range_and_length_are_bounded(self):
        qa.validate_token_ids([2, 123, 262207], 262208)
        for ids in ([], [262208], [-1], [True], [1.0], [1] * 129):
            with self.subTest(ids=ids[:3]), self.assertRaises(ValueError):
                qa.validate_token_ids(ids, 262208)

    def test_unmounted_drive_rejected_before_creation(self):
        with mock.patch.object(qa.sys, 'platform', 'linux'), mock.patch.object(qa.os.path, 'ismount', return_value=False), mock.patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(qa.NotReady):
                qa.persistent_paths(Path('/content/drive/MyDrive/ForgeColab'), None)
        mkdir.assert_not_called()

    def test_run_without_mounted_drive_reports_not_ready(self):
        capture = io.StringIO()
        with mock.patch.object(qa, 'persistent_paths', side_effect=qa.NotReady('Mount Drive')), mock.patch.object(qa, 'run_component', side_effect=AssertionError('No inference')), contextlib.redirect_stdout(capture):
            self.assertEqual(qa.main(['--run']), 2)
        result = json.loads(capture.getvalue())
        self.assertEqual(result['status'], 'not_ready')
        self.assertFalse(result['inference_attempted'])

    def test_evidence_write_is_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.json'
            qa.atomic_json(path, {'status': 'not_ready', 'inference_attempted': False})
            self.assertEqual(json.loads(path.read_text())['status'], 'not_ready')
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
