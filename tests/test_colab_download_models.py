"""Small offline transfer simulations; never downloads a real model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("downloads", Path(__file__).resolve().parents[1] / "colab" / "download_models.py")
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


class Response:
    def __init__(self, status, data, headers=None):
        self.status_code, self.data, self.headers = status, data, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, **kwargs):
        yield self.data
    def close(self):
        pass


class Session:
    def __init__(self, response):
        self.response, self.calls = response, []
    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response.pop(0) if isinstance(self.response, list) else self.response
    def close(self):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models, self.state = self.root / "models", self.root / "download-state"
        self.models.mkdir()
        self.content = b"model-test-bytes"
        self.item = {"path": "Stable-diffusion/test.safetensors", "bytes": len(self.content),
                     "sha256": hashlib.sha256(self.content).hexdigest(),
                     "url": "https://civitai.com/api/download/models/123?fileId=456"}
    def partial(self, data):
        identity = hashlib.sha256((self.item["path"] + downloads.expected_hash(self.item)).encode()).hexdigest()
        path = self.state / "parts" / (identity + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    def transfer(self, response):
        session = Session(response)
        result = downloads.download_file(self.item, self.models, self.state, retries=1, session_factory=lambda: session)
        return result, session
    def test_success_publishes_only_after_checksum(self):
        result, session = self.transfer(Response(200, self.content, {"Content-Length": str(len(self.content))}))
        self.assertEqual((self.models / self.item["path"]).read_bytes(), self.content)
        self.assertEqual(result["status"], "verified")
        self.assertFalse(session.trust_env)
        self.assertNotIn("Authorization", session.calls[0][1]["headers"])
    def test_resume_requests_exact_prefix_offset(self):
        part = self.partial(self.content[:5])
        result, session = self.transfer(Response(206, self.content[5:], {"Content-Range": f"bytes 5-{len(self.content)-1}/{len(self.content)}", "Content-Length": str(len(self.content)-5)}))
        self.assertEqual(session.calls[0][1]["headers"]["Range"], "bytes=5-")
        self.assertFalse(part.exists())
        self.assertEqual((self.models / self.item["path"]).read_bytes(), self.content)
    def test_server_ignoring_range_never_appends_full_file(self):
        self.partial(self.content[:5])
        self.transfer(Response(200, self.content, {"Content-Length": str(len(self.content))}))
        self.assertEqual((self.models / self.item["path"]).read_bytes(), self.content)
    def test_wrong_range_does_not_destroy_existing_prefix(self):
        part = self.partial(self.content[:5])
        with self.assertRaises(downloads.IntegrityError):
            self.transfer(Response(206, self.content[5:], {"Content-Range": "bytes 0-14/15"}))
        self.assertEqual(part.read_bytes(), self.content[:5])
    def test_size_mismatch_rejected_before_write(self):
        with self.assertRaises(downloads.IntegrityError):
            self.transfer(Response(200, b"html", {"Content-Length": "4"}))
        self.assertFalse((self.models / self.item["path"]).exists())
    def test_corrupt_file_is_never_published(self):
        with self.assertRaises(downloads.IntegrityError):
            self.transfer(Response(200, b"x" * len(self.content)))
        self.assertFalse((self.models / self.item["path"]).exists())
    def test_existing_user_file_is_not_replaced(self):
        target = self.models / self.item["path"]
        target.parent.mkdir()
        target.write_bytes(b"user file")
        with self.assertRaises(downloads.IntegrityError):
            self.transfer(Response(200, self.content))
        self.assertEqual(target.read_bytes(), b"user file")
    def test_existing_verified_file_needs_no_network(self):
        target = self.models / self.item["path"]
        target.parent.mkdir()
        target.write_bytes(self.content)
        with mock.patch.object(downloads.requests, "Session", side_effect=AssertionError("No network expected")):
            result = downloads.download_file(self.item, self.models, self.state)
        self.assertTrue(result["existing"])
    def test_complete_partial_is_verified_and_promoted(self):
        self.partial(self.content)
        result, session = self.transfer(Response(200, b"must not be read"))
        self.assertEqual(session.calls, [])
        self.assertEqual(result["status"], "verified")
    def test_401_is_not_retried_or_authenticated(self):
        with self.assertRaises(downloads.AccessDenied):
            self.transfer(Response(401, b"login required"))
        self.assertFalse((self.models / self.item["path"]).exists())
    def test_git_blob_checksum_is_supported(self):
        self.item.pop("sha256")
        self.item["git_sha1"] = hashlib.sha1(f"blob {len(self.content)}\0".encode() + self.content).hexdigest()
        self.transfer(Response(200, self.content))
        self.assertTrue(downloads.verify_file(self.models / self.item["path"], self.item))
    def test_unsafe_paths_urls_and_unpinned_revisions_rejected(self):
        for change in ({"path": "../outside"}, {"path": "C:/outside"}, {"url": "http://civitai.com/api/download/models/123?fileId=456"}, {"url": "https://evil.test/model"}, {"url": "https://huggingface.co/user/repo/resolve/main/model"}, {"url": "https://civitai.com/api/download/models/123?fileId=456&token=secret"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                downloads.validate_item({**self.item, **change})
    def test_unmounted_drive_rejected_without_creating_paths(self):
        with mock.patch.object(downloads.sys, "platform", "linux"), mock.patch.object(downloads.os.path, "ismount", return_value=False), mock.patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "Mount Google Drive"):
                downloads.validate_storage(Path("/content/drive/MyDrive/ForgeColab"))
        mkdir.assert_not_called()
    def test_staging_option_is_not_available(self):
        with self.assertRaises(SystemExit):
            downloads.main(['--staging-root', '/content/forge-model-staging'])
        self.assertFalse(hasattr(downloads, 'validate_staging'))
    def test_api_key_never_follows_cross_domain_redirect(self):
        session = Session([Response(302, b'', {'Location': 'https://b2.civitai.com/file?signed=temporary'}), Response(200, self.content)])
        downloads.download_file(self.item, self.models, self.state, retries=1, session_factory=lambda: session, api_key='test-key')
        self.assertEqual(session.calls[0][1]['headers']['Authorization'], 'Bearer test-key')
        self.assertNotIn('Authorization', session.calls[1][1]['headers'])
        self.assertTrue(all(not call[1]['allow_redirects'] for call in session.calls))
    def test_api_key_never_sent_to_huggingface(self):
        self.item['url'] = 'https://huggingface.co/user/repo/resolve/' + 'a' * 40 + '/file?download=true'
        session = Session(Response(200, self.content))
        downloads.download_file(self.item, self.models, self.state, retries=1, session_factory=lambda: session, api_key='test-key')
        self.assertNotIn('Authorization', session.calls[0][1]['headers'])
    def test_insecure_redirect_never_receives_request(self):
        session = Session(Response(302, b'', {'Location': 'http://civitai.com/file'}))
        with self.assertRaises(downloads.IntegrityError):
            downloads.download_file(self.item, self.models, self.state, retries=1, session_factory=lambda: session, api_key='test-key')
        self.assertEqual(len(session.calls), 1)
    def test_embeddings_have_separate_receipt(self):
        embedding = next(item for item in downloads.CATALOG['files'] if item['storage'] == 'embeddings')
        downloads.merge_verified_manifest(self.root, ['embeddings/' + embedding['path']], 'embeddings')
        self.assertTrue((self.root / 'embeddings-manifest.json').is_file())
        self.assertFalse((self.root / 'models-manifest.json').exists())
    def test_dry_plan_does_not_read_key_or_write_files(self):
        with mock.patch.object(downloads, 'validate_storage', side_effect=AssertionError('No writes')):
            self.assertEqual(downloads.main([]), 0)
    def test_catalog_is_complete_and_credential_free(self):
        public = downloads.CATALOG["files"]
        self.assertEqual(len(public), 52)
        self.assertEqual(sum(item["bytes"] for item in public), 140109960303)
        self.assertEqual(sum(item['group'] == 'checkpoint' for item in public), 8)
        self.assertEqual(downloads.CATALOG['default_checkpoint'], 'waiIllustriousSDXL_v150.safetensors')
        self.assertNotIn('Novsw', repr(downloads.CATALOG))
        self.assertNotIn('D:', repr(downloads.CATALOG))
        for item in public:
            downloads.validate_item(item)
            self.assertNotIn("token=", item.get("url", ""))


if __name__ == "__main__":
    unittest.main()

