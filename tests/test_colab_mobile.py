"""Colab mobile patch contracts; never modify the running Forge checkout.

Run with: python -m unittest discover -s tests -p test_colab_mobile.py -v
These tests exercise temporary copies and do not start a server or load models.
Node syntax checks run when Node is available on PATH (or FORGE_TEST_NODE).
Real touch/browser interaction needs separate browser/device acceptance testing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "colab" / "kit"
ORIGINAL_JS = REPO / "javascript" / "outpaint.js"
NODE = os.environ.get("FORGE_TEST_NODE") or shutil.which("node")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAFETY = load_module("colab_test_outpaint_safety", KIT / "outpaint_safety_patch.py")
with patch.dict(sys.modules, {"outpaint_safety_patch": SAFETY}):
    MOBILE = load_module("colab_test_mobile_patch", KIT / "mobile_patch.py")


class ColabMobilePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="forge-colab-mobile-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "javascript").mkdir()
        (self.root / "launch.py").write_text("# Isolated patch fixture\n", encoding="utf-8")
        self.javascript = self.root / "javascript" / "outpaint.js"
        self.javascript.write_bytes(ORIGINAL_JS.read_bytes())
        self.stylesheet = self.root / "user.css"
        self.custom_css = b"/* Existing user preferences */\r\n.custom { color: navy; }\r\n"
        self.stylesheet.write_bytes(self.custom_css)
        self.guard = self.root / "javascript" / "00_forge_mobile_guard.js"

    def invoke(self, *arguments, script="mobile_patch.py", success=True):
        result = subprocess.run(
            [sys.executable, str(KIT / script), "--root", str(self.root), *arguments],
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def test_current_outpaint_source_matches_audited_revision(self):
        source = SAFETY.normalized(ORIGINAL_JS.read_bytes().decode("utf-8"))
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), SAFETY.AUDITED_SHA256)

    def test_check_previews_all_files_without_writing(self):
        before = self.snapshot()
        result = self.invoke("--check")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(result.stdout.count("Would update:"), 3)
        self.assertIn("no files changed", result.stdout)
        self.assertNotIn("applied to this copy", result.stdout)

    def test_repeat_apply_is_byte_identical_and_keeps_custom_css(self):
        self.invoke()
        first = self.snapshot()
        result = self.invoke()
        self.assertEqual(first, self.snapshot())
        self.assertEqual(result.stdout.count("Already current:"), 3)
        self.assertTrue(self.stylesheet.read_bytes().startswith(self.custom_css))
        self.assertEqual(self.guard.read_bytes(), (KIT / "mobile_guard.js").read_bytes())
        css = self.stylesheet.read_text(encoding="utf-8")
        for start, end in (
            (MOBILE.CSS_START, MOBILE.CSS_END),
            ("/* BEGIN FORGE MOBILE PARAMETER GUARD */", "/* END FORGE MOBILE PARAMETER GUARD */"),
            (SAFETY.CSS_START, SAFETY.CSS_END),
        ):
            self.assertEqual(css.count(start), 1)
            self.assertEqual(css.count(end), 1)

    def test_apply_without_existing_user_stylesheet(self):
        self.stylesheet.unlink()
        self.invoke()
        self.assertIn(MOBILE.CSS_START, self.stylesheet.read_text(encoding="utf-8"))
        self.assertTrue(self.guard.is_file())

    def test_unknown_source_fails_before_any_write(self):
        self.javascript.write_bytes(self.javascript.read_bytes() + b"\n// Unknown customization\n")
        before = self.snapshot()
        result = self.invoke(success=False)
        self.assertEqual(before, self.snapshot())
        self.assertIn("differs from the audited checkout", result.stderr)

    def test_incomplete_or_duplicate_markers_fail_before_any_write(self):
        cases = (
            MOBILE.CSS_START,
            MOBILE.CSS_END + "\n" + MOBILE.CSS_START,
            MOBILE.CSS_START + "\n" + MOBILE.CSS_START + "\n" + MOBILE.CSS_END,
            "/* BEGIN FORGE MOBILE PARAMETER GUARD */",
            SAFETY.CSS_START,
        )
        for broken in cases:
            with self.subTest(markers=broken):
                self.stylesheet.write_text(broken, encoding="utf-8")
                before = self.snapshot()
                self.invoke(success=False)
                self.assertEqual(before, self.snapshot())

    def test_modified_existing_safety_patch_is_rejected(self):
        self.invoke()
        text = self.javascript.read_bytes().decode("utf-8")
        self.javascript.write_bytes(text.replace("Gesture incomplete.", "Changed gesture.").encode())
        before = self.snapshot()
        result = self.invoke(success=False)
        self.assertEqual(before, self.snapshot())
        self.assertIn("patch has been modified", result.stderr)

    def test_existing_guard_is_replaced_with_current_canonical_guard(self):
        self.guard.write_text("// Previous generated guard\n", encoding="utf-8")
        self.invoke()
        self.assertEqual(self.guard.read_bytes(), (KIT / "mobile_guard.js").read_bytes())

    def test_crlf_source_keeps_crlf_line_endings(self):
        source = SAFETY.normalized(self.javascript.read_bytes().decode()).replace("\n", "\r\n")
        self.javascript.write_bytes(source.encode())
        self.invoke()
        data = self.javascript.read_bytes()
        self.assertIn(b"\r\n", data)
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))

    def test_standalone_outpaint_and_mobile_patch_order_is_idempotent(self):
        self.invoke(script="outpaint_safety_patch.py")
        self.invoke()
        first = self.snapshot()
        self.invoke(script="outpaint_safety_patch.py")
        self.invoke()
        self.assertEqual(first, self.snapshot())

    def test_desktop_geometry_and_immediate_commit_branch_are_retained(self):
        self.invoke()
        source = self.javascript.read_text(encoding="utf-8")
        self.assertIn("const hit = coarsePointer ? 44 : 28;", source)
        self.assertIn("const corner = coarsePointer ? 48 : 38;", source)
        self.assertIn("touchDraft: Boolean(touchMode() && touchSession)", source)
        self.assertIn("} else {\n            synchronizeArea(finalArea, active.context, true);", source)
        self.assertIn("} else {\n            synchronizeArea(candidate, state.context, true);", source)

    def test_patch_has_no_app_bridge_and_english_outpaint_controls(self):
        for name in ("mobile.css", "mobile_guard.css", "mobile_guard.js", "mobile_patch.py", "outpaint_safety_patch.py"):
            source = (KIT / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertNotRegex(source, r"(?i)forgepocket|intent://|__FORGE_PAIR_TOKEN__|AndroidBridge")
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", SAFETY.HELPERS))
        for label in ("Edit outpaint", "Apply and lock", "Discard and lock"):
            self.assertIn(label, SAFETY.HELPERS)

    @unittest.skipUnless(NODE, "Node is unavailable; set FORGE_TEST_NODE to enable JavaScript syntax checks")
    def test_generated_javascript_parses(self):
        self.invoke()
        for path in (self.javascript, self.guard):
            with self.subTest(file=path.name):
                result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
