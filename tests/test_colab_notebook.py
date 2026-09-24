"""Offline notebook flow checks; no Colab access, package installs, or models."""
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "colab/Forge_Colab.ipynb").read_text(encoding="utf-8"))
CODE = ["".join(cell["source"]) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]
REMOTE = "https://github.com/pepperedmutton/stable-diffusion-webui-forge.git"


class NotebookTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.storage = Path(directory.name) / "Drive storage"
        self.repo = self.storage / "forge"
        (self.repo / ".git").mkdir(parents=True)
        (self.repo / ".colab").mkdir()
        (self.repo / ".colab/installed.json").write_text("{}")
        python = self.repo / "runtimes/qwen-image-2.1/bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"interpreter fixture")
        module = ast.parse(CODE[1])
        # Define the actual notebook functions without executing the selected action.
        self.assertIsInstance(module.body[-1], ast.Expr)
        self.assertEqual(module.body[-1].value.func.id, "run_action")
        module.body.pop()
        self.scope = {}
        exec(compile(module, "Forge_Colab.ipynb", "exec"), self.scope)
        self.scope["storage"] = self.storage
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(sys, "platform", "linux"))
        stack.enter_context(mock.patch("os.path.ismount", return_value=True))
        self.origin = stack.enter_context(mock.patch("subprocess.check_output", return_value=REMOTE + "\n"))

    def test_notebook_has_no_saved_outputs_and_only_one_action_dispatch(self):
        self.assertEqual(len(CODE), 2)
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        for cell in NOTEBOOK["cells"]:
            self.assertNotRegex("".join(cell["source"]), "[\u3400-\u9fff]")
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        self.assertEqual(self.scope["action"], "Start Forge")
        calls = [node for node in ast.walk(ast.parse(CODE[1]))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "run_action"]
        self.assertEqual(len(calls), 1)

    def test_each_menu_choice_runs_only_its_own_action(self):
        for selected in ("Start Forge", "Install environments", "Download models", "Create Qwen NF4", "Create Qwen INT8"):
            with self.subTest(action=selected), mock.patch("runpy.run_path") as launcher, mock.patch("subprocess.run") as command, mock.patch("getpass.getpass", return_value="test-only-key"):
                self.scope["run_action"](selected)
                self.assertEqual(launcher.call_count + command.call_count, 1)
                if selected in {"Start Forge", "Install environments"}:
                    launcher.assert_called_once_with(str(self.repo / "colab/start.py"), run_name="__main__")
                else:
                    self.assertTrue(command.call_args.kwargs["check"])
                    self.assertNotIn("shell", command.call_args.kwargs)
                if selected.startswith("Create Qwen"):
                    args = command.call_args.args[0]
                    precision = selected.rsplit(" ", 1)[1].lower()
                    self.assertEqual(args[0], str(self.repo / "runtimes/qwen-image-2.1/bin/python"))
                    self.assertEqual(args[args.index("--precision") + 1], precision)
                    self.assertEqual(args[args.index("--source") + 1], str(self.storage / "models/diffusers/Qwen-Image-2.1"))
                    self.assertEqual(args[args.index("--output") + 1], str(self.storage / ("models/diffusers/Qwen-Image-2.1-" + precision.upper())))

    def test_launcher_arguments_are_restored_even_when_the_launcher_fails(self):
        before = sys.argv
        seen = []

        def failure(*args, **kwargs):
            seen.append(list(sys.argv))
            raise SystemExit(1)

        with mock.patch("runpy.run_path", side_effect=failure):
            with self.assertRaises(SystemExit):
                self.scope["run_action"]("Install environments")
        self.assertIs(sys.argv, before)
        self.assertEqual(seen, [[str(self.repo / "colab/start.py"), "--drive-root", str(self.storage), "--install-only"]])

    def test_failed_quantization_stops_without_launching_forge(self):
        failure = subprocess.CalledProcessError(1, ["quantization fixture"])
        with mock.patch("subprocess.run", side_effect=failure) as command, mock.patch("runpy.run_path") as launcher:
            with self.assertRaises(subprocess.CalledProcessError):
                self.scope["run_action"]("Create Qwen NF4")
        command.assert_called_once()
        launcher.assert_not_called()

    def test_download_key_is_not_an_argument_or_persisted_in_kernel_environment(self):
        observed = []

        def download(args, **kwargs):
            self.assertEqual(kwargs["env"]["CIVITAI_API_KEY"], "test-only-key")
            self.assertFalse(any("test-only-key" in item for item in args))
            self.assertEqual(os.environ.get("CIVITAI_API_KEY"), "existing-parent-value")
            observed.append(kwargs["env"])
            raise subprocess.CalledProcessError(1, args)

        with mock.patch.dict(os.environ, {"CIVITAI_API_KEY": "existing-parent-value"}), mock.patch("getpass.getpass", return_value="test-only-key"), mock.patch("subprocess.run", side_effect=download):
            with self.assertRaises(subprocess.CalledProcessError):
                self.scope["run_action"]("Download models")
            self.assertEqual(os.environ["CIVITAI_API_KEY"], "existing-parent-value")
        self.assertNotIn("CIVITAI_API_KEY", observed[0])

    def test_missing_install_or_wrong_remote_is_preserved_and_not_started(self):
        (self.repo / ".colab/installed.json").unlink()
        with mock.patch("runpy.run_path") as launcher:
            with self.assertRaisesRegex(RuntimeError, "Finish Install"):
                self.scope["run_action"]("Start Forge")
            self.origin.return_value = "https://github.com/other/project.git"
            with self.assertRaisesRegex(RuntimeError, "another repository"):
                self.scope["run_action"]("Install environments")
        launcher.assert_not_called()
        self.assertTrue((self.repo / ".git").is_dir())

    def test_unmounted_drive_never_creates_a_local_fallback_install(self):
        with mock.patch("os.path.ismount", return_value=False), mock.patch("subprocess.run") as command, mock.patch("runpy.run_path") as launcher:
            with self.assertRaisesRegex(RuntimeError, "Connect Google Drive"):
                self.scope["run_action"]("Install environments")
        command.assert_not_called()
        launcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
