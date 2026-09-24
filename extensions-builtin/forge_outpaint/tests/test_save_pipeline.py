from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT))

from forge_outpaint import OutpaintGeometry, restore_input_exact  # noqa: E402


class ImageSaveParams:
    def __init__(self, image, p, filename, pnginfo):
        self.image = image
        self.p = p
        self.filename = filename
        self.pnginfo = pnginfo


def load_images_module(events, source_position, outside_position):
    options = types.SimpleNamespace(
        enable_pnginfo=True,
        jpeg_quality=90,
        webp_lossless=False,
        grid_save_to_dirs=False,
        save_to_dirs=False,
        directories_filename_pattern="",
        samples_filename_pattern="",
        save_images_add_number=False,
        save_images_replace_action="Replace",
        target_side_length=4096,
        export_for_4chan=False,
        img_downscale_threshold=4,
        save_txt=False,
        directories_max_prompt_words=8,
        data={"CLIP_stop_at_last_layers": 1},
    )

    def add_stealth_pnginfo(params):
        events.append("stealth")
        changed = params.image.copy()
        changed.putpixel(source_position, (11, 12, 13, 14))
        params.image = changed

    def before_image_saved_callback(params):
        events.append("before")
        changed = params.image.copy()
        changed.putpixel(source_position, (21, 22, 23, 24))
        changed.putpixel(outside_position, (31, 32, 33, 34))
        params.image = changed

    def image_saved_callback(params):
        events.append("saved")
        if not os.path.isfile(params.filename):
            raise AssertionError("The saved image does not exist.")

    modules_package = types.ModuleType("modules")
    modules_package.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.cmd_opts = types.SimpleNamespace(
        unix_filenames_sanitization=False,
        filenames_max_length=128,
    )
    shared.opts = options
    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.ImageSaveParams = ImageSaveParams
    script_callbacks.before_image_saved_callback = before_image_saved_callback
    script_callbacks.image_saved_callback = image_saved_callback
    stealth_infotext = types.ModuleType("modules.stealth_infotext")
    stealth_infotext.add_stealth_pnginfo = add_stealth_pnginfo
    stealth_infotext.read_info_from_image_stealth = lambda image: None
    errors = types.ModuleType("modules.errors")
    errors.display = lambda *args, **kwargs: None
    errors.report = lambda *args, **kwargs: None
    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.samplers_map = {}
    paths_internal = types.ModuleType("modules.paths_internal")
    paths_internal.roboto_ttf_file = ""

    modules_package.shared = shared
    modules_package.script_callbacks = script_callbacks
    modules_package.stealth_infotext = stealth_infotext
    modules_package.errors = errors
    modules_package.sd_samplers = sd_samplers

    piexif = types.ModuleType("piexif")
    piexif.__path__ = []
    piexif_helper = types.ModuleType("piexif.helper")
    piexif.helper = piexif_helper
    pillow_avif = types.ModuleType("pillow_avif")

    module_name = "_forge_outpaint_images_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / "modules" / "images.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("The image module cannot be loaded.")

    images_module = importlib.util.module_from_spec(spec)
    replacements = {
        module_name: images_module,
        "modules": modules_package,
        "modules.shared": shared,
        "modules.script_callbacks": script_callbacks,
        "modules.stealth_infotext": stealth_infotext,
        "modules.errors": errors,
        "modules.sd_samplers": sd_samplers,
        "modules.paths_internal": paths_internal,
        "piexif": piexif,
        "piexif.helper": piexif_helper,
        "pillow_avif": pillow_avif,
    }
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(images_module)

    return images_module


def transparent_source() -> Image.Image:
    source = Image.new("RGBA", (3, 2))
    source.putdata(
        [
            (1, 2, 3, 0),
            (4, 5, 6, 17),
            (7, 8, 9, 63),
            (10, 11, 12, 127),
            (13, 14, 15, 191),
            (16, 17, 18, 255),
        ]
    )
    return source


class SavePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = transparent_source()
        self.geometry = OutpaintGeometry.from_margins(
            self.source.size,
            left=2,
            right=1,
            top=1,
            bottom=1,
            alignment=1,
        )
        self.source_position = self.geometry.source_position
        self.outside_position = (0, 0)
        self.events = []
        self.images = load_images_module(
            self.events,
            self.source_position,
            self.outside_position,
        )

    def save(self, directory, *, skip_stealth_pnginfo):
        logical = Image.new("RGBA", self.geometry.canvas_size, (41, 42, 43, 44))
        logical.paste(self.source, self.geometry.source_position)

        def finalize(candidate):
            self.events.append("finalize")
            self.assertEqual((21, 22, 23, 24), candidate.getpixel(self.source_position))
            return restore_input_exact(candidate, self.source, self.geometry)

        return self.images.save_image(
            logical,
            directory,
            "",
            extension="png",
            info="test metadata",
            forced_filename="protected",
            skip_stealth_pnginfo=skip_stealth_pnginfo,
            finalize_image=finalize,
        )[0]

    def assert_saved_pixels(self, path) -> None:
        with Image.open(path) as saved:
            saved.load()
            rgba = saved.convert("RGBA")
            self.assertEqual("test metadata", saved.info.get("parameters"))

        restored = rgba.crop(self.geometry.source_box)
        self.assertEqual(self.source.tobytes(), restored.tobytes())
        self.assertEqual((31, 32, 33, 34), rgba.getpixel(self.outside_position))

    def test_finalizer_runs_after_pixel_changes_and_before_disk_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, skip_stealth_pnginfo=False)
            self.assert_saved_pixels(path)

        self.assertEqual(["stealth", "before", "finalize", "saved"], self.events)

    def test_skip_stealth_keeps_text_metadata_and_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(directory, skip_stealth_pnginfo=True)
            self.assert_saved_pixels(path)

        self.assertEqual(["before", "finalize", "saved"], self.events)


if __name__ == "__main__":
    unittest.main()
