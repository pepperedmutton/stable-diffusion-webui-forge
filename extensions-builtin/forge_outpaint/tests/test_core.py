from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

from PIL import Image


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT))

from forge_outpaint import (  # noqa: E402
    OutpaintGeometry,
    assert_input_unchanged,
    build_generation_mask,
    build_work_canvas,
    normalize_source,
    restore_input_exact,
    source_pixel_sha256,
)


def rgba_grid(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 29 % 256, y * 47 % 256, (x + y) * 61 % 256, (x * 17 + y * 13) % 256)
    return image


class GeometryTests(unittest.TestCase):
    def test_asymmetric_odd_geometry(self) -> None:
        geometry = OutpaintGeometry.from_margins(
            (101, 77),
            left=13,
            right=22,
            top=7,
            bottom=18,
        )

        self.assertEqual((136, 102), geometry.canvas_size)
        self.assertEqual((192, 128), geometry.work_size)
        self.assertEqual((13, 7), geometry.source_position)
        self.assertEqual((13, 7, 114, 84), geometry.source_box)
        self.assertEqual((0, 56, 0, 26), geometry.work_padding)
        self.assertEqual(136 * 102, geometry.logical_pixels)
        self.assertEqual(192 * 128, geometry.work_pixels)

    def test_canvas_position_geometry(self) -> None:
        geometry = OutpaintGeometry.from_canvas_position(
            (101, 77),
            canvas_size=(136, 102),
            source_position=(13, 7),
        )

        self.assertEqual((13, 22, 7, 18), (geometry.left, geometry.right, geometry.top, geometry.bottom))
        self.assertEqual((136, 102), geometry.logical_size)

    def test_invalid_areas(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            OutpaintGeometry.from_margins((10, 10))
        with self.assertRaises(ValueError):
            OutpaintGeometry.from_margins((10, 10), left=-1)
        with self.assertRaisesRegex(ValueError, "outside"):
            OutpaintGeometry.from_canvas(
                (10, 10),
                canvas_size=(12, 12),
                source_position=(3, 0),
            )
        with self.assertRaises(ValueError):
            OutpaintGeometry.from_canvas(
                (10, 10),
                canvas_size=(12, 12),
                source_position=(-1, 0),
            )

    def test_aligned_pixels_enforce_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "too many pixels"):
            OutpaintGeometry.from_margins(
                (64, 64),
                right=1,
                bottom=1,
                max_pixels=16_000,
            )


class ImageTests(unittest.TestCase):
    def test_normalize_rgb_and_rgba(self) -> None:
        rgb = Image.new("RGB", (3, 2), (11, 22, 33))
        normalized_rgb = normalize_source(rgb)
        self.assertEqual("RGBA", normalized_rgb.mode)
        self.assertEqual((11, 22, 33, 255), normalized_rgb.getpixel((1, 1)))

        rgba = rgba_grid(3, 2)
        normalized_rgba = normalize_source(rgba)
        self.assertEqual(rgba.tobytes(), normalized_rgba.tobytes())

    def test_normalize_applies_exif_orientation(self) -> None:
        source = Image.new("RGB", (2, 3), (0, 0, 0))
        source.putpixel((0, 0), (255, 0, 0))
        source.getexif()[274] = 6

        normalized = normalize_source(source)

        self.assertEqual((3, 2), normalized.size)
        self.assertEqual((255, 0, 0, 255), normalized.getpixel((2, 0)))

    def test_edge_canvas_extends_border_pixels(self) -> None:
        source = Image.new("RGBA", (2, 2))
        source.putdata(
            [
                (1, 2, 3, 4),
                (5, 6, 7, 8),
                (9, 10, 11, 12),
                (13, 14, 15, 16),
            ]
        )
        geometry = OutpaintGeometry.from_margins(
            source.size,
            left=1,
            right=1,
            top=1,
            bottom=1,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "edge")

        self.assertEqual((1, 2, 3, 4), canvas.getpixel((0, 0)))
        self.assertEqual((5, 6, 7, 8), canvas.getpixel((3, 0)))
        self.assertEqual((9, 10, 11, 12), canvas.getpixel((0, 3)))
        self.assertEqual((13, 14, 15, 16), canvas.getpixel((3, 3)))
        self.assertEqual(source.tobytes(), canvas.crop(geometry.source_box).tobytes())

    def test_edge_top_only_avoids_a_flat_white_initialization(self) -> None:
        source = Image.new("RGBA", (6, 4), (250, 250, 250, 255))
        for y in range(2, source.height):
            for x in range(source.width):
                source.putpixel(
                    (x, y),
                    (20 + x * 17, 30 + y * 23, 40 + (x + y) * 11, 255),
                )
        geometry = OutpaintGeometry.from_margins(
            source.size,
            top=4,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "Edge", seed=12345)
        top_region = canvas.crop((0, 0, geometry.canvas_width, geometry.top))

        self.assertEqual((0, 0, 4, 0), (geometry.left, geometry.right, geometry.top, geometry.bottom))
        self.assertEqual(source.width, canvas.width)
        self.assertEqual(source.height + geometry.top, canvas.height)
        self.assertNotEqual(
            Image.new("RGBA", top_region.size, (250, 250, 250, 255)).tobytes(),
            top_region.tobytes(),
        )
        top_bytes = top_region.tobytes()
        top_pixels = {top_bytes[index : index + 4] for index in range(0, len(top_bytes), 4)}
        self.assertGreater(len(top_pixels), 1)
        self.assertEqual(source.tobytes(), canvas.crop(geometry.source_box).tobytes())

        overwritten = canvas.copy()
        overwritten.paste((255, 0, 255, 0), geometry.source_box)
        restored = restore_input_exact(overwritten, source, geometry)
        self.assertEqual(source.tobytes(), restored.crop(geometry.source_box).tobytes())
        assert_input_unchanged(restored, source, geometry)

    def test_edge_top_only_keeps_nonuniform_border_behavior(self) -> None:
        source = Image.new("RGBA", (4, 3), (9, 10, 11, 255))
        top_row = [
            (10, 20, 30, 255),
            (40, 50, 60, 255),
            (70, 80, 90, 255),
            (100, 110, 120, 255),
        ]
        for x, pixel in enumerate(top_row):
            source.putpixel((x, 0), pixel)
        geometry = OutpaintGeometry.from_margins(
            source.size,
            top=3,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "edge", seed=12345)

        for y in range(geometry.top):
            self.assertEqual(top_row, [canvas.getpixel((x, y)) for x in range(source.width)])
        self.assertEqual(source.tobytes(), canvas.crop(geometry.source_box).tobytes())

    def test_edge_fallback_applies_to_low_information_top_only(self) -> None:
        source = Image.new("RGBA", (20, 20))
        for y in range(source.height):
            for x in range(source.width):
                source.putpixel(
                    (x, y),
                    (
                        250 if y == 0 else (x * 31 + y * 17) % 240,
                        250 if y == 0 else (x * 13 + y * 37) % 240,
                        250 if y == 0 else (x * 43 + y * 11) % 240,
                        (x * 19 + y * 23) % 256,
                    ),
                )
        for x in range(source.width):
            source.putpixel((x, 0), (250, 250, 250, 255))
        geometry = OutpaintGeometry.from_margins(
            source.size,
            right=4,
            top=4,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "Edge", seed=12345)

        expected_top_rows = [
            [source.getpixel((x, source_y)) for x in range(source.width)]
            for source_y in (4, 3, 2, 1)
        ]
        actual_top_rows = [
            [canvas.getpixel((x, canvas_y)) for x in range(source.width)]
            for canvas_y in range(geometry.top)
        ]
        self.assertEqual(expected_top_rows, actual_top_rows)

        for source_y in range(source.height):
            expected_right_pixel = source.getpixel((source.width - 1, source_y))
            canvas_y = geometry.top + source_y
            self.assertEqual(
                [expected_right_pixel] * geometry.right,
                [
                    canvas.getpixel((canvas_x, canvas_y))
                    for canvas_x in range(source.width, geometry.canvas_width)
                ],
            )
        self.assertEqual(source.tobytes(), canvas.crop(geometry.source_box).tobytes())

    def test_edge_fallback_treats_transparent_top_as_low_information(self) -> None:
        source = Image.new("RGBA", (18, 18))
        for y in range(source.height):
            for x in range(source.width):
                if y == 0:
                    source.putpixel(
                        (x, y),
                        ((x * 71) % 256, (x * 43) % 256, (x * 29) % 256, 0),
                    )
                else:
                    source.putpixel(
                        (x, y),
                        ((x * 17 + y * 41) % 256, (x * 47 + y * 13) % 256, (x * 23 + y * 31) % 256, 255),
                    )
        geometry = OutpaintGeometry.from_margins(
            source.size,
            top=3,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "Edge", seed=12345)

        self.assertEqual(
            [
                [source.getpixel((x, source_y)) for x in range(source.width)]
                for source_y in (3, 2, 1)
            ],
            [
                [canvas.getpixel((x, canvas_y)) for x in range(source.width)]
                for canvas_y in range(geometry.top)
            ],
        )
        self.assertEqual(source.tobytes(), canvas.crop(geometry.source_box).tobytes())

        overwritten = canvas.copy()
        overwritten.paste((255, 0, 255, 255), geometry.source_box)
        restored = restore_input_exact(overwritten, source, geometry)
        self.assertEqual(source.tobytes(), restored.crop(geometry.source_box).tobytes())
        assert_input_unchanged(restored, source, geometry)

    def test_reflect_canvas_uses_mirrored_pixels(self) -> None:
        source = Image.new("RGBA", (3, 1))
        source.putdata([(10, 0, 0, 255), (20, 0, 0, 255), (30, 0, 0, 255)])
        geometry = OutpaintGeometry.from_margins(
            source.size,
            left=2,
            right=2,
            alignment=1,
        )

        canvas = build_work_canvas(source, geometry, "Reflect")

        self.assertEqual(
            [30, 20, 10, 20, 30, 20, 10],
            [canvas.getpixel((x, 0))[0] for x in range(canvas.width)],
        )

    def test_noise_is_deterministic_and_preserves_source(self) -> None:
        source = rgba_grid(5, 3)
        geometry = OutpaintGeometry.from_margins(
            source.size,
            left=2,
            right=3,
            top=1,
            bottom=2,
            alignment=8,
        )

        first = build_work_canvas(source, geometry, "Noise", seed=12345)
        second = build_work_canvas(source, geometry, "noise", seed=12345)
        different = build_work_canvas(source, geometry, "Noise", seed=12346)

        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertNotEqual(first.tobytes(), different.tobytes())
        self.assertEqual(source.tobytes(), first.crop(geometry.source_box).tobytes())

    def test_generation_mask_has_selective_context_and_padding(self) -> None:
        geometry = OutpaintGeometry.from_margins(
            (5, 4),
            left=2,
            bottom=1,
            alignment=8,
        )

        mask = build_generation_mask(geometry, context_overlap=2)

        self.assertEqual("L", mask.mode)
        self.assertEqual((8, 8), mask.size)
        self.assertEqual(255, mask.getpixel((0, 0)))
        self.assertEqual(255, mask.getpixel((2, 0)))
        self.assertEqual(0, mask.getpixel((4, 0)))
        self.assertEqual(0, mask.getpixel((6, 0)))
        self.assertEqual(255, mask.getpixel((6, 2)))
        self.assertEqual(255, mask.getpixel((7, 0)))
        self.assertEqual(255, mask.getpixel((4, 7)))

    def test_restore_keeps_exact_rgba_pixels_and_logical_size(self) -> None:
        source = rgba_grid(5, 3)
        geometry = OutpaintGeometry.from_margins(
            source.size,
            left=3,
            right=1,
            top=2,
            bottom=4,
            alignment=8,
        )
        generated = Image.new("RGB", geometry.work_size, (201, 202, 203))

        output = restore_input_exact(generated, source, geometry)

        self.assertEqual("RGBA", output.mode)
        self.assertEqual(geometry.canvas_size, output.size)
        self.assertEqual(source.tobytes(), output.crop(geometry.source_box).tobytes())
        self.assertEqual((201, 202, 203, 255), output.getpixel((0, 0)))
        assert_input_unchanged(output, source, geometry)

        output.putpixel(geometry.source_position, (255, 255, 255, 255))
        with self.assertRaisesRegex(AssertionError, "input image changed"):
            assert_input_unchanged(output, source, geometry)

    def test_restore_rgb_input_uses_opaque_alpha(self) -> None:
        source = Image.new("RGB", (2, 2), (7, 8, 9))
        geometry = OutpaintGeometry.from_margins(source.size, right=1, alignment=1)
        generated = Image.new("RGBA", geometry.work_size, (0, 0, 0, 0))

        output = restore_input_exact(generated, source, geometry)

        self.assertEqual((7, 8, 9, 255), output.getpixel((0, 0)))

    def test_restore_protects_logical_image_after_save_callbacks(self) -> None:
        source = rgba_grid(4, 3)
        geometry = OutpaintGeometry.from_margins(
            source.size,
            left=2,
            right=1,
            top=1,
            alignment=8,
        )
        logical = Image.new("RGBA", geometry.canvas_size, (91, 92, 93, 255))
        logical.paste((255, 0, 255, 17), geometry.source_box)

        output = restore_input_exact(logical, source, geometry)

        self.assertEqual(geometry.canvas_size, output.size)
        self.assertEqual(source.tobytes(), output.crop(geometry.source_box).tobytes())
        self.assertEqual((91, 92, 93, 255), output.getpixel((0, 0)))

    def test_source_hash_uses_normalized_pixels(self) -> None:
        rgb = Image.new("RGB", (2, 1), (1, 2, 3))
        rgba = rgb.convert("RGBA")

        digest = source_pixel_sha256(rgb)

        self.assertEqual(hashlib.sha256(rgba.tobytes()).hexdigest(), digest)
        self.assertEqual(digest, source_pixel_sha256(rgba))

    def test_source_dimensions_must_agree_with_geometry(self) -> None:
        geometry = OutpaintGeometry.from_margins((3, 3), right=1, alignment=1)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            build_work_canvas(Image.new("RGB", (2, 3)), geometry)


if __name__ == "__main__":
    unittest.main()
