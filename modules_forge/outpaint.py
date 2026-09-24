"""Shared geometry and processing helpers for Forge Outpaint."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin


DEFAULT_ALIGNMENT = 64
DEFAULT_MAX_PIXELS = 268_435_456
INITIALIZATION_MODES = ("Edge", "Reflect", "Noise")
REGION_MARGINS = "Margins"
REGION_CANVAS_POSITION = "Canvas position"
FILL_MODES = INITIALIZATION_MODES

_EDGE_CONTEXT_DEPTH = 16
_EDGE_EARLY_DEPTH = 8
_EDGE_FLAT_RANGE = 6.0
_EDGE_CONTEXT_RANGE = 16.0
_EDGE_MINIMUM_JUMP = 12.0

Size = Tuple[int, int]
Box = Tuple[int, int, int, int]


def _strict_integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _pair(name: str, value: Sequence[int], minimum: int) -> Size:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must contain two integers.")

    return (
        _strict_integer(f"{name} width", value[0], minimum),
        _strict_integer(f"{name} height", value[1], minimum),
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class OutpaintGeometry:
    """Store the input image position and the canvas dimensions."""

    source_width: int
    source_height: int
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    alignment: int = DEFAULT_ALIGNMENT
    max_pixels: int = DEFAULT_MAX_PIXELS

    def __post_init__(self) -> None:
        attributes = (
            ("source_width", "source width", self.source_width, 1),
            ("source_height", "source height", self.source_height, 1),
            ("left", "left margin", self.left, 0),
            ("right", "right margin", self.right, 0),
            ("top", "top margin", self.top, 0),
            ("bottom", "bottom margin", self.bottom, 0),
            ("alignment", "alignment", self.alignment, 1),
            ("max_pixels", "maximum pixel count", self.max_pixels, 1),
        )
        for attribute, name, value, minimum in attributes:
            object.__setattr__(self, attribute, _strict_integer(name, value, minimum))

        if not any((self.left, self.right, self.top, self.bottom)):
            raise ValueError("The outpaint area is empty.")

        if self.work_pixels > self.max_pixels:
            raise ValueError("The working canvas has too many pixels.")

    @classmethod
    def from_margins(
        cls,
        source_size: Sequence[int],
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
        alignment: int = DEFAULT_ALIGNMENT,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> "OutpaintGeometry":
        source_width, source_height = _pair("source dimensions", source_size, 1)
        return cls(
            source_width=source_width,
            source_height=source_height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            alignment=alignment,
            max_pixels=max_pixels,
        )

    @classmethod
    def from_canvas(
        cls,
        source_size: Sequence[int],
        *,
        canvas_size: Sequence[int],
        source_position: Sequence[int],
        alignment: int = DEFAULT_ALIGNMENT,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> "OutpaintGeometry":
        source_width, source_height = _pair("source dimensions", source_size, 1)
        canvas_width, canvas_height = _pair("canvas dimensions", canvas_size, 1)
        source_x, source_y = _pair("source position", source_position, 0)

        if source_x + source_width > canvas_width or source_y + source_height > canvas_height:
            raise ValueError("The input image is outside the canvas.")

        return cls(
            source_width=source_width,
            source_height=source_height,
            left=source_x,
            right=canvas_width - source_x - source_width,
            top=source_y,
            bottom=canvas_height - source_y - source_height,
            alignment=alignment,
            max_pixels=max_pixels,
        )

    @classmethod
    def from_canvas_position(
        cls,
        source_size: Sequence[int],
        *,
        canvas_size: Sequence[int],
        source_position: Sequence[int],
        alignment: int = DEFAULT_ALIGNMENT,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> "OutpaintGeometry":
        return cls.from_canvas(
            source_size,
            canvas_size=canvas_size,
            source_position=source_position,
            alignment=alignment,
            max_pixels=max_pixels,
        )

    @property
    def source_size(self) -> Size:
        return self.source_width, self.source_height

    @property
    def source_position(self) -> Size:
        return self.left, self.top

    @property
    def source_x(self) -> int:
        return self.left

    @property
    def source_y(self) -> int:
        return self.top

    @property
    def source_box(self) -> Box:
        return (
            self.source_x,
            self.source_y,
            self.source_x + self.source_width,
            self.source_y + self.source_height,
        )

    @property
    def canvas_width(self) -> int:
        return self.left + self.source_width + self.right

    @property
    def canvas_height(self) -> int:
        return self.top + self.source_height + self.bottom

    @property
    def canvas_size(self) -> Size:
        return self.canvas_width, self.canvas_height

    @property
    def logical_width(self) -> int:
        return self.canvas_width

    @property
    def logical_height(self) -> int:
        return self.canvas_height

    @property
    def logical_size(self) -> Size:
        return self.canvas_size

    @property
    def logical_box(self) -> Box:
        return 0, 0, self.canvas_width, self.canvas_height

    @property
    def work_width(self) -> int:
        return _align_up(self.canvas_width, self.alignment)

    @property
    def work_height(self) -> int:
        return _align_up(self.canvas_height, self.alignment)

    @property
    def work_size(self) -> Size:
        return self.work_width, self.work_height

    @property
    def work_box(self) -> Box:
        return 0, 0, self.work_width, self.work_height

    @property
    def pad_right(self) -> int:
        return self.work_width - self.canvas_width

    @property
    def pad_bottom(self) -> int:
        return self.work_height - self.canvas_height

    @property
    def work_padding(self) -> Tuple[int, int, int, int]:
        return 0, self.pad_right, 0, self.pad_bottom

    @property
    def logical_pixels(self) -> int:
        return self.canvas_width * self.canvas_height

    @property
    def work_pixels(self) -> int:
        return self.work_width * self.work_height


def normalize_source(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL image.")
    return ImageOps.exif_transpose(image).convert("RGBA")


def _source_for_geometry(source: Image.Image, geometry: OutpaintGeometry) -> Image.Image:
    if not isinstance(geometry, OutpaintGeometry):
        raise TypeError("Geometry must be an OutpaintGeometry object.")

    normalized = normalize_source(source)
    if normalized.size != geometry.source_size:
        raise ValueError("Input image dimensions do not agree with geometry.")
    return normalized


def _reflect_indices(length: int, source_start: int, source_length: int) -> np.ndarray:
    if source_length == 1:
        return np.zeros(length, dtype=np.int64)

    relative = np.arange(length, dtype=np.int64) - source_start
    period = 2 * (source_length - 1)
    folded = np.mod(relative, period)
    return np.where(folded < source_length, folded, period - folded)


def _visible_features(source_array: np.ndarray) -> np.ndarray:
    rgba = source_array.astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    return np.concatenate((rgba[:, :, :3] * alpha, rgba[:, :, 3:4]), axis=2)


def _edge_lines(features: np.ndarray, direction: str) -> np.ndarray:
    if direction == "top":
        return features[:_EDGE_CONTEXT_DEPTH]
    if direction == "bottom":
        return features[-_EDGE_CONTEXT_DEPTH:][::-1]
    if direction == "left":
        return features[:, :_EDGE_CONTEXT_DEPTH].transpose(1, 0, 2)
    if direction == "right":
        return features[:, -_EDGE_CONTEXT_DEPTH:].transpose(1, 0, 2)[::-1]
    raise ValueError("Direction must be top, bottom, left, or right.")


def _channel_range(values: np.ndarray) -> float:
    flattened = values.reshape(-1, values.shape[-1])
    low, high = np.percentile(flattened, (5, 95), axis=0)
    return float(np.max(high - low))


def _edge_requires_reflect(features: np.ndarray, direction: str) -> bool:
    lines = _edge_lines(features, direction)
    if lines.shape[0] < 2:
        return False

    edge_range = _channel_range(lines[0])
    context_range = _channel_range(lines)
    transition_count = min(_EDGE_EARLY_DEPTH, lines.shape[0] - 1)
    transitions = np.abs(lines[1 : transition_count + 1] - lines[:transition_count])
    maximum_jump = float(np.max(np.mean(transitions, axis=(1, 2))))

    return (
        edge_range <= _EDGE_FLAT_RANGE
        and context_range >= _EDGE_CONTEXT_RANGE
        and maximum_jump >= _EDGE_MINIMUM_JUMP
    )


def _edge_reflect_directions(
    source_array: np.ndarray,
    geometry: OutpaintGeometry,
) -> tuple[str, ...]:
    features = _visible_features(source_array)
    active_directions = (
        ("left", geometry.left),
        ("right", geometry.right),
        ("top", geometry.top),
        ("bottom", geometry.bottom),
    )
    return tuple(
        direction
        for direction, margin in active_directions
        if margin and _edge_requires_reflect(features, direction)
    )


def build_work_canvas(
    source: Image.Image,
    geometry: OutpaintGeometry,
    mode: str = "Edge",
    seed: int = 0,
) -> Image.Image:
    normalized = _source_for_geometry(source, geometry)
    if not isinstance(mode, str):
        raise TypeError("Initialization mode must be text.")

    canonical_modes = {item.casefold(): item for item in INITIALIZATION_MODES}
    canonical_mode = canonical_modes.get(mode.strip().casefold())
    if canonical_mode is None:
        raise ValueError("Initialization mode must be Edge, Reflect, or Noise.")

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("Seed must be an integer.")
    seed = int(seed)

    source_array = np.asarray(normalized, dtype=np.uint8)
    if canonical_mode == "Noise":
        random = np.random.default_rng(seed % (1 << 64))
        output = np.empty((geometry.work_height, geometry.work_width, 4), dtype=np.uint8)
        output[:, :, :3] = random.integers(
            0,
            256,
            size=(geometry.work_height, geometry.work_width, 3),
            dtype=np.uint8,
        )
        output[:, :, 3] = 255
    else:
        if canonical_mode == "Edge":
            x_indices = np.clip(
                np.arange(geometry.work_width, dtype=np.int64) - geometry.source_x,
                0,
                geometry.source_width - 1,
            )
            y_indices = np.clip(
                np.arange(geometry.work_height, dtype=np.int64) - geometry.source_y,
                0,
                geometry.source_height - 1,
            )

            reflect_directions = _edge_reflect_directions(source_array, geometry)
            if "left" in reflect_directions or "right" in reflect_directions:
                reflected_x = _reflect_indices(
                    geometry.work_width,
                    geometry.source_x,
                    geometry.source_width,
                )
                if "left" in reflect_directions:
                    x_indices[: geometry.source_x] = reflected_x[: geometry.source_x]
                if "right" in reflect_directions:
                    source_end = geometry.source_x + geometry.source_width
                    x_indices[source_end:] = reflected_x[source_end:]

            if "top" in reflect_directions or "bottom" in reflect_directions:
                reflected_y = _reflect_indices(
                    geometry.work_height,
                    geometry.source_y,
                    geometry.source_height,
                )
                if "top" in reflect_directions:
                    y_indices[: geometry.source_y] = reflected_y[: geometry.source_y]
                if "bottom" in reflect_directions:
                    source_end = geometry.source_y + geometry.source_height
                    y_indices[source_end:] = reflected_y[source_end:]
        else:
            x_indices = _reflect_indices(
                geometry.work_width,
                geometry.source_x,
                geometry.source_width,
            )
            y_indices = _reflect_indices(
                geometry.work_height,
                geometry.source_y,
                geometry.source_height,
            )
        output = source_array[y_indices[:, None], x_indices[None, :]].copy()

    canvas = Image.fromarray(output)
    canvas.paste(normalized, geometry.source_position)
    return canvas


def build_generation_mask(
    geometry: OutpaintGeometry,
    context_overlap: int = 32,
) -> Image.Image:
    if not isinstance(geometry, OutpaintGeometry):
        raise TypeError("Geometry must be an OutpaintGeometry object.")
    overlap = _strict_integer("context overlap", context_overlap, 0)

    mask = Image.new("L", geometry.work_size, 255)
    mask.paste(0, geometry.source_box)
    x0, y0, x1, y1 = geometry.source_box

    if geometry.left:
        mask.paste(255, (x0, y0, min(x0 + overlap, x1), y1))
    if geometry.right:
        mask.paste(255, (max(x1 - overlap, x0), y0, x1, y1))
    if geometry.top:
        mask.paste(255, (x0, y0, x1, min(y0 + overlap, y1)))
    if geometry.bottom:
        mask.paste(255, (x0, max(y1 - overlap, y0), x1, y1))
    return mask


def _generated_rgba(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Generated image must be a PIL image.")
    return ImageOps.exif_transpose(image).convert("RGBA")


def restore_input_exact(
    generated: Image.Image,
    source: Image.Image,
    geometry: OutpaintGeometry,
) -> Image.Image:
    normalized_source = _source_for_geometry(source, geometry)
    normalized_generated = _generated_rgba(generated)
    if normalized_generated.size == geometry.work_size:
        output = normalized_generated.crop(geometry.logical_box)
    elif normalized_generated.size == geometry.canvas_size:
        output = normalized_generated.copy()
    else:
        raise ValueError(
            "Generated image dimensions do not agree with the working or output canvas."
        )

    output.paste(normalized_source, geometry.source_position)
    assert_input_unchanged(output, normalized_source, geometry)
    return output


def assert_input_unchanged(
    output: Image.Image,
    source: Image.Image,
    geometry: OutpaintGeometry,
) -> None:
    normalized_source = _source_for_geometry(source, geometry)
    normalized_output = _generated_rgba(output)
    if normalized_output.size != geometry.canvas_size:
        raise ValueError("Output image dimensions do not agree with the canvas.")

    restored_source = normalized_output.crop(geometry.source_box)
    if restored_source.tobytes() != normalized_source.tobytes():
        raise AssertionError("Pixels in the input image changed.")


def source_pixel_sha256(source: Image.Image) -> str:
    return hashlib.sha256(normalize_source(source).tobytes()).hexdigest()


def _input_integer(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        result = int(value.strip())
    else:
        raise ValueError(f"{name} must be an integer.")

    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be {minimum} or more.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be {maximum} or less.")
    return result


def _region_mode(value):
    normalized = str(value or REGION_MARGINS).strip().casefold()
    if normalized == REGION_MARGINS.casefold():
        return REGION_MARGINS
    if normalized == REGION_CANVAS_POSITION.casefold():
        return REGION_CANVAS_POSITION
    raise ValueError("Region mode must be Margins or Canvas position.")


def _fill_mode(value):
    normalized = str(value or "Edge").strip().casefold()
    choices = {item.casefold(): item for item in FILL_MODES}
    if normalized not in choices:
        raise ValueError("Fill mode must be Edge, Reflect, or Noise.")
    return choices[normalized]


def _megapixels(value):
    if isinstance(value, bool):
        raise ValueError("Maximum megapixels must be a finite positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Maximum megapixels must be a finite positive number.") from error
    if not math.isfinite(result) or result < 0.1 or result > 64:
        raise ValueError("Maximum megapixels must be from 0.1 through 64.")
    return result


def _metadata_value(values, index, fallback):
    if values is not None and index < len(values):
        return values[index]
    return fallback


def _logical_infotext(information, geometry):
    if not isinstance(information, str):
        return information

    head, separator, parameters = information.rpartition("\n")
    pattern = (
        rf"(?P<prefix>(?:^|, )Size: ){geometry.work_width}x{geometry.work_height}"
        r"(?=,|$)"
    )
    parameters = re.sub(
        pattern,
        rf"\g<prefix>{geometry.canvas_width}x{geometry.canvas_height}",
        parameters,
        count=1,
    )
    return f"{head}{separator}{parameters}"


def _png_base64(image):
    metadata = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)

    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG", pnginfo=metadata)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _sample_images(
    proc,
    expected_count: int,
    *,
    return_mask: bool,
    return_mask_composite: bool,
):
    stride = 1 + int(return_mask) + int(return_mask_composite)
    first = int(proc.index_of_first_image or 0)
    information_count = len(proc.infotexts or [])
    count = min(expected_count, information_count)

    result = []
    for index in range(count):
        image_index = first + index * stride
        if image_index >= len(proc.images):
            break
        result.append((index, proc.images[image_index]))
    return result


def _check_saved_png(path, source, geometry):
    full_path = os.path.abspath(path)
    try:
        with Image.open(full_path) as saved_image:
            if saved_image.format != "PNG":
                raise ValueError("The file format is not PNG.")
            saved_image.load()
            persisted = saved_image.convert("RGBA")
        assert_input_unchanged(persisted, source, geometry)
    except Exception as error:
        raise RuntimeError(
            "Forge Outpaint cannot make sure that the PNG has unchanged input "
            f'pixels: "{full_path}". The file is still at this location. {error}'
        ) from error


def run_outpaint(
    p,
    region_mode=REGION_MARGINS,
    left=128,
    right=128,
    top=128,
    bottom=128,
    canvas_width=1024,
    canvas_height=1024,
    source_x=0,
    source_y=0,
    context_overlap=32,
    mask_blur=8,
    fill_mode="Edge",
    max_megapixels=4.0,
    *,
    invoke: Callable[[], object] | None = None,
):
    """Run Outpaint around a zero-argument processing callback."""

    from modules import images
    from modules.processing import fix_seed, process_images
    from modules.shared import opts, state

    if not p.init_images or not isinstance(p.init_images[0], Image.Image):
        raise ValueError("Select an input image for Forge Outpaint.")

    source = normalize_source(p.init_images[0])
    for additional_source in p.init_images[1:]:
        if not isinstance(additional_source, Image.Image):
            raise ValueError("All Forge Outpaint input images must be PIL images.")
        normalized_additional = normalize_source(additional_source)
        if (
            normalized_additional.size != source.size
            or normalized_additional.tobytes() != source.tobytes()
        ):
            raise ValueError("Forge Outpaint accepts one distinct input image.")

    selected_region_mode = _region_mode(region_mode)
    selected_fill_mode = _fill_mode(fill_mode)
    if selected_region_mode == REGION_MARGINS:
        left = _input_integer("Left margin", left, minimum=0)
        right = _input_integer("Right margin", right, minimum=0)
        top = _input_integer("Top margin", top, minimum=0)
        bottom = _input_integer("Bottom margin", bottom, minimum=0)
    else:
        canvas_width = _input_integer("Canvas width", canvas_width, minimum=1)
        canvas_height = _input_integer("Canvas height", canvas_height, minimum=1)
        source_x = _input_integer("Input X position", source_x, minimum=0)
        source_y = _input_integer("Input Y position", source_y, minimum=0)
    context_overlap = _input_integer("Context overlap", context_overlap, minimum=0)
    mask_blur = _input_integer("Mask blur", mask_blur, minimum=0, maximum=256)
    max_megapixels = _megapixels(max_megapixels)

    maximum_pixels = int(max_megapixels * 1_000_000)
    try:
        if selected_region_mode == REGION_MARGINS:
            geometry = OutpaintGeometry.from_margins(
                source.size,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                alignment=64,
                max_pixels=maximum_pixels,
            )
        else:
            geometry = OutpaintGeometry.from_canvas(
                source.size,
                canvas_size=(canvas_width, canvas_height),
                source_position=(source_x, source_y),
                alignment=64,
                max_pixels=maximum_pixels,
            )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Forge Outpaint cannot use this image region. {error}") from error

    original_do_not_save_samples = p.do_not_save_samples
    original_do_not_save_grid = p.do_not_save_grid
    save_samples_requested = (
        bool(p.override_settings.get("samples_save", opts.samples_save))
        and not original_do_not_save_samples
    )
    save_incomplete_images = bool(
        p.override_settings.get("save_incomplete_images", opts.save_incomplete_images)
    )
    return_mask = bool(p.override_settings.get("return_mask", opts.return_mask))
    return_mask_composite = bool(
        p.override_settings.get("return_mask_composite", opts.return_mask_composite)
    )
    save_to_dirs = bool(p.override_settings.get("save_to_dirs", opts.save_to_dirs))
    fix_seed(p)
    seed = p.seed[0] if isinstance(p.seed, (list, tuple)) else p.seed
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("Seed must be an integer.")

    try:
        work_canvas = build_work_canvas(
            source,
            geometry,
            mode=selected_fill_mode,
            seed=int(seed),
        )
        generation_mask = build_generation_mask(
            geometry,
            context_overlap=context_overlap,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Forge Outpaint cannot prepare the input image. {error}") from error

    restore_overrides_afterwards = p.override_settings_restore_afterwards
    original_save_init_option = opts.save_init_img
    had_save_init_override = "save_init_img" in p.override_settings
    original_save_init_override = p.override_settings.get("save_init_img")
    original_processing_values = {
        "init_images": p.init_images,
        "image_mask": p.image_mask,
        "latent_mask": p.latent_mask,
        "inpaint_full_res": p.inpaint_full_res,
        "inpainting_fill": p.inpainting_fill,
        "inpainting_mask_invert": p.inpainting_mask_invert,
        "mask_round": p.mask_round,
        "mask_blur_x": p.mask_blur_x,
        "mask_blur_y": p.mask_blur_y,
        "width": p.width,
        "height": p.height,
    }

    p.init_images = [work_canvas]
    p.image_mask = generation_mask
    p.latent_mask = None
    p.inpaint_full_res = False
    p.inpainting_fill = 1
    p.inpainting_mask_invert = 0
    p.mask_round = False
    p.mask_blur = mask_blur
    p.width, p.height = geometry.work_size
    p.do_not_save_samples = True
    p.do_not_save_grid = True
    p.override_settings["save_init_img"] = False

    source_hash = source_pixel_sha256(source)
    generation_params = {
        "Outpaint": "Forge Outpaint",
        "Outpaint region mode": selected_region_mode,
        "Outpaint source size": f"{geometry.source_width}x{geometry.source_height}",
        "Outpaint canvas size": f"{geometry.canvas_width}x{geometry.canvas_height}",
        "Outpaint working size": f"{geometry.work_width}x{geometry.work_height}",
        "Outpaint source position": f"{geometry.source_x},{geometry.source_y}",
        "Outpaint margins": (
            f"{geometry.left},{geometry.right},{geometry.top},{geometry.bottom}"
        ),
        "Outpaint context overlap": context_overlap,
        "Outpaint mask blur": mask_blur,
        "Outpaint fill mode": selected_fill_mode,
        "Outpaint maximum megapixels": max_megapixels,
        "Outpaint source SHA-256": source_hash,
        "Outpaint output format": "PNG",
    }
    if selected_fill_mode == "Edge":
        fallback_directions = _edge_reflect_directions(np.asarray(source), geometry)
        if fallback_directions:
            generation_params["Outpaint Edge fallback"] = ",".join(
                direction.title() for direction in fallback_directions
            )
    p.extra_generation_params.update(generation_params)

    try:
        proc = process_images(p) if invoke is None else invoke()
        if proc is None:
            raise RuntimeError("Forge Outpaint processing did not return a result.")

        expected_count = int(p.n_iter) * int(p.batch_size)
        protected_images = []
        should_save_samples = save_samples_requested and (
            save_incomplete_images or not state.interrupted and not state.skipped
        )

        for metadata_index, generated in _sample_images(
            proc,
            expected_count,
            return_mask=return_mask,
            return_mask_composite=return_mask_composite,
        ):
            try:
                protected = restore_input_exact(generated, source, geometry)
                assert_input_unchanged(protected, source, geometry)
            except Exception as error:
                raise RuntimeError(
                    "Forge Outpaint cannot restore the input pixels in output image "
                    f"{metadata_index + 1}. {error}"
                ) from error

            information = _logical_infotext(
                _metadata_value(proc.infotexts, metadata_index, proc.info),
                geometry,
            )
            if metadata_index < len(proc.infotexts):
                proc.infotexts[metadata_index] = information
            if isinstance(information, str):
                protected.info["parameters"] = information

            protected_images.append(protected)

            if should_save_samples:
                seed_value = _metadata_value(proc.all_seeds, metadata_index, proc.seed)
                prompt_value = _metadata_value(
                    proc.all_prompts, metadata_index, proc.prompt
                )
                saved_path, _ = images.save_image(
                    protected.copy(),
                    p.outpath_samples,
                    "",
                    seed_value,
                    prompt_value,
                    extension="png",
                    info=information,
                    p=p,
                    save_to_dirs=save_to_dirs,
                    skip_stealth_pnginfo=True,
                    finalize_image=lambda candidate: restore_input_exact(
                        candidate,
                        source,
                        geometry,
                    ),
                )
                _check_saved_png(saved_path, source, geometry)

        proc.images = (
            [_png_base64(image) for image in protected_images]
            if p.is_api
            else protected_images
        )
        proc.infotexts = proc.infotexts[: len(protected_images)]
        if proc.infotexts:
            proc.info = proc.infotexts[0]
        proc.width = geometry.canvas_width
        proc.height = geometry.canvas_height
        proc.index_of_first_image = 0
        proc.extra_images = []
        return proc
    finally:
        p.do_not_save_samples = original_do_not_save_samples
        p.do_not_save_grid = original_do_not_save_grid
        for name, value in original_processing_values.items():
            setattr(p, name, value)
        if had_save_init_override:
            p.override_settings["save_init_img"] = original_save_init_override
        else:
            p.override_settings.pop("save_init_img", None)
        if not restore_overrides_afterwards:
            desired_save_init_option = (
                original_save_init_override
                if had_save_init_override
                else original_save_init_option
            )
            if opts.save_init_img != desired_save_init_option:
                opts.set(
                    "save_init_img",
                    desired_save_init_option,
                    is_api=True,
                    run_callbacks=False,
                )


__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_MAX_PIXELS",
    "INITIALIZATION_MODES",
    "REGION_MARGINS",
    "REGION_CANVAS_POSITION",
    "FILL_MODES",
    "OutpaintGeometry",
    "assert_input_unchanged",
    "build_generation_mask",
    "build_work_canvas",
    "normalize_source",
    "restore_input_exact",
    "run_outpaint",
    "source_pixel_sha256",
]
