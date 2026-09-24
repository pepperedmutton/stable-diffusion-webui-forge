"""Compatibility exports for the shared Forge Outpaint core."""

from modules_forge.outpaint import (
    DEFAULT_ALIGNMENT,
    DEFAULT_MAX_PIXELS,
    INITIALIZATION_MODES,
    OutpaintGeometry,
    assert_input_unchanged,
    build_generation_mask,
    build_work_canvas,
    normalize_source,
    restore_input_exact,
    source_pixel_sha256,
)

__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_MAX_PIXELS",
    "INITIALIZATION_MODES",
    "OutpaintGeometry",
    "assert_input_unchanged",
    "build_generation_mask",
    "build_work_canvas",
    "normalize_source",
    "restore_input_exact",
    "source_pixel_sha256",
]
