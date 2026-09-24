#!/usr/bin/env python3
"""Apply Colab mobile overrides to an explicitly selected Forge copy.

This does not add pinch-to-zoom or touch panning to the original ForgeCanvas.
Use --check to preview changes without writing. Repeated runs are idempotent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
import outpaint_safety_patch


CSS_START = "/* BEGIN FORGE COLAB MOBILE OVERRIDES */"
CSS_END = "/* END FORGE COLAB MOBILE OVERRIDES */"
JS_START = "// BEGIN FORGE MOBILE HIT TARGETS"
JS_END = "// END FORGE MOBILE HIT TARGETS"


def read_utf8(path: Path) -> str:
    # Decode bytes directly so unrelated CRLF line endings stay unchanged.
    return path.read_bytes().decode("utf-8")


def newline_for(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def replace_marked(text: str, start: str, end: str, block: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one complete patch block: {start}")
    begin = text.index(start)
    finish = text.index(end) + len(end)
    if finish <= begin:
        raise ValueError(f"Patch markers are out of order: {start}")
    return text[:begin] + block + text[finish:]


def patched_css(current: str, additions: str) -> str:
    nl = newline_for(current)
    additions = nl.join(additions.strip().splitlines())
    block = CSS_START + nl + additions + nl + CSS_END
    if CSS_START in current or CSS_END in current:
        return replace_marked(current, CSS_START, CSS_END, block)
    separator = "" if not current else (nl if current.endswith(("\n", "\r")) else nl * 2)
    return current + separator + block + nl


def patched_javascript(current: str) -> str:
    nl = newline_for(current)
    # The desktop hit geometry stays at the checkout's original 28/38 pixels.
    block = nl.join([
        JS_START,
        '        const coarsePointer = window.matchMedia("(pointer: coarse)").matches;',
        "        const hit = coarsePointer ? 44 : 28;",
        "        const corner = coarsePointer ? 48 : 38;",
        "        " + JS_END,
    ])
    if JS_START in current or JS_END in current:
        return replace_marked(current, JS_START, JS_END, block)
    original = re.compile(r"(?m)^        const hit = 28;\r?\n        const corner = 38;")
    if len(original.findall(current)) != 1:
        raise ValueError("Outpaint source differs from the audited checkout; no files were changed.")
    return original.sub(lambda _: "        " + block, current, count=1)


def atomic_write(path: Path, value: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Forge copy to adapt")
    parser.add_argument("--css", type=Path, default=Path(__file__).with_name("mobile.css"))
    parser.add_argument("--check", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if not (root / "launch.py").is_file():
        parser.error("--root must contain this Forge checkout's launch.py")
    javascript = root / "javascript" / "outpaint.js"
    guard_script = root / "javascript" / "00_forge_mobile_guard.js"
    stylesheet = root / "user.css"
    if not javascript.is_file():
        parser.error("--root must contain javascript/outpaint.js")
    if not args.css.is_file():
        parser.error(f"CSS file does not exist: {args.css}")
    try:
        old_js = read_utf8(javascript)
        old_css = read_utf8(stylesheet) if stylesheet.exists() else ""
        guard_js = read_utf8(Path(__file__).with_name("mobile_guard.js"))
        guard_css = read_utf8(Path(__file__).with_name("mobile_guard.css"))
        combined_css = patched_css(old_css, read_utf8(args.css))
        guard_start = "/* BEGIN FORGE MOBILE PARAMETER GUARD */"
        guard_end = "/* END FORGE MOBILE PARAMETER GUARD */"
        guard_block = guard_start + "\n" + guard_css.strip() + "\n" + guard_end
        if guard_start in combined_css or guard_end in combined_css:
            combined_css = replace_marked(combined_css, guard_start, guard_end, guard_block)
        else:
            combined_css += "\n" + guard_block + "\n"
        combined_css = outpaint_safety_patch.patched_css(combined_css)
        # Validate all modifications before any write.
        changes = [
            (javascript, old_js, outpaint_safety_patch.patched_javascript(patched_javascript(old_js))),
            (stylesheet, old_css, combined_css),
            (guard_script, read_utf8(guard_script) if guard_script.exists() else "", guard_js),
        ]
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    for path, old, new in changes:
        if old == new:
            print(f"Already current: {path}")
        elif args.check:
            print(f"Would update: {path}")
        else:
            atomic_write(path, new)
            print(f"Updated: {path}")
    if args.check:
        print("Mobile protection check complete; no files changed.")
    else:
        print("Mobile protection applied to this copy. Reload Forge to use it.")


if __name__ == "__main__":
    main()
