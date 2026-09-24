# Forge Outpaint

Forge Outpaint is a standard Img2img operation next to Inpaint.
It expands an input image into a larger canvas and uses the current Img2img
prompt, model, sampler, and seed.

The extension keeps the input image pixels unchanged.
It restores the input image after sampling and all image post-processing.
It saves final samples as PNG files.

## User interface

1. Open the Img2img tab.
2. Open the Outpaint operation next to Inpaint.
3. Upload an input image.
4. Enter the prompt and the negative prompt.
5. Pull the frame edges to set the output area.
6. Pull the input image reference to set its position in the output frame.
7. If necessary, open `Exact values` and enter the region values.
8. Select Generate. Keep the Script menu set to `None`.

The editor displays the actual input image as a reference. It changes geometry
values only. It does not resample or replace the input image.

`Margins` adds independent areas at the left, right, top, and bottom.
`Canvas position` sets the canvas size and the input image position.

The standard Img2img Width and Height controls do not set the outpaint canvas.
Forge Outpaint calculates an internal aligned canvas and crops it to the requested size.

The `Forge Outpaint` Script menu entry remains available for older API clients
and saved workflows. Do not select it when you use the native Outpaint operation.

## API

Use the standard `POST /sdapi/v1/img2img` route.
Set `script_name` to `Forge Outpaint`.

The positional `script_args` order is:

1. Region mode
2. Left
3. Right
4. Top
5. Bottom
6. Canvas width
7. Canvas height
8. Input X
9. Input Y
10. Context overlap
11. Mask blur
12. Fill mode
13. Maximum canvas area in megapixels

Use `Margins` or `Canvas position` for the region mode.
Use `Edge`, `Reflect`, or `Noise` for the fill mode.
`Edge` uses reflected image context when an outer edge is nearly uniform but nearby pixels contain detail.
Send all 13 arguments. Values for the inactive region mode are ignored.
The default maximum canvas area is 4 megapixels.

API image responses are PNG data even when the global sample format is JPEG or WebP.
When `save_images` is false, the extension does not save the working canvas or a final sample.

## Pixel protection contract

The extension applies EXIF orientation and converts the input image to RGBA.
The matching RGBA region in each PNG result is pixel-identical to that normalized input image.
Later conversion to JPEG or lossy WebP can change these pixels.
The PNG metadata records the requested final canvas size and the aligned working size.
Outpaint PNG files keep standard text metadata, but do not use pixel-based stealth metadata.

## Tests

Run the tests from the Forge repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s extensions-builtin\forge_outpaint\tests -v
```
