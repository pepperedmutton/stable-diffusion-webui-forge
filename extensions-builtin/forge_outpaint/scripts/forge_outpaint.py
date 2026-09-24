from __future__ import annotations

import gradio as gr

import modules.scripts as scripts
from modules_forge.outpaint import (
    FILL_MODES,
    REGION_CANVAS_POSITION,
    REGION_MARGINS,
    run_outpaint,
)


class Script(scripts.Script):
    def title(self):
        return "Forge Outpaint"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        gr.HTML(
            "<p>Forge Outpaint uses the input image. It ignores the main Width and "
            "Height controls. The output format is always PNG.</p>"
        )

        region_mode = gr.Radio(
            label="Region mode",
            choices=[REGION_MARGINS, REGION_CANVAS_POSITION],
            value=REGION_MARGINS,
            elem_id=self.elem_id("region_mode"),
        )

        with gr.Row():
            left = gr.Number(
                label="Left margin",
                value=128,
                precision=0,
                minimum=0,
                maximum=16384,
                step=1,
                elem_id=self.elem_id("left"),
            )
            right = gr.Number(
                label="Right margin",
                value=128,
                precision=0,
                minimum=0,
                maximum=16384,
                step=1,
                elem_id=self.elem_id("right"),
            )
            top = gr.Number(
                label="Top margin",
                value=128,
                precision=0,
                minimum=0,
                maximum=16384,
                step=1,
                elem_id=self.elem_id("top"),
            )
            bottom = gr.Number(
                label="Bottom margin",
                value=128,
                precision=0,
                minimum=0,
                maximum=16384,
                step=1,
                elem_id=self.elem_id("bottom"),
            )

        with gr.Row():
            canvas_width = gr.Number(
                label="Canvas width",
                value=1024,
                precision=0,
                minimum=1,
                maximum=32768,
                step=1,
                elem_id=self.elem_id("canvas_width"),
            )
            canvas_height = gr.Number(
                label="Canvas height",
                value=1024,
                precision=0,
                minimum=1,
                maximum=32768,
                step=1,
                elem_id=self.elem_id("canvas_height"),
            )
            source_x = gr.Number(
                label="Input X position",
                value=0,
                precision=0,
                minimum=0,
                maximum=32768,
                step=1,
                elem_id=self.elem_id("source_x"),
            )
            source_y = gr.Number(
                label="Input Y position",
                value=0,
                precision=0,
                minimum=0,
                maximum=32768,
                step=1,
                elem_id=self.elem_id("source_y"),
            )

        with gr.Row():
            context_overlap = gr.Number(
                label="Context overlap",
                value=32,
                precision=0,
                minimum=0,
                maximum=1024,
                step=1,
                elem_id=self.elem_id("context_overlap"),
            )
            mask_blur = gr.Number(
                label="Mask blur",
                value=8,
                precision=0,
                minimum=0,
                maximum=256,
                step=1,
                elem_id=self.elem_id("mask_blur"),
            )
            fill_mode = gr.Dropdown(
                label="Fill mode",
                choices=list(FILL_MODES),
                value="Edge",
                elem_id=self.elem_id("fill_mode"),
            )
            max_megapixels = gr.Number(
                label="Maximum megapixels",
                value=4.0,
                precision=1,
                minimum=0.1,
                maximum=64,
                step=0.1,
                elem_id=self.elem_id("max_megapixels"),
            )

        gr.HTML('<div id="forge_outpaint_preview" class="forge-outpaint-preview"></div>')

        return [
            region_mode,
            left,
            right,
            top,
            bottom,
            canvas_width,
            canvas_height,
            source_x,
            source_y,
            context_overlap,
            mask_blur,
            fill_mode,
            max_megapixels,
        ]

    def run(
        self,
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
    ):
        return run_outpaint(
            p,
            region_mode,
            left,
            right,
            top,
            bottom,
            canvas_width,
            canvas_height,
            source_x,
            source_y,
            context_overlap,
            mask_blur,
            fill_mode,
            max_megapixels,
        )
