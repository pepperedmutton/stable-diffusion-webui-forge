import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.gemma_engine import GemmaTextProcessingEngine
from backend import memory_management

from modules.shared import opts


LUMINA_PROMPT_PREFIX_STEM = "You are an assistant designed to generate anime images based on textual prompts."
LUMINA_PROMPT_PREFIX = f"{LUMINA_PROMPT_PREFIX_STEM} <Prompt Start>"


class StableDiffusionLumina2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Lumina2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False
        self._spatial_multiple = 16

        clip = CLIP(
            model_dict={
                "gemma2_2b": huggingface_components["text_encoder"],
            },
            tokenizer_dict={
                "gemma2_2b": huggingface_components["tokenizer"],
            },
        )

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(shift=float(getattr(opts, "lumina_sampler_shift", 4.5) or 4.5))

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        def lumina_cfg(args):
            cond = args["cond"]
            if args["cond_scale"] <= 1:
                return cond

            guided = args["uncond"] + (cond - args["uncond"]) * args["cond_scale"]

            if getattr(opts, "lumina_cfg_normalization", True):
                cond_norm = torch.norm(cond.float(), dim=-1, keepdim=True)
                guided_norm = torch.norm(guided.float(), dim=-1, keepdim=True).clamp_min(1e-6)
                guided = guided * (cond_norm / guided_norm).to(guided.dtype)

            return guided

        unet.set_model_sampler_cfg_function(lumina_cfg, disable_cfg1_optimization=True)

        self.text_processing_engine = GemmaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.gemma2_2b,
            tokenizer=clip.tokenizer.gemma2_2b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def _get_max_sequence_length(self):
        return int(getattr(opts, "lumina_max_sequence_length", 256) or 256)

    def _prepare_prompts(self, prompt):
        texts = list(prompt)
        prepared = []

        for text in texts:
            stripped_text = text.lstrip()

            # Avoid duplicating the mandatory Lumina prefix if the user pasted it manually.
            if stripped_text.startswith(LUMINA_PROMPT_PREFIX) or stripped_text.startswith(LUMINA_PROMPT_PREFIX_STEM):
                prepared.append(text)
                continue

            prepared.append(f"{LUMINA_PROMPT_PREFIX}\n\n{text}")

        return prepared

    def fix_dimensions(self, width, height):
        width = max(self._spatial_multiple, int(width) // self._spatial_multiple * self._spatial_multiple)
        height = max(self._spatial_multiple, int(height) // self._spatial_multiple * self._spatial_multiple)
        return width, height

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        prepared_prompt = self._prepare_prompts(prompt)
        prompt_embeds, attention_mask = self.text_processing_engine(
            prepared_prompt,
            max_length=self._get_max_sequence_length(),
        )
        return dict(crossattn=prompt_embeds, attention_mask=attention_mask)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        prepared_prompt = self._prepare_prompts([prompt])[0]
        token_count = len(self.text_processing_engine.tokenize([prepared_prompt], max_length=4096)[0])
        return token_count, self._get_max_sequence_length()

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
