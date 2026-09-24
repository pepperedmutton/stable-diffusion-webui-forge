"""Krea 2 diffusion engine for the local Forge runtime."""

import torch

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionFlux
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .text_engine import KREA2_MAX_TOKENS, Krea2TextProcessingEngine


class Krea2(ForgeDiffusionEngine):
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={"gemma2_2b": huggingface_components["text_encoder"]},
            tokenizer_dict={"gemma2_2b": huggingface_components["tokenizer"]},
        )
        vae = VAE(model=huggingface_components["vae"])

        sampling = estimated_config.sampling_settings or {}
        # Krea defines its shift as ``mu`` in exp(mu) flow time.  Keep the
        # legacy key as a fallback for configurations saved by older builds.
        mu = float(sampling.get("mu", sampling.get("shift", 1.15)))
        k_predictor = PredictionFlux(mu=mu)
        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine = Krea2TextProcessingEngine(
            text_encoder=clip.cond_stage_model.gemma2_2b,
            tokenizer=clip.tokenizer.gemma2_2b,
        )
        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()
        self.is_wan = True
        self.use_shift = False

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        embeds, attention_mask = self.text_processing_engine(list(prompt))
        return {"crossattn": embeds, "attention_mask": attention_mask}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine.tokenize([prompt])[0])
        return token_count, KREA2_MAX_TOKENS

    @staticmethod
    def fix_dimensions(width, height):
        width = max(16, (int(width) + 15) // 16 * 16)
        height = max(16, (int(height) + 15) // 16 * 16)
        return width, height

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
