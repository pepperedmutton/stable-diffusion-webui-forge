import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionAnima, PredictionDiscreteFlow
from backend import memory_management

ANIMA_DEFAULT_NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima, model_list.AnimaSchnell, model_list.AnimaCosmos]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False
        self.is_cosmos_anima = isinstance(estimated_config, model_list.AnimaCosmos)

        if self.is_cosmos_anima:
            clip = CLIP(
                model_dict={
                    'gemma2_2b': huggingface_components['text_encoder'],
                },
                tokenizer_dict={
                    'gemma2_2b': huggingface_components['tokenizer'],
                    't5xxl': huggingface_components['tokenizer_2'],
                },
            )
        else:
            clip = CLIP(
                model_dict={
                    'clip_l': huggingface_components['text_encoder'],
                    't5xxl': huggingface_components['text_encoder_2']
                },
                tokenizer_dict={
                    'clip_l': huggingface_components['tokenizer'],
                    't5xxl': huggingface_components['tokenizer_2']
                }
            )

        vae = VAE(model=huggingface_components['vae'])

        if self.is_cosmos_anima:
            sampling_settings = estimated_config.sampling_settings or {}
            k_predictor = PredictionDiscreteFlow(
                shift=float(sampling_settings.get("shift", 3.0)),
                multiplier=float(sampling_settings.get("multiplier", 1.0)),
            )
            self.use_distilled_cfg_scale = True
        else:
            if 'schnell' in estimated_config.huggingface_repo.lower():
                k_predictor = PredictionAnima(
                    mu=1.0
                )
            else:
                k_predictor = PredictionAnima(
                    seq_len=4096,
                    base_seq_len=256,
                    max_seq_len=4096,
                    base_shift=0.5,
                    max_shift=1.15,
                )
                self.use_distilled_cfg_scale = True

        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        if self.is_cosmos_anima:
            self.text_processing_engine_qwen = QwenTextProcessingEngine(
                text_encoder=clip.cond_stage_model.gemma2_2b,
                tokenizer=clip.tokenizer.gemma2_2b,
                max_length=512,
            )
            self.tokenizer_t5 = clip.tokenizer.t5xxl
        else:
            self.text_processing_engine_l = ClassicTextProcessingEngine(
                text_encoder=clip.cond_stage_model.clip_l,
                tokenizer=clip.tokenizer.clip_l,
                embedding_dir=dynamic_args['embedding_dir'],
                embedding_key='clip_l',
                embedding_expected_shape=768,
                emphasis_name=dynamic_args['emphasis_name'],
                text_projection=False,
                minimal_clip_skip=1,
                clip_skip=1,
                return_pooled=True,
                final_layer_norm=True,
            )

            self.text_processing_engine_t5 = T5TextProcessingEngine(
                text_encoder=clip.cond_stage_model.t5xxl,
                tokenizer=clip.tokenizer.t5xxl,
                emphasis_name=dynamic_args['emphasis_name'],
            )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        if not self.is_cosmos_anima:
            self.text_processing_engine_l.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        prompt_texts = list(prompt)

        # Mirror the official ComfyUI Anima workflow baseline: keep a default
        # negative quality filter when the negative prompt is left empty.
        if getattr(prompt, "is_negative_prompt", False):
            prompt_texts = [p if str(p).strip() else ANIMA_DEFAULT_NEGATIVE for p in prompt_texts]

        if self.is_cosmos_anima:
            cond_qwen, _ = self.text_processing_engine_qwen(prompt_texts, max_length=512)

            t5_batch = self.tokenizer_t5(
                prompt_texts,
                padding=False,
                truncation=True,
                max_length=511,
                add_special_tokens=False,
            )["input_ids"]

            t5_eos = self.tokenizer_t5.eos_token_id if self.tokenizer_t5.eos_token_id is not None else 1
            t5_pad = self.tokenizer_t5.pad_token_id if self.tokenizer_t5.pad_token_id is not None else 0
            t5_batch = [ids + [int(t5_eos)] for ids in t5_batch]
            max_t5 = max(len(ids) for ids in t5_batch)

            t5_ids = torch.full((len(t5_batch), max_t5), int(t5_pad), dtype=torch.long)
            t5_weights = torch.zeros((len(t5_batch), max_t5, 1), dtype=torch.float32)
            for i, ids in enumerate(t5_batch):
                n = len(ids)
                t5_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
                t5_weights[i, :n, 0] = 1.0

            cond = dict(
                crossattn=cond_qwen,
                t5xxl_ids=t5_ids,
                t5xxl_weights=t5_weights,
            )
        else:
            cond_l, pooled_l = self.text_processing_engine_l(prompt_texts)
            cond_t5 = self.text_processing_engine_t5(prompt_texts)
            cond = dict(crossattn=cond_t5, vector=pooled_l)

        if self.use_distilled_cfg_scale:
            distilled_cfg_scale = getattr(prompt, 'distilled_cfg_scale', 3.5) or 3.5
            cond['guidance'] = torch.FloatTensor([distilled_cfg_scale] * len(prompt))
            print(f'Distilled CFG Scale: {distilled_cfg_scale}')
        else:
            print('Distilled CFG Scale will be ignored for Schnell')

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        if self.is_cosmos_anima:
            token_count = len(self.tokenizer_t5(prompt, add_special_tokens=False)["input_ids"])
        else:
            token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

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
