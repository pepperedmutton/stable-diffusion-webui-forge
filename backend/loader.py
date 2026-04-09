import os
import torch
import logging
import importlib

import backend.args
import huggingface_guess

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from transformers import modeling_utils

from backend import memory_management
from backend.utils import read_arbitrary_config, load_torch_file, beautiful_print_gguf_state_dict_statics
from backend.state_dict import try_filter_state_dict, load_state_dict
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.misc.lumina2_state_dict import convert_lumina2_to_diffusers, is_lumina2_original_state_dict

from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sd20 import StableDiffusion2
from backend.diffusion_engine.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
from backend.diffusion_engine.sd35 import StableDiffusion3
from backend.diffusion_engine.lumina2 import StableDiffusionLumina2
from backend.diffusion_engine.anima import Anima
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.chroma import Chroma


possible_models = [StableDiffusion, StableDiffusion2, StableDiffusionXLRefiner, StableDiffusionXL, StableDiffusion3, StableDiffusionLumina2, Chroma, Anima, Flux]


logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ['feature_extractor', 'safety_checker']:
        return None

    if lib_name in ['transformers', 'diffusers']:
        if component_name in ['scheduler']:
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if component_name.startswith('tokenizer'):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            comp = cls.from_pretrained(os.path.join(repo_path, component_name))
            comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            return comp
        if cls_name in ['AutoencoderKL', 'WanVAEModel']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have VAE state dict!'

            if cls_name == 'AutoencoderKL':
                config = IntegratedAutoencoderKL.load_config(config_path)

                with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                    model = IntegratedAutoencoderKL.from_config(config)

                if 'decoder.up_blocks.0.resnets.0.norm1.weight' in state_dict.keys(): #diffusers format
                    state_dict = huggingface_guess.diffusers_convert.convert_vae_state_dict(state_dict)
                load_state_dict(model, state_dict, ignore_start='loss.')
            else:
                from backend.nn.wan_vae import IntegratedWanVAEModel
                config = read_arbitrary_config(config_path)
                with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                    model = IntegratedWanVAEModel(**config)
                load_state_dict(model, state_dict)
            return model
        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have CLIP state dict!'

            from transformers import CLIPTextConfig, CLIPTextModel
            config = CLIPTextConfig.from_pretrained(config_path)

            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

            load_state_dict(model, state_dict, ignore_errors=[
                'transformer.text_projection.weight',
                'transformer.text_model.embeddings.position_ids',
                'logit_scale'
            ], log_name=cls_name)

            return model
        if cls_name == 'Gemma2Model':
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have Gemma2 state dict!'

            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            normalized_state_dict = {}
            for key, value in state_dict.items():
                if key == 'logit_scale':
                    continue
                if key.startswith('transformer.'):
                    key = key[len('transformer.'):]
                if key.startswith('model.'):
                    key = key[len('model.'):]
                normalized_state_dict[key] = value

            # Anima uses a Qwen3-0.6B style text encoder (q_norm / k_norm + head_dim=128),
            # but model_index labels it as Gemma2 for compatibility.
            is_qwen3_style = any(k.endswith('.self_attn.q_norm.weight') for k in normalized_state_dict.keys())
            if is_qwen3_style:
                from backend.nn.qwen3 import IntegratedQwen3Model

                config = read_arbitrary_config(config_path)
                with modeling_utils.no_init_weights():
                    with using_forge_operations(**to_args, manual_cast_enabled=True):
                        model = IntegratedQwen3Model(config=config).to(**to_args)

                missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
                if len(missing) > 0:
                    print(f'Qwen3Model Missing: {missing}')
                if len(unexpected) > 0:
                    print(f'Qwen3Model Unexpected: {unexpected}')
                return model

            from transformers import Gemma2Config, Gemma2Model
            config = Gemma2Config.from_pretrained(config_path)

            # Keep no_init_weights to avoid init path incompatibility with forge operations,
            # then explicitly initialize unmatched parameters after loading.
            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = Gemma2Model(config).to(**to_args)

            missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)

            # Some Anima text encoder checkpoints (Qwen3-0.6B style) only partially match Gemma2.
            # Without explicit init, missing params can contain undefined data and destabilize sampling.
            named_params = dict(model.named_parameters())
            named_buffers = dict(model.named_buffers())
            with torch.no_grad():
                for key in missing:
                    tensor = named_params.get(key, None)
                    if tensor is None:
                        tensor = named_buffers.get(key, None)
                    if tensor is None:
                        continue
                    if key.endswith(".bias"):
                        tensor.zero_()
                    elif tensor.ndim == 1:
                        tensor.fill_(1.0)
                    else:
                        torch.nn.init.normal_(tensor, mean=0.0, std=float(getattr(config, "initializer_range", 0.02)))

            if len(missing) > 0:
                print(f'{cls_name} Missing: {missing}')
            if len(unexpected) > 0:
                print(f'{cls_name} Unexpected: {unexpected}')
            return model
        if cls_name == 'T5EncoderModel':
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have T5 state dict!'

            from backend.nn.t5 import IntegratedT5
            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
                print(f'Using Detected T5 Data Type: {state_dict_dtype}')
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ['nf4', 'fp4', 'gguf']:
                    print(f'Using pre-quant state dict!')
                    if state_dict_dtype in ['gguf']:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f'Using Default T5 Data Type: {storage_dtype}')

            if storage_dtype in ['nf4', 'fp4', 'gguf']:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = IntegratedT5(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = IntegratedT5(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=['transformer.encoder.embed_tokens.weight', 'logit_scale'])

            return model
        if cls_name in ['UNet2DConditionModel', 'AnimaTransformer2DModel', 'FluxTransformer2DModel', 'SD3Transformer2DModel', 'ChromaTransformer2DModel', 'Lumina2Transformer2DModel']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have model state dict!'

            model_loader = None
            if cls_name == 'UNet2DConditionModel':
                model_loader = lambda c: IntegratedUNet2DConditionModel.from_config(c)
            elif cls_name == 'AnimaTransformer2DModel':
                anima_cosmos_cls = getattr(huggingface_guess.model_list, 'AnimaCosmos', None)
                is_anima_cosmos = (anima_cosmos_cls is not None and isinstance(guess, anima_cosmos_cls))
                # huggingface_guess.guess() strips `image_model`, so use model class and config hints.
                is_anima_cosmos = is_anima_cosmos or (
                    'max_img_h' in guess.unet_config and 'patch_spatial' in guess.unet_config
                )

                if is_anima_cosmos:
                    from backend.nn.anima_cosmos import IntegratedAnimaCosmosTransformer2DModel
                    model_loader = lambda c: IntegratedAnimaCosmosTransformer2DModel(**c)
                    if any(k.startswith("net.") for k in state_dict.keys()):
                        state_dict = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}
                    if not any(k.startswith("inner_model.") for k in state_dict.keys()):
                        state_dict = {f"inner_model.{k}": v for k, v in state_dict.items()}
                else:
                    from backend.nn.anima import IntegratedAnimaTransformer2DModel
                    model_loader = lambda c: IntegratedAnimaTransformer2DModel(**c)
            elif cls_name == 'FluxTransformer2DModel':
                from backend.nn.flux import IntegratedFluxTransformer2DModel
                model_loader = lambda c: IntegratedFluxTransformer2DModel(**c)
            elif cls_name == 'ChromaTransformer2DModel':
                from backend.nn.chroma import IntegratedChromaTransformer2DModel
                model_loader = lambda c: IntegratedChromaTransformer2DModel(**c)
            elif cls_name == 'SD3Transformer2DModel':
                from backend.nn.mmditx import MMDiTX
                model_loader = lambda c: MMDiTX(**c)
            elif cls_name == 'Lumina2Transformer2DModel':
                from backend.nn.lumina2 import IntegratedLumina2Transformer2DModel
                model_loader = lambda c: IntegratedLumina2Transformer2DModel(**c)
                if is_lumina2_original_state_dict(state_dict):
                    state_dict = convert_lumina2_to_diffusers(state_dict)

            unet_config = guess.unet_config.copy()
            state_dict_parameters = memory_management.state_dict_parameters(state_dict)
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)

            unet_storage_dtype_overwrite = backend.args.dynamic_args.get('forge_unet_storage_dtype')

            if unet_storage_dtype_overwrite is not None:
                storage_dtype = unet_storage_dtype_overwrite
            elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
                print(f'Using Detected UNet Type: {state_dict_dtype}')
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ['nf4', 'fp4', 'gguf']:
                    print(f'Using pre-quant state dict!')
                    if state_dict_dtype in ['gguf']:
                        beautiful_print_gguf_state_dict_statics(state_dict)

            load_device = memory_management.get_torch_device()
            computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
            offload_device = memory_management.unet_offload_device()

            if storage_dtype in ['nf4', 'fp4', 'gguf']:
                initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
                with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                    model = model_loader(unet_config)
            else:
                initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
                need_manual_cast = storage_dtype != computation_dtype
                to_args = dict(device=initial_device, dtype=storage_dtype)

                with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast):
                    model = model_loader(unet_config).to(**to_args)

            load_state_dict(model, state_dict)

            if hasattr(model, '_internal_dict'):
                model._internal_dict = FrozenDict(unet_config)
            else:
                model.config = unet_config

            model.storage_dtype = storage_dtype
            model.computation_dtype = computation_dtype
            model.load_device = load_device
            model.initial_device = initial_device
            model.offload_device = offload_device

            return model

    print(f'Skipped: {component_name} = {lib_name}.{cls_name}')
    return None


def replace_state_dict(sd, asd, guess):
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    if 'enc.blk.0.attn_k.weight' in asd:
        wierd_t5_format_from_city96 = {
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        wierd_t5_pre_quant_keys_from_city96 = ['shared.weight']
        asd_new = {}
        for k, v in asd.items():
            for s, d in wierd_t5_format_from_city96.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in wierd_t5_pre_quant_keys_from_city96:
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    if "decoder.conv_in.weight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    if "conv1.weight" in asd and "decoder.conv1.weight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v


    ##  identify model type
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    sd3_test_key = "model.diffusion_model.final_layer.adaLN_modulation.1.bias"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"

    model_type = "-"
    if legacy_test_key in sd:
        match sd[legacy_test_key].shape[1]:
            case 768:
                model_type = "sd1"
            case 1024:
                model_type = "sd2"
            case 1280:
                model_type = "xlrf"     # sdxl refiner model
            case 2048:
                model_type = "sdxl"
    elif flux_test_key in sd:
        model_type = "flux"
    elif sd3_test_key in sd:
        model_type = "sd3"

    ##  prefixes used by various model types for CLIP-L
    prefix_L = {
        "-"   : None,
        "sd1" : "cond_stage_model.transformer.",
        "sd2" : None,
        "xlrf": None,
        "sdxl": "conditioner.embedders.0.transformer.",
        "flux": "text_encoders.clip_l.transformer.",
        "sd3" : "text_encoders.clip_l.transformer.",
    }
    ##  prefixes used by various model types for CLIP-G
    prefix_G = {
        "-"   : None,
        "sd1" : None,
        "sd2" : None,
        "xlrf": "conditioner.embedders.0.model.transformer.",
        "sdxl": "conditioner.embedders.1.model.transformer.",
        "flux": None,
        "sd3" : "text_encoders.clip_g.transformer.",
    }
    ##  prefixes used by various model types for CLIP-H
    prefix_H = {
        "-"   : None,
        "sd1" : None,
        "sd2" : "conditioner.embedders.0.model.",
        "xlrf": None,
        "sdxl": None,
        "flux": None,
        "sd3" : None,
    }


    ##  VAE format 0 (extracted from model, could be sd1, sd2, sdxl, sd3).
    if "first_stage_model.decoder.conv_in.weight" in asd:
        channels = asd["first_stage_model.decoder.conv_in.weight"].shape[1]
        if model_type == "sd1" or model_type == "sd2" or model_type == "xlrf" or model_type == "sdxl":
            if channels == 4:
                for k, v in asd.items():
                    sd[k] = v
        elif model_type == "sd3":
            if channels == 16:
                for k, v in asd.items():
                    sd[k] = v

    ##  CLIP-H
    CLIP_H = {     #   key to identify source model             old_prefix
        'cond_stage_model.model.ln_final.weight'            : 'cond_stage_model.model.',
#        'text_model.encoder.layers.0.layer_norm1.bias'      : 'text_model'.    # would need converting
        }
    for CLIP_key in CLIP_H.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 1024:
            new_prefix = prefix_H[model_type]
            old_prefix = CLIP_H[CLIP_key]

            if new_prefix is not None:
                for k, v in asd.items():
                    new_k = k.replace(old_prefix, new_prefix)
                    sd[new_k] = v

    ##  CLIP-G
    CLIP_G = {     #   key to identify source model                                                old_prefix
        'conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias'               : 'conditioner.embedders.1.model.transformer.',
        'text_encoders.clip_g.transformer.text_model.encoder.layers.0.layer_norm1.bias' : 'text_encoders.clip_g.transformer.',
        'text_model.encoder.layers.0.layer_norm1.bias'                                  : '',
        'transformer.resblocks.0.ln_1.bias'                                             : 'transformer.'
    }
    for CLIP_key in CLIP_G.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 1280:
            new_prefix = prefix_G[model_type]
            old_prefix = CLIP_G[CLIP_key]

            if new_prefix is not None:
                if "resblocks" not in CLIP_key and model_type != "sd3": # need to convert
                    def convert_transformers(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "{}text_model.embeddings.position_embedding.weight" : "{}positional_embedding",
                            "{}text_model.embeddings.token_embedding.weight"    : "{}token_embedding.weight",
                            "{}text_model.final_layer_norm.weight"              : "{}ln_final.weight",
                            "{}text_model.final_layer_norm.bias"                : "{}ln_final.bias",
                            "text_projection.weight"                            : "{}text_projection",
                        }
                        resblock_to_replace = {
                            "layer_norm1"           : "ln_1",
                            "layer_norm2"           : "ln_2",
                            "mlp.fc1"               : "mlp.c_fc",
                            "mlp.fc2"               : "mlp.c_proj",
                            "self_attn.out_proj"    : "attn.out_proj" ,
                        }

                        for x in keys_to_replace:   #   remove trailing 'transformer.' from new prefix
                            k = x.format(prefix_from)
                            statedict[keys_to_replace[x].format(prefix_to[:-12])] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}resblocks.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.q_proj", y)
                                weightsQ = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.k_proj", y)
                                weightsK = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.v_proj", y)
                                weightsV = statedict.pop(k_from)

                                k_to = "{}resblocks.{}.attn.in_proj_{}".format(prefix_to, resblock, y)

                                statedict[k_to] = torch.cat((weightsQ, weightsK, weightsV))
                        return statedict

                    asd = convert_transformers(asd, old_prefix, new_prefix, 32)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    ##  CLIP-L
    CLIP_L = {     #   key to identify source model                                                    old_prefix
        'cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias'         : 'cond_stage_model.transformer.',
        'conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias'  : 'conditioner.embedders.0.transformer.',
        'text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias'     : 'text_encoders.clip_l.transformer.',
        'text_model.encoder.layers.0.layer_norm1.bias'                                      : '',
        'transformer.resblocks.0.ln_1.bias'                                                 : 'transformer.'
    }

    for CLIP_key in CLIP_L.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 768:
            new_prefix = prefix_L[model_type]
            old_prefix = CLIP_L[CLIP_key]

            if new_prefix is not None:
                if "resblocks" in CLIP_key: # need to convert
                    def transformers_convert(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "positional_embedding"  : "{}text_model.embeddings.position_embedding.weight",
                            "token_embedding.weight": "{}text_model.embeddings.token_embedding.weight",
                            "ln_final.weight"       : "{}text_model.final_layer_norm.weight",
                            "ln_final.bias"         : "{}text_model.final_layer_norm.bias",
                            "text_projection"       : "text_projection.weight",
                        }
                        resblock_to_replace = {
                            "ln_1"          : "layer_norm1",
                            "ln_2"          : "layer_norm2",
                            "mlp.c_fc"      : "mlp.fc1",
                            "mlp.c_proj"    : "mlp.fc2",
                            "attn.out_proj" : "self_attn.out_proj",
                        }

                        for k in keys_to_replace:
                            statedict[keys_to_replace[k].format(prefix_to)] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
                                weights = statedict.pop(k_from)
                                shape_from = weights.shape[0] // 3
                                for x in range(3):
                                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                                    statedict[k_to] = weights[shape_from*x:shape_from*(x + 1)]
                        return statedict

                    asd = transformers_convert(asd, old_prefix, new_prefix, 12)
                    for k, v in asd.items():
                        sd[k] = v
                
                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v


    if 'encoder.block.0.layer.0.SelfAttention.k.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = v

    if "model.embed_tokens.weight" in asd or "embed_tokens.weight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}gemma2_2b.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            if k == "logit_scale":
                continue
            sd[f"{text_encoder_key_prefix}gemma2_2b.{k}"] = v

    return sd


def preprocess_state_dict(sd):
    if not any(k.startswith("model.diffusion_model") for k in sd.keys()):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}

    return sd


def detect_anima_cosmos_state_dict(sd):
    anima_keys = (
        "model.diffusion_model.blocks.0.mlp.layer1.weight",
        "model.diffusion_model.net.blocks.0.mlp.layer1.weight",
    )
    llm_adapter_prefixes = (
        "model.diffusion_model.llm_adapter.",
        "model.diffusion_model.net.llm_adapter.",
    )

    if any(k in sd for k in anima_keys):
        return True

    if any(any(x.startswith(prefix) for x in sd.keys()) for prefix in llm_adapter_prefixes):
        return True

    return False


def split_state_dict(sd, additional_state_dicts: list = None):
    loaded_sd = load_torch_file(sd)
    # Never mutate cached checkpoint dicts in-place. sd_models keeps them in RAM cache,
    # and split_state_dict() does destructive key transforms later.
    original_sd = dict(loaded_sd) if isinstance(loaded_sd, dict) else loaded_sd
    preprocessed_sd = preprocess_state_dict(original_sd)

    # Keep the existing behavior first, then fall back to the raw state dict.
    # Some checkpoints can be recognized only before prefix normalization.
    sd = preprocessed_sd
    guess = huggingface_guess.guess(sd)
    if guess is None and original_sd is not preprocessed_sd:
        fallback_guess = huggingface_guess.guess(original_sd)
        if fallback_guess is not None:
            sd = original_sd
            guess = fallback_guess

    if guess is None:
        if detect_anima_cosmos_state_dict(preprocessed_sd) or detect_anima_cosmos_state_dict(original_sd):
            raise RuntimeError(
                "Detected Anima (Cosmos Predict2) checkpoint format, but this Forge runtime "
                "does not include a compatible loader for that architecture."
            )
        raise RuntimeError("Unable to detect a supported model architecture from checkpoint state dict.")

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            asd = load_torch_file(asd)
            sd = replace_state_dict(sd, asd, guess)
            del asd

    guess.clip_target = guess.clip_target(sd)
    guess.model_type = guess.model_type(sd)
    guess.ztsnr = 'ztsnr' in sd

    sd = guess.process_vae_state_dict(sd)

    state_dict = {
        guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess

# To be removed once PR merged on huggingface_guess
chroma_is_in_huggingface_guess = hasattr(huggingface_guess.model_list, "Chroma")

if not chroma_is_in_huggingface_guess:
    class GuessChroma:
        huggingface_repo = 'Chroma'
        unet_extra_config = {
            'guidance_out_dim': 3072,
            'guidance_hidden_dim': 5120,
            'guidance_n_layers': 5
        }
        unet_remove_config = ['guidance_embed']
@torch.inference_mode()
def forge_loader(sd, additional_state_dicts=None, source_path=None):
    try:
        state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts)
    except Exception as e:
        raise ValueError(f'Failed to recognize model type! ({e})') from e
    
    if not chroma_is_in_huggingface_guess \
        and estimated_config.huggingface_repo == "black-forest-labs/FLUX.1-schnell"  \
        and "transformer" in state_dicts \
        and "distilled_guidance_layer.layers.0.in_layer.bias" in state_dicts["transformer"]:
        estimated_config.huggingface_repo = GuessChroma.huggingface_repo
        for x in GuessChroma.unet_extra_config:
            estimated_config.unet_config[x] = GuessChroma.unet_extra_config[x]
        for x in GuessChroma.unet_remove_config:
            del estimated_config.unet_config[x]
        state_dicts['text_encoder'] = state_dicts['text_encoder_2']
        del state_dicts['text_encoder_2'] 
    repo_name = estimated_config.huggingface_repo

    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component

    yaml_config = None
    yaml_config_prediction_type = None

    try:
        import yaml
        from pathlib import Path
        yaml_source_path = source_path if source_path is not None else sd
        if isinstance(yaml_source_path, (str, bytes, os.PathLike)):
            config_filename = os.path.splitext(yaml_source_path)[0] + '.yaml'
            if Path(config_filename).is_file():
                with open(config_filename, 'r') as stream:
                    yaml_config = yaml.safe_load(stream)
    except ImportError:
        pass

    # Fix Huggingface prediction type using .yaml config or estimated config detection
    prediction_types = {
        'EPS': 'epsilon',
        'V_PREDICTION': 'v_prediction',
        'EDM': 'edm',
    }

    has_prediction_type = 'scheduler' in huggingface_components and hasattr(huggingface_components['scheduler'], 'config') and 'prediction_type' in huggingface_components['scheduler'].config

    if yaml_config is not None:
        yaml_config_prediction_type: str = (
                yaml_config.get('model', {}).get('params', {}).get('parameterization', '')
            or  yaml_config.get('model', {}).get('params', {}).get('denoiser_config', {}).get('params', {}).get('scaling_config', {}).get('target', '')
        )
        if yaml_config_prediction_type == 'v' or yaml_config_prediction_type.endswith(".VScaling"):
            yaml_config_prediction_type = 'v_prediction'
        else:
            # Use estimated prediction config if no suitable prediction type found
            yaml_config_prediction_type = ''

    if has_prediction_type:
        if yaml_config_prediction_type:
            huggingface_components['scheduler'].config.prediction_type = yaml_config_prediction_type
        else:
            huggingface_components['scheduler'].config.prediction_type = prediction_types.get(estimated_config.model_type.name, huggingface_components['scheduler'].config.prediction_type)

    if not chroma_is_in_huggingface_guess and estimated_config.huggingface_repo == "Chroma":
        return Chroma(estimated_config=estimated_config, huggingface_components=huggingface_components)
    for M in possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print('Failed to recognize model type!')
    return None
