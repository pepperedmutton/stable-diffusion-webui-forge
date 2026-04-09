import torch


def is_lumina2_original_state_dict(state_dict):
    return any(
        key in state_dict
        for key in [
            "cap_embedder.0.weight",
            "model.diffusion_model.cap_embedder.0.weight",
        ]
    )


def convert_lumina2_to_diffusers(checkpoint):
    checkpoint = dict(checkpoint)
    converted_state_dict = {}

    checkpoint.pop("norm_final.weight", None)

    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith("model.diffusion_model."):
            checkpoint[key.replace("model.diffusion_model.", "", 1)] = checkpoint.pop(key)

    keys = list(checkpoint.keys())

    lumina_key_map = {
        "cap_embedder": "time_caption_embed.caption_embedder",
        "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
        "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
        "attention": "attn",
        ".out.": ".to_out.0.",
        "k_norm": "norm_k",
        "q_norm": "norm_q",
        "w1": "linear_1",
        "w2": "linear_2",
        "w3": "linear_3",
        "adaLN_modulation.1": "norm1.linear",
    }
    attention_norm_map = {
        "attention_norm1": "norm1.norm",
        "attention_norm2": "norm2",
    }
    context_refiner_map = {
        "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
        "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
        "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
        "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
    }
    final_layer_map = {
        "final_layer.adaLN_modulation.1": "norm_out.linear_1",
        "final_layer.linear": "norm_out.linear_2",
    }

    def convert_qkv_tensor(tensor, diffusers_key):
        q_dim = 2304
        k_dim = 768
        v_dim = 768
        to_q, to_k, to_v = torch.split(tensor, [q_dim, k_dim, v_dim], dim=0)
        return {
            diffusers_key.replace("qkv", "to_q"): to_q,
            diffusers_key.replace("qkv", "to_k"): to_k,
            diffusers_key.replace("qkv", "to_v"): to_v,
        }

    for key in keys:
        diffusers_key = key

        for src, dst in context_refiner_map.items():
            diffusers_key = diffusers_key.replace(src, dst)
        for src, dst in final_layer_map.items():
            diffusers_key = diffusers_key.replace(src, dst)
        for src, dst in attention_norm_map.items():
            diffusers_key = diffusers_key.replace(src, dst)
        for src, dst in lumina_key_map.items():
            diffusers_key = diffusers_key.replace(src, dst)

        value = checkpoint.pop(key)
        if "qkv" in diffusers_key:
            converted_state_dict.update(convert_qkv_tensor(value, diffusers_key))
        else:
            converted_state_dict[diffusers_key] = value

    return converted_state_dict
