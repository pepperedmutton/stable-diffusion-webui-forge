#!/usr/bin/env python3
"""Run the preserved Gemma text encoder as a standalone component.

Default --plan never imports Torch, loads weights, or runs inference. Use a NEW
process with the persistent Qwen interpreter, after the Forge GPU job has ended:
  forge/runtimes/qwen-image-2.1/bin/python forge/colab/verify_gemma.py --run --device cuda

This proves text-encoder execution only. This fork has no active NewBie image
pipeline, so success here does not mean Gemma generated an image. No model,
tokenizer, package or configuration is downloaded by this harness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

DRIVE_ROOT = Path('/content/drive/MyDrive/ForgeColab')
MODEL_RELATIVE = 'models/text_encoder/gemma_3_4b_it_bf16.safetensors'
MODEL_BYTES = 7765267250
MODEL_SHA256 = '9ca0ed0c8b0093357c044fc528f439d369fb1029200dcd8011328771e8af8d6e'
CONFIG_SOURCE = 'https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298/text_encoder/config.json'
TEXT_CONFIG = {'_sliding_window_pattern': 6, 'attention_bias': False, 'attention_dropout': 0.0, 'attn_logit_softcapping': None, 'dtype': 'bfloat16', 'final_logit_softcapping': None, 'head_dim': 256, 'hidden_activation': 'gelu_pytorch_tanh', 'hidden_size': 2560, 'initializer_range': 0.02, 'intermediate_size': 10240, 'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention'], 'max_position_embeddings': 131072, 'model_type': 'gemma3_text', 'num_attention_heads': 8, 'num_hidden_layers': 34, 'num_key_value_heads': 4, 'query_pre_attn_scalar': 256, 'rms_norm_eps': 1e-06, 'rope_local_base_freq': 10000.0, 'rope_scaling': {'factor': 8.0, 'rope_type': 'linear'}, 'rope_theta': 1000000.0, 'sliding_window': 1024, 'use_bidirectional_attention': False, 'use_cache': True, 'vocab_size': 262208}
TEXT_CONFIG_SHA256 = '0721b6853a6c84de76c8297ddceb5162cea0c6a017431ac4c411deb09c0f14bb'
PROMPTS = ('A red bicycle beside a blue wall.', 'A small wooden boat on a quiet lake.')
EXPECTED_TENSORS = 444
EXPECTED_LAYERS = 34
EXPECTED_HIDDEN_SIZE = 2560


class NotReady(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config():
    if config_digest(TEXT_CONFIG) != TEXT_CONFIG_SHA256:
        raise ValueError('Bundled publisher configuration checksum mismatch')
    expected = {'hidden_size': EXPECTED_HIDDEN_SIZE, 'num_hidden_layers': EXPECTED_LAYERS,
                'vocab_size': 262208, 'intermediate_size': 10240, 'num_attention_heads': 8,
                'num_key_value_heads': 4, 'head_dim': 256, 'model_type': 'gemma3_text'}
    if any(TEXT_CONFIG.get(key) != value for key, value in expected.items()):
        raise ValueError('Bundled Gemma architecture does not match the preserved checkpoint')


def normalize_keys(state):
    if 'spiece_model' not in state:
        raise ValueError('The embedded SentencePiece tokenizer is missing')
    if len(state) != EXPECTED_TENSORS + 1:
        raise ValueError('Unexpected tensor count in Gemma checkpoint')
    if any(not name.startswith('model.') for name in state if name != 'spiece_model'):
        raise ValueError('Unexpected Gemma checkpoint key; no parameters were ignored')
    return {name[len('model.'):]: value for name, value in state.items() if name != 'spiece_model'}


def validate_token_ids(ids, vocabulary):
    if not ids or len(ids) > 128 or any(type(value) is not int or not 0 <= value < vocabulary for value in ids):
        raise ValueError('The embedded tokenizer returned invalid or excessive token IDs')


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, document):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def persistent_paths(drive_root, output):
    mount = Path('/content/drive')
    if sys.platform != 'linux' or not os.path.ismount(mount) or not (mount / 'MyDrive').is_dir():
        raise NotReady('Mount Google Drive at /content/drive before running; no output folder was created')
    drive = drive_root.expanduser().resolve()
    if drive == mount / 'MyDrive' or not drive.is_relative_to(mount / 'MyDrive'):
        raise ValueError('Use a dedicated mounted Drive folder')
    model = drive / MODEL_RELATIVE
    destination = output or drive / 'qa/results' / ('gemma-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    destination = destination.expanduser().resolve()
    if not destination.is_relative_to(drive / 'qa/results') or destination == drive / 'qa/results':
        raise ValueError('Store each result under the mounted Drive qa/results directory')
    if model.is_symlink() or not model.resolve().is_relative_to(drive) or not model.is_file():
        raise NotReady('The verified Gemma model must already exist in the Drive models folder')
    if model.stat().st_size != MODEL_BYTES:
        raise NotReady('The Gemma model is incomplete or has a different file size')
    if destination.exists():
        raise ValueError('This result directory already exists; use a fresh directory')
    return drive, model, destination


def run_component(args, drive, model_path, output, report):
    # These are defensive; this script never calls from_pretrained or any client.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HOME'] = str(drive / 'cache/huggingface')
    os.environ['TORCH_HOME'] = str(drive / 'cache/torch')
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'safetensors', 'sentencepiece'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise NotReady('Run this harness with the installed Qwen runtime; a required package is missing') from None
    if versions['transformers'] != '5.17.0':
        raise NotReady('Use the existing Qwen runtime with Transformers 5.17.0; the main Forge runtime is not compatible with this harness')
    report['versions'] = versions
    report['stage'] = 'verify_model_checksum'
    atomic_json(output / 'results.json', report)
    if digest_file(model_path) != MODEL_SHA256:
        raise ValueError('Gemma weights do not match the verified publisher file')
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    import sentencepiece
    from transformers import Gemma3TextConfig, Gemma3TextModel

    if args.device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise NotReady('This CUDA device cannot execute the requested BF16 component check')
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
    report.update(stage='load_exact_weights', device=str(device), dtype=str(dtype),
                  model_sha256_verified=MODEL_SHA256)
    atomic_json(output / 'results.json', report)
    state = load_file(str(model_path), device='cpu')
    normalized = normalize_keys(state)
    tokenizer_tensor = state['spiece_model']
    if tokenizer_tensor.dtype != torch.uint8 or list(tokenizer_tensor.shape) != [4689074]:
        raise ValueError('Unexpected embedded tokenizer representation')
    tokenizer_bytes = tokenizer_tensor.numpy().tobytes()
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_bytes)
    report['embedded_tokenizer_sha256'] = hashlib.sha256(tokenizer_bytes).hexdigest()
    config = Gemma3TextConfig(**copy.deepcopy(TEXT_CONFIG))
    config._attn_implementation = 'eager'
    config.use_cache = False
    with init_empty_weights(include_buffers=False):
        model = Gemma3TextModel(config)
    expected_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    actual_shapes = {name: tuple(value.shape) for name, value in normalized.items()}
    if actual_shapes != expected_shapes:
        raise ValueError('Gemma tensor keys or shapes do not exactly match the publisher architecture')
    model.load_state_dict(normalized, strict=True, assign=True)
    del normalized, state, tokenizer_tensor, tokenizer_bytes
    if any(value.is_meta for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError('A Gemma parameter or buffer remained uninitialized')
    model = model.to(device=device, dtype=dtype).eval()
    report.update(stage='actual_forward', strict_loaded_tensors=len(expected_shapes),
                  architecture='Gemma3TextModel')
    atomic_json(output / 'results.json', report)
    counts = Counter()
    handles = []
    results, vectors = [], []
    def completed(_module, _inputs, value, label):
        hidden = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(hidden, torch.Tensor) or not hidden.numel() or not bool(torch.isfinite(hidden).all()):
            raise ValueError('A Gemma decoder layer returned an invalid tensor')
        counts[label] += 1
    for index, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(lambda module, inputs, value, label=index: completed(module, inputs, value, label)))
    try:
        with torch.inference_mode():
            for text in PROMPTS:
                ids = tokenizer.encode(text, out_type=int)
                if tokenizer.bos_id() >= 0:
                    ids.insert(0, tokenizer.bos_id())
                validate_token_ids(ids, config.vocab_size)
                before = dict(counts)
                started = time.monotonic()
                token_ids = torch.tensor([ids], dtype=torch.long, device=device)
                if not report['inference_attempted']:
                    report['inference_attempted'] = True
                    atomic_json(output / 'results.json', report)
                hidden = model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False, return_dict=True).last_hidden_state
                if tuple(hidden.shape) != (1, len(ids), EXPECTED_HIDDEN_SIZE) or not bool(torch.isfinite(hidden).all()):
                    raise ValueError('Gemma returned an invalid hidden-state shape or non-finite output')
                hidden_float = hidden.float()
                if float(hidden_float.std()) <= 1e-6:
                    raise ValueError('Gemma returned a constant hidden state')
                if any(counts[index] - before.get(index, 0) != 1 for index in range(EXPECTED_LAYERS)):
                    raise ValueError('Not all 34 Gemma decoder layers completed exactly one forward')
                last = hidden_float[0, -1].cpu().contiguous()
                vectors.append(last)
                results.append({'prompt': text, 'token_ids': ids, 'shape': list(hidden.shape),
                                'finite': True, 'std': float(hidden_float.std()), 'min': float(hidden_float.min()),
                                'max': float(hidden_float.max()), 'last_token_sha256': hashlib.sha256(last.numpy().tobytes()).hexdigest(),
                                'completed_decoder_layers': EXPECTED_LAYERS, 'seconds': round(time.monotonic() - started, 3)})
                report['prompts'] = results
                atomic_json(output / 'results.json', report)
                del hidden, hidden_float, token_ids
        if results[0]['token_ids'] == results[1]['token_ids'] or torch.allclose(vectors[0], vectors[1], rtol=1e-5, atol=1e-6):
            raise ValueError('Different prompts did not produce different final-token representations')
        import numpy as np
        np.savez_compressed(output / 'last-token-vectors.npz', prompt_1=vectors[0].numpy(), prompt_2=vectors[1].numpy())
        report.update(prompts=results, decoder_layer_completed_calls={str(i): counts[i] for i in range(EXPECTED_LAYERS)},
                      prompt_difference_l2=float(torch.linalg.vector_norm(vectors[0] - vectors[1])),
                      vector_evidence='last-token-vectors.npz', status='pass', stage='complete')
    finally:
        for handle in handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--plan', action='store_true')
    parser.add_argument('--drive-root', type=Path, default=DRIVE_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    validate_config()
    provenance = {'file': MODEL_RELATIVE, 'bytes': MODEL_BYTES, 'sha256': MODEL_SHA256,
                  'config_source': CONFIG_SOURCE, 'bundled_text_config_sha256': TEXT_CONFIG_SHA256,
                  'proof_scope': 'Standalone text encoder only; no NewBie image pipeline is tested'}
    if not args.run:
        print(json.dumps({'mode': 'plan_only_no_inference', 'provenance': provenance,
                          'checks': ['Exact checkpoint SHA256', '444 tensor keys and shapes', 'Embedded tokenizer',
                                     'Two real forwards through all 34 decoder layers', 'Finite nonconstant [1,tokens,2560] output',
                                     'Different prompt representations'], 'default_device': args.device}, indent=2))
        return 0
    if 'modules.shared' in sys.modules:
        parser.error('Use a new Qwen process; do not run this inside the Forge server process')
    try:
        drive, model, output = persistent_paths(args.drive_root, args.output)
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'not_ready', 'error': str(error), 'inference_attempted': False}))
        return 2
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'started_at': now(), 'provenance': provenance, 'inference_attempted': False}
    atomic_json(output / 'results.json', report)
    try:
        run_component(args, drive, model, output, report)
    except NotReady as error:
        report.update(status='not_ready', error=str(error))
    except Exception as error:
        report.update(status='fail', error_type=type(error).__name__, error=str(error))
    except BaseException:
        report.update(status='interrupted', error='The component check was interrupted')
        raise
    finally:
        report['finished_at'] = now()
        atomic_json(output / 'results.json', report)
    print(json.dumps({'status': report['status'], 'results': str(output / 'results.json'), 'proof_scope': provenance['proof_scope']}, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
