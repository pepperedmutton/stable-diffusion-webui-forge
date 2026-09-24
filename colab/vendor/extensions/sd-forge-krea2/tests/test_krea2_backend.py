import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.argv[:] = [sys.argv[0]]
EXTENSION_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = Path(__file__).resolve().parents[3]
for entry in (
    FORGE_ROOT,
    EXTENSION_ROOT,
    FORGE_ROOT / "packages_3rdparty",
    FORGE_ROOT / "repositories" / "huggingface_guess",
):
    sys.path.insert(0, str(entry))

import torch

from backend import attention as forge_attention
from backend.modules.k_prediction import PredictionFlux
import krea2.dit as dit
import krea2.engine as engine
import scripts.krea2_register as registration


def _write_safetensors_header(path, tensors, metadata=None):
    header = dict(tensors)
    if metadata is not None:
        header["__metadata__"] = metadata
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)


def _tensor(shape):
    return {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}


class ScheduleAndEngineTests(unittest.TestCase):
    def test_prediction_flux_matches_official_mu_formula(self):
        mu = 1.15
        predictor = PredictionFlux(mu=mu, pseudo_timestep_range=10000)
        for timestep in (0.1, 0.5, 0.9):
            index = int(timestep * 10000) - 1
            expected = math.exp(mu) / (math.exp(mu) + (1.0 / timestep - 1.0))
            self.assertAlmostEqual(predictor.sigmas[index].item(), expected, places=6)

    def test_engine_builds_prediction_flux_with_mu(self):
        captured = []

        class DummyClip:
            def __init__(self, **_kwargs):
                self.cond_stage_model = SimpleNamespace(gemma2_2b=object())
                self.tokenizer = SimpleNamespace(gemma2_2b=object())

        class DummyTextEngine:
            def __init__(self, **_kwargs):
                pass

        def predictor_factory(**kwargs):
            captured.append(kwargs)
            return PredictionFlux(**kwargs)

        config = SimpleNamespace(
            sampling_settings={"mu": 1.15},
            inpaint_model=lambda: False,
        )
        components = {
            "text_encoder": object(),
            "tokenizer": object(),
            "vae": object(),
            "transformer": object(),
        }
        with (
            mock.patch.object(engine, "CLIP", DummyClip),
            mock.patch.object(engine, "VAE", lambda **_kwargs: object()),
            mock.patch.object(engine, "Krea2TextProcessingEngine", DummyTextEngine),
            mock.patch.object(engine, "PredictionFlux", predictor_factory),
            mock.patch.object(engine.UnetPatcher, "from_model", return_value=object()),
        ):
            engine.Krea2(config, components)
        self.assertEqual(captured, [{"mu": 1.15}])

    def test_model_registration_uses_flux_type_and_mu(self):
        base = engine.Krea2.matched_guesses[0]
        self.assertEqual(base.sampling_settings, {"mu": 1.15})
        self.assertIs(
            base.model_type(None, {}),
            registration.model_list.ModelType.FLUX
            if hasattr(registration, "model_list")
            else __import__("huggingface_guess.model_list", fromlist=["ModelType"]).ModelType.FLUX,
        )

    def test_dimensions_round_up_to_sixteen(self):
        self.assertEqual(engine.Krea2.fix_dimensions(832, 1248), (832, 1248))
        self.assertEqual(engine.Krea2.fix_dimensions(833, 1249), (848, 1264))
        self.assertEqual(engine.Krea2.fix_dimensions(1, 0), (16, 16))


class AttentionMaskTests(unittest.TestCase):
    def test_backend_specific_mask_shapes(self):
        padding = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        xformers = dit._prepare_attention_mask(
            padding, 3, 4, torch.float32, "cpu", "attention_xformers"
        )
        pytorch = dit._prepare_attention_mask(
            padding, 3, 4, torch.float32, "cpu", "attention_pytorch"
        )
        basic = dit._prepare_attention_mask(
            padding, 3, 4, torch.float32, "cpu", "attention_basic"
        )
        self.assertEqual(tuple(xformers.bias.shape), (6, 4, 4))
        self.assertEqual(tuple(pytorch.bias.shape), (2, 1, 4, 4))
        self.assertEqual(tuple(basic.bias.shape), (2, 1, 4, 4))
        self.assertEqual(xformers.bias[0, 0, 0].item(), 0.0)
        self.assertEqual(xformers.bias[0, 0, 2].item(), -torch.finfo(torch.float32).max)

    def test_full_valid_mask_has_no_allocation(self):
        result = dit._prepare_attention_mask(
            torch.ones(2, 512), 48, 512, torch.bfloat16, "cpu", "attention_xformers"
        )
        self.assertIsNone(result)

    def test_large_xformers_mask_uses_pytorch_backend(self):
        padding = torch.tensor([[1, 1, 0], [1, 1, 1]])
        with mock.patch.object(dit, "_XFORMERS_MASK_LIMIT_BYTES", 1):
            result = dit._prepare_attention_mask(
                padding, 2, 3, torch.float32, "cpu", "attention_xformers"
            )
        self.assertEqual(result.backend_name, "attention_pytorch")
        self.assertEqual(tuple(result.bias.shape), (2, 1, 3, 3))

    def _assert_padding_isolation(self, backend, device, dtype, dim, heads, kvheads, tolerance):
        previous = dit.attention_function
        dit.attention_function = backend
        try:
            torch.manual_seed(9)
            layer = dit.Attention(
                dim,
                heads,
                kvheads=kvheads,
                operations=torch.nn,
                device=device,
                dtype=dtype,
            )
            for parameter in layer.parameters():
                torch.nn.init.uniform_(parameter, -0.02, 0.02)

            short = torch.randn(1, 3, dim, device=device, dtype=dtype)
            batch = torch.randn(2, 5, dim, device=device, dtype=dtype)
            batch[0, :3] = short[0]
            batch[0, 3:] = 100
            padding = torch.tensor(
                [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
                device=device,
                dtype=torch.bool,
            )
            with torch.inference_mode():
                expected = layer(short, mask=torch.ones(1, 3, device=device, dtype=torch.bool))
                actual = layer(batch, mask=padding)
            torch.testing.assert_close(
                actual[0, :3].cpu(),
                expected[0].cpu(),
                rtol=tolerance,
                atol=tolerance,
            )
            self.assertEqual(actual.shape[0], 2)
            self.assertEqual(heads, layer.heads)
        finally:
            dit.attention_function = previous

    def test_pytorch_batch_padding_isolation(self):
        self._assert_padding_isolation(
            forge_attention.attention_pytorch,
            "cpu",
            torch.float32,
            16,
            4,
            2,
            1e-5,
        )

    def test_basic_batch_padding_isolation(self):
        self._assert_padding_isolation(
            forge_attention.attention_basic,
            "cpu",
            torch.float32,
            16,
            4,
            2,
            1e-5,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the xformers test")
    def test_xformers_batch_padding_isolation(self):
        if not hasattr(forge_attention, "attention_xformers"):
            self.skipTest("xformers is not available")
        self._assert_padding_isolation(
            forge_attention.attention_xformers,
            "cuda",
            torch.float16,
            128,
            2,
            1,
            2e-3,
        )


class ComponentDiscoveryTests(unittest.TestCase):
    def test_discovery_uses_tensor_structure_not_filename(self):
        valid_text = {
            "model.embed_tokens.weight": _tensor((151936, 2560)),
            "model.layers.0.self_attn.q_norm.weight": _tensor((128,)),
            "model.layers.35.self_attn.q_norm.weight": _tensor((128,)),
            "model.layers.35.mlp.down_proj.weight": _tensor((2560, 9728)),
        }
        valid_vae = {
            "encoder.conv1.weight": _tensor((96, 3, 3, 3, 3)),
            "decoder.conv1.weight": _tensor((384, 16, 3, 3, 3)),
            "decoder.head.2.weight": _tensor((3, 96, 3, 3, 3)),
            "conv1.weight": _tensor((32, 32, 1, 1, 1)),
        }
        wrong = {"decoder.conv_in.weight": _tensor((512, 16, 3, 3))}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_dir = root / "text_encoder"
            vae_dir = root / "VAE"
            text_dir.mkdir()
            vae_dir.mkdir()
            _write_safetensors_header(
                text_dir / "qwen3vl_4b_fp8_scaled.safetensors",
                wrong,
                {"modelspec.architecture": "Flux.1"},
            )
            _write_safetensors_header(
                text_dir / "renamed_encoder.safetensors",
                valid_text,
                {"modelspec.architecture": "Qwen3-VL"},
            )
            _write_safetensors_header(
                vae_dir / "qwen_image_vae.safetensors",
                wrong,
                {"modelspec.architecture": "Flux.1-AE"},
            )
            _write_safetensors_header(
                vae_dir / "renamed_vae.safetensors",
                valid_vae,
                {"modelspec.architecture": "Qwen Image VAE"},
            )

            from modules import paths, shared

            command_options = SimpleNamespace(text_encoder_dirs=[], vae_dirs=[])
            with (
                mock.patch.object(paths, "models_path", str(root)),
                mock.patch.object(shared, "cmd_opts", command_options),
            ):
                found = registration._find_krea2_modules()

        self.assertEqual(
            [os.path.basename(path) for path in found],
            ["renamed_encoder.safetensors", "renamed_vae.safetensors"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
