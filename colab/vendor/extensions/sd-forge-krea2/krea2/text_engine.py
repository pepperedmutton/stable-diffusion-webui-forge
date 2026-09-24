"""Krea 2 text conditioning for the local Forge Qwen3 implementation."""

import torch

KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_MAX_TOKENS = 512
KREA2_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
KREA2_TEMPLATE = KREA2_PREFIX + "{}" + KREA2_SUFFIX


class Krea2TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

        self.prefix_tokens = self._encode(KREA2_PREFIX)
        self.suffix_tokens = self._encode(KREA2_SUFFIX)
        if len(self.prefix_tokens) != 34 or len(self.suffix_tokens) != 5:
            raise RuntimeError(
                "Krea 2 requires the bundled Qwen tokenizer "
                "(expected a 34-token prefix and a 5-token suffix)"
            )
        self.body_token_limit = KREA2_MAX_TOKENS - len(self.suffix_tokens)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 151643

    def _encode(self, text):
        return list(
            self.tokenizer(
                str(text),
                add_special_tokens=False,
            )["input_ids"]
        )

    def tokenize(self, texts, truncation=False):
        result = []
        for text in texts:
            body = self._encode(str(text).strip())
            if truncation:
                body = body[:self.body_token_limit]
            result.append(body + self.suffix_tokens)
        return result

    def _build_batch(self, texts):
        sequences = []
        for text in texts:
            body = self._encode(str(text).strip())[:self.body_token_limit]
            sequences.append(self.prefix_tokens + body + self.suffix_tokens)

        if not sequences:
            raise ValueError("Krea 2 conditioning requires at least one prompt")

        batch_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), batch_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, :length] = 1
        return input_ids, attention_mask

    @torch.inference_mode()
    def __call__(self, texts):
        input_ids, attention_mask = self._build_batch(texts)
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= max(KREA2_TAP_LAYERS):
            raise RuntimeError("Krea 2 requires all 36 Qwen3-VL hidden layers")

        stacked = torch.stack(
            [hidden_states[index] for index in KREA2_TAP_LAYERS],
            dim=1,
        )
        prefix_length = len(self.prefix_tokens)
        stacked = stacked[:, :, prefix_length:, :]
        attention_mask = attention_mask[:, prefix_length:]

        batch, layers, sequence, width = stacked.shape
        prompt_embeds = (
            stacked.permute(0, 2, 1, 3)
            .reshape(batch, sequence, layers * width)
            .to(dtype=self.text_encoder.dtype)
        )
        return prompt_embeds, attention_mask
