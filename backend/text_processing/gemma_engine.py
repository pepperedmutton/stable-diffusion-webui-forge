import torch

from backend import memory_management


class GemmaTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, max_length=256):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

    def tokenize(self, texts, max_length=None):
        max_length = max_length or self.max_length
        return self.tokenizer(
            texts,
            truncation=False,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]

    def __call__(self, texts, max_length=None):
        max_length = max_length or self.max_length
        device = memory_management.text_encoder_device()

        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        prompt_embeds = outputs.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.text_encoder.dtype)

        return prompt_embeds, attention_mask
