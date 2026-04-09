import torch

from backend import memory_management


class QwenTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, max_length=512):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

    def __call__(self, texts, max_length=None):
        max_length = max_length or self.max_length
        device = memory_management.text_encoder_device()

        text_inputs = self.tokenizer(
            texts,
            padding=True,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Qwen tokenizer can return float empty tensors for empty prompts when
        # add_special_tokens=False. Force integer token/mask dtypes early.
        input_ids = text_inputs.input_ids.to(dtype=torch.long)
        attention_mask = text_inputs.attention_mask.to(dtype=torch.long)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 151643)

        # Some tokenizers can return a zero-width tensor for empty prompts when
        # add_special_tokens=False. Materialize one column to keep indexing valid.
        if input_ids.shape[1] == 0:
            batch = input_ids.shape[0]
            input_ids = torch.full((batch, 1), int(pad_id), dtype=torch.long)
            attention_mask = torch.zeros((batch, 1), dtype=torch.long)

        # Keep at least one visible token to avoid fully-masked rows.
        empty_rows = attention_mask.sum(dim=1) == 0
        if torch.any(empty_rows):
            input_ids[empty_rows, 0] = int(pad_id)
            attention_mask[empty_rows, 0] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        prompt_embeds = outputs.last_hidden_state
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.text_encoder.dtype)
        return prompt_embeds, attention_mask
