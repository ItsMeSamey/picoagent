"""Local HF model adapter using exactly the SFT harness protocol.

An untrained base model is not expected to produce valid tool JSON. This adapter
does not silently repair model responses or substitute oracle actions.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from picoagent.harness.protocol import END_MESSAGE, parse_assistant, render_messages


class HFPolicy:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        max_new_tokens: int = 512,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if not Path(model).is_dir() and not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
            raise ValueError("remote models require an exact 40-character revision")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(seed)
        self.seed = seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("HFPolicy currently supports cpu or cuda")
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        adapter = Path(model) / "adapter_config.json"
        if adapter.is_file():
            metadata_path = Path(model) / "picoagent_inference.json"
            if not metadata_path.is_file():
                raise ValueError("adapter lacks pinned picoagent inference metadata")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_revision = metadata.get("base_model_revision", "")
            if metadata.get("mode") != "qlora" or not re.fullmatch(r"[a-f0-9]{40}", base_revision):
                raise ValueError("adapter base must have an exact pinned revision")
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                metadata["base_model_id"], revision=base_revision,
                dtype=dtype, trust_remote_code=False,
            )
            self.model = PeftModel.from_pretrained(base, model).to(device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=dtype, trust_remote_code=False,
            ).to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.last_generation: dict[str, Any] = {}
        self._torch = torch

    def count_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        return len(self.tokenizer.encode(
            render_messages(messages, tools, add_generation_prompt=True),
            add_special_tokens=False,
        ))

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        prompt = render_messages(messages, tools, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        n_input = encoded["input_ids"].shape[-1]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and n_input + self.max_new_tokens > limit:
            raise ValueError("prompt plus generation exceeds model context; compact before generation")
        with self._torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
                stop_strings=[END_MESSAGE],
                tokenizer=self.tokenizer,
            )
        generated = output[0, n_input:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.last_generation = {
            "input_tokens": n_input,
            "output_tokens": int(generated.numel()),
            "raw_text": text,
        }
        return parse_assistant(text)
