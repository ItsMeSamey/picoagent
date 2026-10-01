"""Offline, CPU-only plumbing check using a newly initialized random model.

The illustrative tool result below is deliberately labeled unexecuted. This
function does not produce an agent benchmark result or a usable trained agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .data import canonical_json, prepare_dataset
from .train import run_training


def run_smoke(output_dir: str | Path, *, device: str = "cpu", segment_steps: int | None = None,
              train_records: int = 2, gradient_accumulation_steps: int = 2,
              save_steps: int = 1, checkpoint_interval_seconds: float | None = 300.0,
              max_steps: int = 2, eval_steps: int | None = None,
              checkpoint_before_eval: bool = False, continue_through_checkpoints: bool = False,
              dropout: float = 0.0) -> dict[str, Any]:
    if device not in {"cpu", "cuda", "xla"}:
        raise ValueError("Smoke device must be explicit: cpu, cuda, or xla")
    if type(train_records) is not int or train_records < 2:
        raise ValueError("Smoke train_records must be an integer of at least two")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps <= 0:
        raise ValueError("Smoke gradient_accumulation_steps must be a positive integer")
    if type(save_steps) is not int or save_steps <= 0:
        raise ValueError("Smoke save_steps must be a positive integer")
    import os
    if device != "xla":
        os.environ["USE_TORCH_XLA"] = "0"
    try:
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
    except ImportError as exc:
        raise RuntimeError("Install picoagent[train] to run the optional CPU smoke check") from exc
    from picoagent.harness.protocol import render_messages
    from picoagent.harness.tools import TOOL_SCHEMAS

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    set_seed(7)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    dev_number = 4 if train_records == 2 else 99
    records = [("train", number) for number in range(2, 2 + train_records)] + [("dev", dev_number)]
    for split, number in records:
        call_id = f"smoke-call-{number}"
        rows[split].append({
            "schema_version": 1, "trace_id": f"smoke-trace-{number}", "task_id": f"smoke-task-{number}",
            "family": f"smoke-{split}", "template_id": f"smoke-template-{split}", "split": split,
            "status": "unexecuted", "provenance": {"source": "pipeline_smoke", "execution": "unexecuted_fixture"},
            "verification": {"passed": False},
            "messages": [
                {"role": "system", "content": "CPU plumbing fixture only. Use Python and return the result."},
                {"role": "user", "content": f"Compute {number} + 1."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": f"print({number} + 1)"})}}]},
                {"role": "tool", "tool_call_id": call_id, "content": str(number + 1)},
                {"role": "assistant", "content": str(number + 1)},
            ],
        })
    raw_paths = {}
    for split, records in rows.items():
        path = root / f"raw-{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
        raw_paths[split] = path
    manifest = prepare_dataset(raw_paths["train"], raw_paths["dev"], root / "dataset", smoke_only=True)
    corpus = [render_messages(row["messages"], tools=TOOL_SCHEMAS) for records in rows.values() for row in records]
    tokenizer_core = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_core.decoder = decoders.ByteLevel()
    tokenizer_core.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=384,
        special_tokens=["[UNK]", "[PAD]", "[EOS]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]", model_max_length=2048)
    local_model = root / "random-initial-model"
    tokenizer.save_pretrained(local_model)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=2048, n_ctx=2048,
        n_embd=32, n_layer=1, n_head=2, bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))
    model.save_pretrained(local_model)
    config = TrainingConfig(model_id=str(local_model), model_revision=None, dataset_manifest=str(manifest),
        output_dir=str(root / "run"), max_seq_length=2048, per_device_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps, gradient_checkpointing=True, learning_rate=1e-3,
        max_steps=max_steps, save_steps=save_steps, checkpoint_interval_seconds=checkpoint_interval_seconds,
        eval_steps=eval_steps, checkpoint_before_eval=checkpoint_before_eval,
        logging_steps=1, precision="auto", device=device, seed=7, smoke_test=True)
    (root / "smoke-config.json").write_text(canonical_json(config.as_dict()) + "\n")
    result = run_training(config, segment_steps=segment_steps,
                          continue_through_checkpoints=continue_through_checkpoints)
    result["interpretation"] = "Random model pipeline smoke, synthetic unexecuted fixture, pipeline validation only; no agent capability or benchmark claim"
    return result
