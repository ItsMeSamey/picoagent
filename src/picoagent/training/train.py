"""Single-process Hugging Face full SFT, with an explicitly separate QLoRA mode."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .config import TrainingConfig, select_precision
from .checkpointing import CheckpointTimer
from .data import sha256_file, verify_dataset
from .encoding import AssistantOnlyCollator, encode_records
from .device import resolve_device
from .provenance import checkpoint_evidence, code_evidence, environment_evidence, now_utc, tree_hashes, verify_checkpoint, write_json, preflight_resume, cached_model_evidence


def _snapshot_code(output: Path, evidence: dict[str, Any]) -> None:
    package = Path(__file__).resolve().parents[1]
    project = package.parent.parent
    for relative in evidence["files"]:
        source = package / relative.removeprefix("src/picoagent/") if relative.startswith("src/picoagent/") else project / relative
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def run_training(config: TrainingConfig, *, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
    """Train locally. Never launch/lease a GPU, push a model, or evaluate a benchmark.

    Full mode updates every parameter using FP32 master weights plus CUDA BF16/
    FP16 autocast. This reduces activation precision, not Adam-state storage.
    QLoRA freezes a 4-bit base and trains adapters; it is never reported as full SFT.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This audited baseline supports one process/device; distributed training requires a separately validated config")
    initial_manifest_hash = sha256_file(config.dataset_manifest)
    manifest, rows = verify_dataset(config.dataset_manifest, allow_smoke=config.smoke_test,
                                    allow_native_teacher=config.allow_native_teacher_observed,
                                    allow_artificial_action_plans=config.allow_artificial_action_plans)
    if sha256_file(config.dataset_manifest) != initial_manifest_hash:
        raise ValueError("Dataset manifest changed during verification")
    if config.smoke_test != manifest.get("smoke_only", False):
        raise ValueError("smoke_test must exactly match the immutable dataset's smoke_only flag")
    output = Path(config.output_dir).resolve()
    if resume_from_checkpoint is None and output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be new or empty; use --resume for an existing audited run")
    if resume_from_checkpoint is not None:
        checkpoint = Path(resume_from_checkpoint).resolve()
        preflight_resume(output, checkpoint, max_steps=config.max_steps)
        if not (output / "run_manifest.json").is_file():
            raise ValueError("Resume requires the original run_manifest.json")
    try:
        import torch
        device, device_metadata = resolve_device(config.device, torch)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("Install picoagent[train] in your training environment first") from exc

    cuda = device == "cuda"
    bf16_supported = cuda and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    precision = select_precision(config.precision, cuda_available=cuda, bf16_supported=bf16_supported, xla_available=device == "xla")
    if config.training_mode == "qlora" and not cuda:
        raise ValueError("QLoRA in this baseline requires a supported CUDA GPU and picoagent[qlora]")
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
    set_seed(config.seed)
    code = code_evidence()
    config_payload = config.as_dict()
    manifest_hash = initial_manifest_hash
    environment = environment_evidence()
    hardware = {**device_metadata, "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(), "torch_version": torch.__version__}
    # Paths may move when an ephemeral runtime is restored. Preserve original
    # paths in provenance, but compare content identity and hyperparameters.
    identity_config = {key: value for key, value in config_payload.items() if key not in {"output_dir", "dataset_manifest"}}
    identity = {"config": identity_config, "dataset_manifest_sha256": manifest_hash,
                "source_tree_sha256": code["tree_sha256"], "precision": precision, "device": device,
                "hardware": hardware, "environment": environment}
    if resume_from_checkpoint:
        original = json.loads((output / "run_manifest.json").read_text())
        if original.get("identity") != identity:
            raise ValueError("Resume identity mismatch: config, source, dataset, hardware, precision and package versions must match the original run")
        verify_checkpoint(checkpoint, sha256_file(output / "run_manifest.json"))


    common = {"revision": config.model_revision, "trust_remote_code": False, "local_files_only": config.smoke_test}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, **common)
    if not tokenizer.is_fast:
        raise ValueError("The configured model must have a fast tokenizer for audited assistant-only loss")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an existing pad or EOS token; no silent vocabulary changes")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if manifest["schema"] == "picoagent.artificial_action_plan.dataset.v1":
        from huggingface_hub import hf_hub_download
        from picoagent.data.artificial_plans import validate_note_tokenizer
        tokenizer_json = hf_hub_download(config.model_id, "tokenizer.json",
                                         revision=config.model_revision, local_files_only=True, token=False)
        validate_note_tokenizer(config.dataset_manifest, tokenizer, model_id=config.model_id,
                                revision=config.model_revision, tokenizer_json_path=tokenizer_json)
    train_data, train_stats = encode_records(rows["train"], tokenizer, config.max_seq_length)
    dev_data, dev_stats = encode_records(rows["dev"], tokenizer, config.max_seq_length)
    load_args: dict[str, Any] = dict(common, dtype=torch.float32, attn_implementation="eager")
    if config.training_mode == "qlora":
        compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        load_args.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype), device_map={"": torch.cuda.current_device()})
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_args)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and config.max_seq_length > context_limit:
        raise ValueError(f"max_seq_length exceeds model context limit {context_limit}")
    model_cache = {} if config.smoke_test else cached_model_evidence(config.model_id, config.model_revision)
    resolved_revision = model_cache.get("resolved_revision")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.training_mode == "qlora":
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("QLoRA requires picoagent[qlora]") from exc
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=config.lora_rank,
            lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules="all-linear", bias="none"))
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if config.training_mode == "full" and trainable_parameters != total_parameters:
        raise ValueError("Full SFT invariant violated: frozen parameters exist")

    output.mkdir(parents=True, exist_ok=True)
    if not resume_from_checkpoint:
        _snapshot_code(output, code)
        (output / "dataset_manifest.json").write_bytes(Path(config.dataset_manifest).read_bytes())
        dataset_snapshot = output / "dataset_snapshot"
        if manifest["schema"] == "picoagent.artificial_action_plan.dataset.v1":
            from picoagent.data.artificial_plans import _copy_verified_plan_snapshot
            copied_manifest = _copy_verified_plan_snapshot(config.dataset_manifest, dataset_snapshot,
                verified_manifest=manifest, expected_manifest_sha256=manifest_hash)
            if sha256_file(copied_manifest) != manifest_hash:
                raise ValueError("Artificial action-plan evidence changed while snapshotting")
        elif manifest["schema"] == "picoagent.native_teacher.dataset.v1":
            from picoagent.data.native_admission import _copy_verified_native_snapshot
            copied_manifest = _copy_verified_native_snapshot(config.dataset_manifest, dataset_snapshot,
                verified_manifest=manifest, expected_manifest_sha256=manifest_hash)
            if sha256_file(copied_manifest) != manifest_hash:
                raise ValueError("Native evidence changed while creating run snapshot")
        else:
            dataset_snapshot.mkdir()
            shutil.copyfile(config.dataset_manifest, dataset_snapshot / "manifest.json")
            for split, entry in manifest["splits"].items():
                source = Path(config.dataset_manifest).resolve().parent / entry["path"]
                target = dataset_snapshot / entry["path"]
                shutil.copyfile(source, target)
                if sha256_file(target) != entry["sha256"]:
                    raise ValueError(f"Dataset changed while creating run snapshot: {split}")
                os.chmod(target, 0o444)
            os.chmod(dataset_snapshot / "manifest.json", 0o444)
        tokenizer.save_pretrained(output / "tokenizer_snapshot")
        evidence = {
            "schema": "picoagent.training.run.v1", "created_at": now_utc(), "identity": identity, "original_config": config_payload,
            "model": {"id": config.model_id, "requested_revision": config.model_revision,
                "resolved_revision": resolved_revision, "cached_assets": model_cache, "local_files": tree_hashes(Path(config.model_id)) if config.smoke_test else None},
            "mode": config.training_mode,
            "parameter_policy": "all_parameters_fp32_master_with_autocast" if config.training_mode == "full" else "frozen_nf4_base_trainable_lora_adapters",
            "parameter_counts": {"total": total_parameters, "trainable": trainable_parameters},
            "effective_batch_size": config.per_device_batch_size * config.gradient_accumulation_steps,
            "dataset": manifest, "tokenization": {"train": train_stats.as_dict(), "dev": dev_stats.as_dict()},
            "tokenizer_files": tree_hashes(output / "tokenizer_snapshot"), "code": code,
            "protocol": "picoagent-text-v1", "label_policy": "exact_model_event_response_only_when_events_exist_else_all_assistant_spans;tool_calls_included_observations_masked",
            "benchmark_evaluation": "not_run", "pretraining_contamination": "unknown",
            "limitations": ["Determinism is best effort across identical software and hardware; cross-device bitwise equivalence is not promised.",
                            "Synthetic trace provenance does not establish clean model pretraining or removal of intrinsic knowledge.",
                            "Development loss is not an agent success score or benchmark result."],
        }
        write_json(output / "run_manifest.json", evidence, exclusive=True)
        os.chmod(output / "run_manifest.json", 0o444)
    run_manifest_hash = sha256_file(output / "run_manifest.json")

    class AuditCheckpoint(TrainerCallback):
        def __init__(self):
            self.timer = CheckpointTimer(config.checkpoint_interval_seconds)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.timer.mark_saved()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # HF calls this after a completed optimizer step, never mid-update.
            if self.timer.due():
                control.should_save = True
                control.should_evaluate = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.is_world_process_zero:
                checkpoint_evidence(output / f"checkpoint-{state.global_step}", run_manifest_hash)
            self.timer.mark_saved()
            return control

    arguments = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, num_train_epochs=config.num_train_epochs, max_steps=config.max_steps,
        warmup_steps=config.warmup_ratio, lr_scheduler_type="cosine", optim="adamw_torch",
        bf16=precision == "bf16" and device != "xla", fp16=precision == "fp16", use_cpu=device == "cpu",
        seed=config.seed, data_seed=config.seed, full_determinism=config.deterministic,
        logging_steps=config.logging_steps, logging_strategy="steps", logging_nan_inf_filter=False,
        eval_strategy="steps", eval_steps=config.save_steps, prediction_loss_only=True,
        save_strategy="steps", save_steps=config.save_steps, save_total_limit=None,
        save_only_model=False, load_best_model_at_end=False, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_num_workers=0,
        dataloader_pin_memory=cuda, remove_unused_columns=False, report_to=[], push_to_hub=False,
    )
    class AuditedTrainer(Trainer):
        def compute_loss_context_manager(self):
            # Avoid Accelerate's legacy global XLA_USE_BF16 casting: keep master
            # weights and optimizer state FP32, autocast forward/loss only.
            if device == "xla" and precision == "bf16":
                return torch.autocast("xla", dtype=torch.bfloat16)
            return super().compute_loss_context_manager()

    trainer = AuditedTrainer(model=model, args=arguments, train_dataset=train_data, eval_dataset=dev_data,
                      processing_class=tokenizer, data_collator=AssistantOnlyCollator(tokenizer.pad_token_id, pad_to_length=config.max_seq_length if device == "xla" else None),
                      callbacks=[AuditCheckpoint()])
    if trainer.args.device.type != device:
        raise RuntimeError(f"Trainer selected {trainer.args.device}, expected {device}; refusing silent backend fallback")
    write_json(output / "run_status.json", {"status": "running", "started_at": now_utc(), "resume_from": resume_from_checkpoint})
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = {"training": result.metrics, "development": trainer.evaluate(), "benchmark": None}
        final_dir = output / ("final-model" if config.training_mode == "full" else "final-adapter")
        model.config.use_cache = True
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        trainer.save_state()
        write_json(final_dir / "picoagent_inference.json", {
            "protocol": "picoagent-text-v1", "mode": config.training_mode,
            "base_model_id": config.model_id, "base_model_revision": config.model_revision,
            "run_manifest_sha256": run_manifest_hash, "smoke_only": config.smoke_test,
        }, exclusive=True)
        write_json(output / "metrics.json", metrics)
        write_json(output / "final_artifacts.json", {"path": final_dir.name, "files": tree_hashes(final_dir)}, exclusive=True)
        write_json(output / "run_status.json", {"status": "completed", "finished_at": now_utc(), "global_step": trainer.state.global_step,
                   "mode": config.training_mode, "smoke_only": config.smoke_test, "artifact": str(final_dir), "benchmark_evaluation": "not_run"})
        return {"output_dir": str(output), "artifact": str(final_dir), "training_mode": config.training_mode,
                "precision": precision, "device": device, "trainable_parameters": trainable_parameters, "metrics": metrics, "smoke_only": config.smoke_test}
    except BaseException as exc:
        write_json(output / "run_status.json", {"status": "failed", "finished_at": now_utc(), "error_type": type(exc).__name__, "error": str(exc)})
        raise
