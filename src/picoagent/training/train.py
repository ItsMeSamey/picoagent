"""Single-process Hugging Face full SFT, with an explicitly separate QLoRA mode."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import TrainingConfig, select_precision
from .checkpointing import CheckpointTimer, save_checkpoint_transaction
from .data import sha256_file, verify_dataset
from .encoding import AssistantOnlyCollator, encode_records
from .device import resolve_device
from .evaluation import validate_evaluation
from .provenance import checkpoint_evidence, code_evidence, environment_evidence, now_utc, tree_hashes, verify_checkpoint, write_json, preflight_resume, cached_model_evidence


def _snapshot_code(output: Path, evidence: dict[str, Any]) -> None:
    package = Path(__file__).resolve().parents[1]
    project = package.parent.parent
    for relative in evidence["files"]:
        source = package / relative.removeprefix("src/picoagent/") if relative.startswith("src/picoagent/") else project / relative
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Output tree contains a symlink; refusing bounded-space accounting: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _output_budget_reservation(output: Path, *, parameters: int, budget_bytes: int,
                              budget_root: Path | None = None) -> dict[str, int]:
    """Conservatively reserve room for the next full checkpoint and final model.

    For full SFT, one full checkpoint needs FP32 model weights and Adam's two
    FP32 moment tensors. A 3.25x parameter-byte estimate plus a 512 MiB margin
    covers checkpoint metadata and serialization overhead; a separate model
    copy is reserved for final export. This is a preflight, not a promise about
    provider quota accounting.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError("output_budget_bytes must be a positive integer")
    parameter_bytes = parameters * 4
    checkpoint_reserve = math.ceil(parameter_bytes * 3.25) + 512 * 1024 * 1024
    final_model_reserve = parameter_bytes
    margin = 1024 * 1024 * 1024
    counted_root = (budget_root or output).resolve(strict=True)
    output = output.resolve(strict=True)
    if output != counted_root and not output.is_relative_to(counted_root):
        raise ValueError("output budget root must contain the training output directory")
    existing_bytes = _tree_bytes(counted_root)
    projected = existing_bytes + checkpoint_reserve + final_model_reserve + margin
    if projected > budget_bytes:
        raise OSError(
            "Insufficient bounded output budget for restored/current artifacts, "
            "one full checkpoint, final model export, and safety margin "
            f"({existing_bytes}+{checkpoint_reserve}+{final_model_reserve}+{margin} > {budget_bytes} bytes)"
        )
    free_bytes = shutil.disk_usage(counted_root).free
    needed_free = checkpoint_reserve + final_model_reserve + margin
    if free_bytes < needed_free:
        raise OSError(
            "Insufficient filesystem free space for one full checkpoint, "
            f"final model export, and safety margin ({free_bytes} < {needed_free} bytes)"
        )
    return {"existing_output_bytes": existing_bytes, "checkpoint_reserve_bytes": checkpoint_reserve,
            "final_model_reserve_bytes": final_model_reserve, "safety_margin_bytes": margin,
            "projected_output_bytes": projected, "output_budget_bytes": budget_bytes,
            "filesystem_free_bytes": free_bytes}


def _evaluate_preserving_rng(trainer: Any, evaluate: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run evaluation without changing the next training step's RNG stream."""
    with tempfile.TemporaryDirectory(prefix="picoagent-eval-rng-") as rng_dir:
        trainer._save_rng_state(rng_dir)
        try:
            return (evaluate or trainer.evaluate)()
        finally:
            trainer._load_rng_state(rng_dir)


def run_training(config: TrainingConfig, *, resume_from_checkpoint: str | None = None,
                 segment_steps: int | None = None, output_budget_bytes: int | None = None,
                 output_budget_root: str | None = None,
                 continue_through_checkpoints: bool = False,
                 durability_timeout_seconds: float | None = None,
                 finalize_only: bool = False) -> dict[str, Any]:
    """Train locally. Never launch/lease a GPU, push a model, or evaluate a benchmark.

    Full mode updates every parameter using FP32 master weights plus CUDA BF16/
    FP16 autocast. This reduces activation precision, not Adam-state storage.
    QLoRA freezes a 4-bit base and trains adapters; it is never reported as full SFT.

    ``segment_steps`` is an operational stop boundary, not part of the
    immutable TrainingConfig or optimizer schedule. Each segment stops only
    after a full checkpoint has been sealed. The same config and runtime
    identity are required to resume the next segment.
    """
    if type(finalize_only) is not bool:
        raise ValueError("finalize_only must be a boolean")
    if finalize_only and (resume_from_checkpoint is None or segment_steps is not None):
        raise ValueError("finalize_only requires --resume and cannot use segment_steps")
    if durability_timeout_seconds is not None and (
        isinstance(durability_timeout_seconds, bool)
        or not isinstance(durability_timeout_seconds, (int, float))
        or not math.isfinite(durability_timeout_seconds) or durability_timeout_seconds <= 0
    ):
        raise ValueError("durability_timeout_seconds must be positive finite seconds")
    if durability_timeout_seconds is not None and not config.checkpoint_before_eval:
        raise ValueError("durability_timeout_seconds requires checkpoint_before_eval")
    if segment_steps is not None and (type(segment_steps) is not int or segment_steps <= 0):
        raise ValueError("segment_steps must be a positive integer when supplied")
    if type(continue_through_checkpoints) is not bool:
        raise ValueError("continue_through_checkpoints must be a boolean")
    if continue_through_checkpoints and segment_steps is None:
        raise ValueError("continue_through_checkpoints requires an explicit segmented run")
    if output_budget_bytes is not None and segment_steps is None and not finalize_only:
        raise ValueError("output_budget_bytes requires an explicit segmented run")
    if output_budget_bytes is not None and (type(output_budget_bytes) is not int or output_budget_bytes <= 0):
        raise ValueError("output_budget_bytes must be a positive integer")
    if output_budget_root is not None and output_budget_bytes is None:
        raise ValueError("output_budget_root requires output_budget_bytes")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This audited baseline supports one process/device; distributed training requires a separately validated config")
    initial_manifest_hash = sha256_file(config.dataset_manifest)
    prepared = None
    if config.prepared_manifest is not None:
        from .prepared import load_prepared_dataset
        prepared = load_prepared_dataset(config)
        manifest, rows = prepared.source_manifest, None
    else:
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
        preflight_resume(output, checkpoint, max_steps=config.max_steps, finalize_only=finalize_only)
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
    identity_config = {key: value for key, value in config_payload.items() if key not in {"output_dir", "dataset_manifest", "prepared_manifest"}}
    identity = {"config": identity_config, "dataset_manifest_sha256": manifest_hash,
                "source_tree_sha256": code["tree_sha256"], "precision": precision, "device": device,
                "hardware": hardware, "environment": environment}
    if prepared is not None:
        identity["prepared_tokens"] = prepared.identity
    if resume_from_checkpoint:
        original = json.loads((output / "run_manifest.json").read_text())
        durability_required = original.get("durability_required", False)
        if type(durability_required) is not bool:
            raise ValueError("Invalid run manifest durability requirement")
        if durability_required and durability_timeout_seconds is None:
            raise ValueError("This run requires --durability-timeout-seconds on every resume")
        if not durability_required and durability_timeout_seconds is not None:
            raise ValueError("Cannot add a durability requirement to an immutable legacy run; create a new run")
        if original.get("identity") != identity:
            raise ValueError("Resume identity mismatch: config, source, dataset, hardware, precision and package versions must match the original run")
        verify_checkpoint(checkpoint, sha256_file(output / "run_manifest.json"))
        if durability_timeout_seconds is not None:
            from .durability import wait_for_durable_ack
            wait_for_durable_ack(output, checkpoint, durability_timeout_seconds)


    common = {"revision": config.model_revision, "trust_remote_code": False, "local_files_only": config.smoke_test}
    if prepared is not None:
        tokenizer = AutoTokenizer.from_pretrained(prepared.path.parent / "tokenizer", use_fast=True,
                                                  trust_remote_code=False, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, **common)
    if not tokenizer.is_fast:
        raise ValueError("The configured model must have a fast tokenizer for audited assistant-only loss")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an existing pad or EOS token; no silent vocabulary changes")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if prepared is None and manifest["schema"] == "picoagent.artificial_action_plan.dataset.v1":
        from huggingface_hub import hf_hub_download
        from picoagent.data.artificial_plans import validate_note_tokenizer
        tokenizer_json = hf_hub_download(config.model_id, "tokenizer.json",
                                         revision=config.model_revision, local_files_only=True, token=False)
        validate_note_tokenizer(config.dataset_manifest, tokenizer, model_id=config.model_id,
                                revision=config.model_revision, tokenizer_json_path=tokenizer_json)
    if prepared is not None:
        from .prepared import validate_training_tokenizer
        validate_training_tokenizer(prepared, tokenizer)
        train_data, dev_data = prepared.datasets["train"], prepared.datasets["dev"]
        train_stats, dev_stats = prepared.stats["train"], prepared.stats["dev"]
    else:
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
        if prepared is not None:
            from .prepared import copy_prepared_snapshots
            copy_prepared_snapshots(prepared, source_destination=dataset_snapshot,
                                    prepared_destination=output / "prepared_snapshot")
        elif manifest["schema"] == "picoagent.artificial_action_plan.dataset.v1":
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
            "durability_required": durability_timeout_seconds is not None,
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

    trainer_holder: dict[str, Any] = {}

    class AuditCheckpoint(TrainerCallback):
        def __init__(self):
            self.timer = CheckpointTimer(config.checkpoint_interval_seconds)
            self.start_step: int | None = None
            self.boundary_step: int | None = None
            self.segment_checkpoint: str | None = None
            self.last_checkpoint: str | None = None
            self.new_checkpoints: list[str] = []
            self.evaluate_after_save = False

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.start_step = state.global_step
            if segment_steps is not None:
                self.boundary_step = min(state.max_steps, state.global_step + segment_steps)
            self.timer.mark_saved()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            at_boundary = self.boundary_step is not None and state.global_step >= self.boundary_step
            timer_due = self.timer.due()
            # Prove the full-size off-runtime backup before substantial training.
            # Subsequent saves keep the configured step/timer cadence unchanged.
            if durability_timeout_seconds is not None and state.global_step == 1:
                control.should_save = True
            if timer_due and not at_boundary:
                control.should_save = True
                if not config.checkpoint_before_eval:
                    # Preserve legacy behavior unless the frozen config opts
                    # into save-before-evaluate with an independent cadence.
                    control.should_evaluate = True
            if at_boundary:
                control.should_save = True
                control.should_training_stop = True
            if config.checkpoint_before_eval and control.should_save and control.should_evaluate:
                self.evaluate_after_save = True
                control.should_evaluate = False
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.is_world_process_zero:
                checkpoint = output / f"checkpoint-{state.global_step}"
                checkpoint_evidence(checkpoint, run_manifest_hash)
                if durability_timeout_seconds is not None:
                    from .durability import wait_for_durable_ack
                    wait_for_durable_ack(output, checkpoint, durability_timeout_seconds)
                self.last_checkpoint = checkpoint.name
                self.new_checkpoints.append(checkpoint.name)
                at_boundary = self.boundary_step is not None and state.global_step >= self.boundary_step
                pause_first = segment_steps is not None and not continue_through_checkpoints
                if pause_first or at_boundary:
                    self.segment_checkpoint = checkpoint.name
                    # Trainer reaches this callback only after weights,
                    # optimizer, scheduler, RNG, and trainer state are saved.
                    control.should_training_stop = True
            self.timer.mark_saved()
            if self.evaluate_after_save:
                self.evaluate_after_save = False
                if not state.is_world_process_zero:
                    raise RuntimeError("Checkpoint-before-evaluate currently requires the single-process baseline")
                checkpoint = output / f"checkpoint-{state.global_step}"
                checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
                metrics = trainer_holder["trainer"].evaluate()
                evaluation_dir = output / "evaluations"
                evaluation_dir.mkdir(exist_ok=True)
                evaluation = {
                    "schema": "picoagent.training.evaluation.v1",
                    "checkpoint": checkpoint.name,
                    "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
                    "global_step": state.global_step,
                    "metrics": metrics,
                }
                validate_evaluation(evaluation, checkpoint)
                write_json(evaluation_dir / f"{checkpoint.name}.json", evaluation, exclusive=True)
            return control

    # The global step/epoch target is still calculated from the original
    # config. Segmentation changes only where this invocation pauses and saves.
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
        eval_strategy="steps", eval_steps=config.eval_steps or config.save_steps, prediction_loss_only=True,
        save_strategy="steps", save_steps=config.save_steps, save_total_limit=None,
        save_only_model=False, load_best_model_at_end=False, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_num_workers=0,
        dataloader_pin_memory=cuda, remove_unused_columns=False, report_to=[], push_to_hub=False,
    )
    class AuditedTrainer(Trainer):
        def _save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
            nonlocal space
            if output_budget_bytes is not None:
                # Check at the actual write boundary, even if legacy evaluation
                # took a long time after on_step_end. Export chunks under the
                # chosen root count against the same bound on every save.
                space = _output_budget_reservation(
                    output, parameters=total_parameters, budget_bytes=output_budget_bytes,
                    budget_root=Path(output_budget_root) if output_budget_root else None)
            save = super()._save_checkpoint
            original_output = self.args.output_dir

            def save_staged(staging_root: Path) -> None:
                self.args.output_dir = str(staging_root)
                try:
                    save(*args, **kwargs)
                finally:
                    self.args.output_dir = original_output

            save_checkpoint_transaction(output, self.state.global_step, save_staged, run_manifest_hash)

        def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # In the opt-in policy, *every* evaluation is isolated, including
            # independent eval-only steps. Otherwise an interruption before a
            # deferred eval could change the resumed training RNG stream.
            evaluate = super().evaluate
            if config.checkpoint_before_eval:
                metrics = _evaluate_preserving_rng(self, lambda: evaluate(*args, **kwargs))
                trainer_holder["last_evaluation"] = (self.state.global_step, dict(metrics))
                return metrics
            return evaluate(*args, **kwargs)

        def compute_loss_context_manager(self):
            # Avoid Accelerate's legacy global XLA_USE_BF16 casting: keep master
            # weights and optimizer state FP32, autocast forward/loss only.
            if device == "xla" and precision == "bf16":
                return torch.autocast("xla", dtype=torch.bfloat16)
            return super().compute_loss_context_manager()

    checkpoint_callback = AuditCheckpoint()
    trainer = AuditedTrainer(model=model, args=arguments, train_dataset=train_data, eval_dataset=dev_data,
                      processing_class=tokenizer, data_collator=AssistantOnlyCollator(tokenizer.pad_token_id, pad_to_length=config.max_seq_length if device == "xla" else None),
                      callbacks=[checkpoint_callback])
    trainer_holder["trainer"] = trainer
    if trainer.args.device.type != device:
        raise RuntimeError(f"Trainer selected {trainer.args.device}, expected {device}; refusing silent backend fallback")
    space = None
    if output_budget_bytes is not None:
        space = _output_budget_reservation(output, parameters=total_parameters, budget_bytes=output_budget_bytes,
                                           budget_root=Path(output_budget_root) if output_budget_root else None)
    before_checkpoints = {path.name for path in output.glob("checkpoint-[0-9]*") if path.is_dir()}
    write_json(output / "run_status.json", {
        "status": "running", "started_at": now_utc(), "resume_from": resume_from_checkpoint,
        "segment_limit_optimizer_steps": segment_steps, "continue_through_checkpoints": continue_through_checkpoints,
        "output_space_preflight": space, "finalize_only": finalize_only,
    })
    try:
        if finalize_only:
            # Never call Trainer.train here: some versions perform an extra
            # optimizer step when resumed at the nominal end of the schedule.
            from transformers.trainer_callback import TrainerState
            trainer._load_from_checkpoint(str(checkpoint))
            trainer.state = TrainerState.load_from_json(str(checkpoint / "trainer_state.json"))
            trainer._load_rng_state(str(checkpoint))
            training_metrics = {"finalize_only": True, "optimizer_updates_this_invocation": 0,
                                "global_step": trainer.state.global_step,
                                "note": "Training timing/loss aggregates are not reconstructed; original step logs remain in trainer state."}
        else:
            result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            training_metrics = result.metrics
        if segment_steps is not None:
            global_step = trainer.state.global_step
            planned_steps = trainer.state.max_steps
            if global_step < planned_steps:
                if checkpoint_callback.segment_checkpoint is None:
                    raise RuntimeError("Segment ended without a newly sealed checkpoint")
                checkpoint = output / checkpoint_callback.segment_checkpoint
                if int(checkpoint.name.split("-")[1]) != global_step:
                    raise RuntimeError("Paused segment step differs from its sealed checkpoint")
                added_checkpoints = {path.name for path in output.glob("checkpoint-[0-9]*") if path.is_dir()} - before_checkpoints
                expected_checkpoints = set(checkpoint_callback.new_checkpoints)
                if added_checkpoints != expected_checkpoints:
                    raise RuntimeError("Segment checkpoint set differs from newly sealed checkpoints")
                if continue_through_checkpoints:
                    if global_step != checkpoint_callback.boundary_step:
                        raise RuntimeError("Continuing segment did not reach its exact optimizer-step boundary")
                elif added_checkpoints != {checkpoint.name}:
                    raise RuntimeError(f"Segment must add exactly one checkpoint; added {sorted(added_checkpoints)}")
                verify_checkpoint(checkpoint, run_manifest_hash)
                checkpoint_manifest_sha256 = sha256_file(checkpoint / "checkpoint_manifest.json")
                actual_output_bytes = None
                if space is not None:
                    counted_root = Path(output_budget_root).resolve(strict=True) if output_budget_root else output
                    actual_output_bytes = _tree_bytes(counted_root)
                    needed_after_pause = actual_output_bytes + space["final_model_reserve_bytes"] + space["safety_margin_bytes"]
                    if needed_after_pause > output_budget_bytes:
                        raise OSError(
                            "Sealed checkpoint exceeded the bounded output budget with final-model reserve "
                            f"({needed_after_pause} > {output_budget_bytes} bytes)"
                        )
                status = {
                    "status": "paused", "paused_at": now_utc(), "global_step": global_step,
                    "planned_global_steps": planned_steps, "checkpoint": checkpoint.name,
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                    "run_manifest_sha256": run_manifest_hash,
                    "resume_from": resume_from_checkpoint,
                    "segment_start_step": checkpoint_callback.start_step,
                    "segment_max_end_step": checkpoint_callback.boundary_step,
                    "segment_limit_optimizer_steps": segment_steps,
                    "continue_through_checkpoints": continue_through_checkpoints,
                    "new_checkpoints": checkpoint_callback.new_checkpoints,
                    "output_space_preflight": space,
                    "observed_output_bytes": actual_output_bytes,
                    "note": "Paused at a complete sealed checkpoint; the frozen training schedule is not complete.",
                }
                write_json(output / "run_status.json", status)
                return {"output_dir": str(output), "status": "paused", "global_step": global_step,
                        "planned_global_steps": planned_steps, "checkpoint": str(checkpoint),
                        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                        "training_mode": config.training_mode, "device": device,
                        "smoke_only": config.smoke_test}
        last_evaluation = trainer_holder.get("last_evaluation")
        development = (last_evaluation[1] if config.checkpoint_before_eval and last_evaluation is not None
                       and last_evaluation[0] == trainer.state.global_step else trainer.evaluate())
        metrics = {"training": training_metrics, "development": development, "benchmark": None}
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
        actual_output_bytes = None
        if space is not None:
            counted_root = Path(output_budget_root).resolve(strict=True) if output_budget_root else output
            actual_output_bytes = _tree_bytes(counted_root)
            if actual_output_bytes > output_budget_bytes:
                raise OSError(f"Completed output exceeds the configured bounded output budget ({actual_output_bytes} > {output_budget_bytes} bytes)")
        write_json(output / "run_status.json", {"status": "completed", "finished_at": now_utc(), "global_step": trainer.state.global_step,
                   "planned_global_steps": trainer.state.max_steps, "mode": config.training_mode,
                   "smoke_only": config.smoke_test, "artifact": str(final_dir), "benchmark_evaluation": "not_run",
                   "segment_limit_optimizer_steps": segment_steps,
                   "continue_through_checkpoints": continue_through_checkpoints,
                   "new_checkpoints": checkpoint_callback.new_checkpoints, "output_space_preflight": space,
                   "observed_output_bytes": actual_output_bytes})
        return {"output_dir": str(output), "artifact": str(final_dir), "status": "completed",
                "global_step": trainer.state.global_step, "planned_global_steps": trainer.state.max_steps,
                "training_mode": config.training_mode,
                "precision": precision, "device": device, "trainable_parameters": trainable_parameters, "metrics": metrics, "smoke_only": config.smoke_test}
    except BaseException as exc:
        write_json(output / "run_status.json", {"status": "failed", "finished_at": now_utc(), "error_type": type(exc).__name__, "error": str(exc)})
        raise
