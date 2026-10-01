# Training validation

## CPU pipeline and interruption recovery: passed

Verified 2026-10-01 11:17 UTC on Python 3.12.14, torch 2.14.1+cpu,
Transformers 5.18.0, Accelerate 1.15.0, Tokenizers 0.23.2, pytest 9.1.1 and setuptools 84.0.0.

This report supersedes the earlier 4.x validation. The project moved to patched
Transformers 5.10+ before production after dependency advisories were identified.
The 5.x API migration uses `dtype`, ratio-valued `warmup_steps`, and mandatory
safetensors defaults; no legacy 4.x runtime is recommended.

This is a **random-model plumbing test**, not a trained-agent evaluation. The local
GPT-2 has 90,592 trainable parameters, one 32-wide layer, two attention heads, and a
384-token byte-level BPE tokenizer. The fixtures are explicitly unexecuted
`pipeline_smoke` data. Production admission rejects them.

Procedure:

1. Initialize the random model/tokenizer with seed 7
2. Run full-parameter FP32 SFT for two optimizer steps, batch size 1 and gradient
   accumulation 2, with assistant-only labels and gradient checkpointing
3. Independently recreate the same initialization and data, interrupt immediately
   after step one's full checkpoint and SHA256 manifest are saved
4. Validate weights, optimizer, scheduler, Trainer/RNG state, and checkpoint hashes
5. Resume the interrupted run to step two using the immutable original config
6. Compare final model tensors and file hashes with the uninterrupted run

Results:

- All 16 stored tensors changed from initialization
- All 16 resumed final tensors are exactly equal to uninterrupted final tensors
- Both final `model.safetensors` files have SHA256
  `5ed4aafa26dcf883e1594c4e74915350008bb4874ce780c9ad1f57bbfc9ca53a`
- Uninterrupted and resumed final development loss: `5.8917012214660645`
- Uninterrupted mean training loss reported by Trainer: `5.961413383483887`
- Resumed invocation reports `2.9807066917419434` as `train_loss`; Hugging Face
  divides this invocation's accumulated loss by the full global step count, so
  **this value must not be compared as a resumed model quality improvement**
- No benchmark was run; benchmark metrics are `null`

Machine-readable values and manifest hashes are in
[`training_validation_cpu.json`](training_validation_cpu.json). The two final
model hashes match; run manifests correctly differ because original output/model
paths and run timestamps differ. Temporary full artifacts were retained at
`/tmp/picoagent-validation-v5-reference` and `/tmp/picoagent-validation-v5-interrupted`
on the execution machine; these paths are not a permanent artifact store.

Automated reproducer:

```sh
PICOAGENT_RUN_ML_TESTS=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests/test_training_smoke.py
```

The same integration test passed. Ordinary dependency-free tests skip the heavy
ML check. After the 5.x pinned-cache identity regression was added, 54 focused training unit tests passed, covering
admission, split/hash integrity, exact assistant/event masking, precision choice,
resume collisions, atomic manifest writes, and safe checkpoint retention. Ruff
passed for training code and tests. The full 55-test training suite, including the
real interruption/resume test, passed again on the patched stack at 11:23 UTC.
A further real run set the step interval above its two-step budget and forced
the wall-clock deadline; step one still produced a sealed, verified full-state
checkpoint. This checks the independent timer-triggered save callback.
The production pin check now records cached config snapshot revision and SHA256
instead of the private `_commit_hash` attribute removed by Transformers 5.18.

The controlled interruption tests recovery after a completed save. It does not
simulate arbitrary process death during every possible individual filesystem write;
partial/unsealed saves are instead rejected by integrity checks and must not be
used as recovery evidence. CPU bitwise equality does not promise equality across
TPU/CUDA hardware or library versions.

## TPU / CUDA / QLoRA status

The XLA backend is implemented with explicit device verification, BF16 autocast,
static sequence padding and Hugging Face's XLA-aware full-state checkpoint path.
TPU runtime validation is pending; merely allocating or detecting a TPU is not a
successful training result. The intended Colab target has torch 2.9.0 and
`torch_xla` 2.9.0; keep those packages matched.

CUDA mixed precision and QLoRA are implemented but have not been executed in this
CPU validation. QLoRA is adapter tuning, not full fine-tuning. Neither production
SmolLM2 training nor official benchmark capability is established by these tests.
