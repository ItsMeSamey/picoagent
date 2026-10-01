# Training configurations

- `smol360m_full.json`: full supervised fine-tuning, automatically prefers an available TPU, then CUDA, then CPU. CUDA selects BF16 on supported hardware or FP16 otherwise
- `smol360m_tpu.json`: requires a TPU/XLA device; BF16 operation-level autocast with FP32 model parameters/optimizer states. No silent CPU fallback
- `smol360m_qlora.json`: separate CUDA-only 4-bit frozen-base/adapters experiment, **not** full fine-tuning

The base is `HuggingFaceTB/SmolLM2-360M`, pinned to immutable commit
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. The [model repository](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/tree/f8027fd0eaeea54caa13c31d31b9fdc459c38b49)
provides architecture/tokenizer/model assets. A model update is a new experiment.
The model's pretraining contamination is unknown. SFT does not remove prior
knowledge or establish that evaluation languages/domains were unseen in pretraining.

## Verified data first

Generate original tasks, collect real container rollouts, and export admitted
train/dev traces using the data CLI. Authored examples and failed/unexecuted
attempts are not production SFT inputs. Then freeze the resulting files:

```sh
python -m picoagent.training prepare \
  --train data/export/train.jsonl --dev data/export/dev.jsonl \
  --output-dir data/snapshots/v1
python -m picoagent.training validate --manifest data/snapshots/v1/manifest.json
python -m picoagent.training train --config configs/smol360m_full.json
```

`prepare` checks source, successful execution/receipt structure, oracle result,
canonical message/tool linkage, disjoint families/templates/task IDs, and duplicate
conversations (normalizing tool call IDs). It snapshots exact source bytes into a
new directory, marks them read-only, and records hashes. Read-only permissions are
not tamper-proof; hashes are rechecked at every training start. Receipts cannot
independently prove the truth of arbitrary externally supplied provenance claims.

The trainer accepts **only train and development** splits. It never opens the
synthetic test/lockbox split or runs official benchmarks. Development loss is not
an agent task-success score. Freeze hyperparameters before one-shot lockbox work;
do not adaptively tune against official benchmark results.

## Loss and protocol

Training and inference share `picoagent.harness.protocol`; no implicit native
chat-template substitution. Only assistant JSON and the end-message delimiter
receive labels, including tool calls. System/user/tool observations and assistant
role headers are masked. Tokens crossing role boundaries are masked too. Padding
is masked; overlong records fail rather than being silently truncated or packed.

When traces contain `model_events`, each example uses the exact recorded model
input and supervises just that event's output. Accepted context-summary events
use their original summary request/response with no tools. A compacted trace without
these events is rejected. Short traces without compaction can use the complete
conversation fallback. Enabling this mechanism does not mean any summarization
training has actually been performed.

## Precision and hardware

The full-SFT path updates **all** parameters. CUDA autocast uses BF16 or FP16;
XLA uses `torch.autocast('xla', dtype=torch.bfloat16)` around forward/loss,
while retaining FP32 master parameters and optimizer state. FP32 is available for
CPU/debugging. Lower-precision activations reduce one part of memory usage;
this is not a promise that the entire full-fine-tuning footprint becomes 16-bit.
Gradient checkpointing, batch size 1, and accumulation are enabled in presets.

Use a matched `torch`/`torch_xla` installation on TPU; never independently upgrade
torch in a preconfigured Colab TPU runtime. The currently validated local CPU stack is
torch 2.14.1+cpu, Transformers 5.18.0, Accelerate 1.15.0, Tokenizers 0.23.2.
Use the patched Transformers 5.10+ dependency range; legacy 4.x validation was
superseded before production. See `docs/training_validation.md` for current evidence.
The TPU target stack supplied by Colab was torch 2.9.0 / torch_xla 2.9.0; runtime
validation is required and is recorded separately from CPU results. XLA pads every
sequence to `max_seq_length` to limit recompilation. This baseline supports one
process/device; no implicit multi-TPU/DDP or QLoRA-on-TPU support is claimed.

## Checkpoints and recovery

Presets save every 10 optimizer steps. An independent 300-second wall-clock
interval also requests evaluation plus a full-state save at the next completed
optimizer step; set `checkpoint_interval_seconds` to `null` to disable that
extra deadline. A long TPU compilation or optimizer step cannot be interrupted
safely for a mid-step save, so the interval is a deadline checked at step ends,
not a guarantee of a save during an unfinished step. Both settings are frozen
in the run config/provenance.

Each run saves its original config, package versions, hardware, model revision,
source hashes and source snapshot, tokenizer snapshot, exact dataset snapshot,
seed, precision, parameter counts and label policy. Each completed checkpoint
contains weights, optimizer, scheduler, Trainer state and RNG state, with a
SHA256 manifest published atomically only after the save finishes. Best means the
lowest development loss; the final exported model remains the last trained step.

Automatic Hugging Face pruning is disabled. The separate checkpoint workflow
copies sealed checkpoints to an explicitly chosen off-runtime destination, verifies
all bytes, and only then prunes runtime duplicates while retaining latest two plus
best. Durable copies and datasets are not deleted by training. A local path alone
does not prove that storage survives runtime loss. An unsynced checkpoint may be
lost if Colab terminates, so configure and verify transfer before a long run.

```sh
python -m picoagent.training train --config configs/smol360m_tpu.json \
  --resume runs/smol360m-tpu-v1/checkpoint-25
```

Resume requires the original immutable run identity, source, package environment,
backend/hardware and dataset hashes. Output/dataset paths may change after restore;
original paths remain in provenance. Refuses older checkpoints when a newer
checkpoint directory exists, completed runs/final artifacts, corrupt state, and
step-count collisions. Preserve/quarantine incomplete later saves before retrying;
never overwrite completed checkpoints. Cross-hardware bitwise reproducibility is
not promised.

## Offline plumbing smoke

```sh
python -m picoagent.training smoke --device cpu --output-dir /tmp/picoagent-cpu-smoke
# Only on an already authorized/configured TPU runtime:
python -m picoagent.training smoke --device xla --output-dir /content/picoagent-xla-smoke
```

Creates a fresh random 90,592-parameter GPT-2 and byte-level tokenizer, uses clearly
labeled unexecuted `pipeline_smoke` fixtures, performs two optimizer steps and
saves full checkpoints. These fixtures cannot enter production training. This
checks machinery, not agent capability, pretrained-model quality, or benchmark
performance. CPU interruption-and-resume was additionally tested after step one.

## Native observed teacher data and mode qualification

Reviewed deterministic teacher programs may use a separately sealed
`picoagent.native_teacher.dataset.v1` snapshot. Set
`allow_native_teacher_observed: true` explicitly in that run's config. The generic
JSONL importer still rejects these records; the native verifier rechecks frozen
sources, exact receipt bytes, model-visible contexts, fixture oracles and source
reviews. Native evidence is never relabelled as container execution, and this
exception does not permit model-generated learner commands on the host.

Presets now use a **4096-token** sequence cap to accommodate complete compaction
request/response examples (the current reference manual decisions reach 3510
tokens). This is an input-admission limit, not a hardware-throughput claim. CUDA
batches dynamically pad; XLA's fixed padding makes this setting costlier. Validate
the whole selected dataset's actual token lengths on CPU before allocating a GPU
or TPU. Do not silently truncate overlong examples to fit a cheaper setting.

Development loss determines the trainer's retained `best` checkpoint for recovery.
Final agent qualification is separate: run the paired full/half/manual dev suite
and compare the weakest mode's family-balanced task success first. Do not describe
a loss-selected checkpoint as an agentic winner before actual tool evaluation.
