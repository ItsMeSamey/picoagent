# picoagent

A small documentation-oriented tool agent, with one Python harness for training
rollouts and inference. The goal is to learn instruction following, choosing
tools, reading references, using Python/Bash, and checking results. This is an
experimental training project, **not a claim of general competence**.

## Current status

- Shared tool harness, structured protocol, KV memory and context compaction
- Original procedural curricula with held-out families and preserved provenance
- Full-weight SFT and separately labeled optional QLoRA
- CPU random-model smoke passed; pretrained model quality is not established
- CUDA/TPU training and sandboxed learned-policy evaluation remain unvalidated
- No agentic benchmark score claimed and no benchmark training data included

Accelerator policy: use GPU/TPU hours sparingly. Any separate accelerator smoke
job is optional and must have a server-side timeout of at most **60 seconds**.
Prepare dependencies and verified trajectories before allocating accelerators.
Assistant-authored candidates are preserved separately and are not
training-eligible until their tool interactions have actually been executed and
checked; no separate paid model API is required for candidate generation.

The initial candidate is `HuggingFaceTB/SmolLM2-360M` at an exact revision.
See [research](docs/research.md) for Granite 350M, Opt.Gear, smaller baselines,
precision trade-offs, and the evidence-led experimental sequence. A pretrained
model already contains knowledge: this project does not claim to erase it or to
prove its unknown pretraining corpus clean.

## Install and test

```bash
uv venv
uv pip install -e '.[train,dev]'
python -m pytest -q
python -m picoagent.training smoke --output-dir /tmp/picoagent-smoke
```

The exact CPU smoke package versions are in `requirements-cpu.lock.txt`; its
PyTorch CPU wheel requires the official PyTorch CPU wheel index. TPU runtimes
must retain their matched `torch`/`torch_xla` pair. Do not install a generic CUDA
PyTorch wheel over Colab's TPU runtime.

## Build the isolated task environment

```bash
docker build -t picoagent-sandbox:v1 -f container/Dockerfile .
docker image inspect picoagent-sandbox:v1 --format '{{.Id}}'
```

Record the resulting immutable image identity in run provenance. Do not mount
credentials, host home directories or the Docker socket into task containers.
Model-generated commands fail closed when Docker/Podman is absent. See
[security](SECURITY.md) and the [harness interface](src/picoagent/harness/INTERFACES.md).

## Data stages

`data/curriculum-v2` contains original task specifications and explicitly
unexecuted authored examples. Those examples are not automatically admitted to
training. Every attempt must retain its environment outputs and oracle checks.
The task fixtures/reference answers are available to the teacher and verifier;
the student receives only its prompt, tools, and model-visible environment.

```bash
python -m picoagent.data validate --manifest data/curriculum-v2/manifest.json
python -m picoagent.data collect \
  --tasks data/curriculum-v2/train.tasks.jsonl \
  --archive-dir data/attempts/train-v1 \
  --runtime docker --image picoagent-sandbox:v1
python -m picoagent.data export \
  --archive-dir data/attempts/train-v1 --output-dir data/rollouts/train-v1
```

Collect development data independently. Keep the final test families sealed
until checkpoint selection is frozen. Prepare an immutable training snapshot
from verified exported train/dev JSONL, then point the configuration at it:

```bash
python -m picoagent.training prepare \
  --train data/rollouts/train-v1/train.jsonl \
  --dev data/rollouts/dev-v1/dev.jsonl --output-dir data/snapshots/v1
python -m picoagent.training train --config configs/smol360m_full.json
```

Scripted-teacher completion rates are data-generator checks, **not model scores**.
Search bootstrap tasks use an original, deterministic local document corpus,
clearly distinguished from live SearXNG retrieval.

## Reviewed native training collection

`data/native-training-collection-v1/manifest.json` combines **20,326 training
problems and 221 dev problems** with preserved actual tool receipts. These are
seeded procedural variants, not that many distinct task families or independently
sampled model trajectories. Teacher author-model identity is null where it was
not exposed; the scalable execution is reviewed procedural replay.

The collection includes CLI/Python work, a small recovery set, local document
search and KV operations, and 216 canonical compaction problems. All three
compaction modes have training/dev examples; manual data includes genuine
nonempty keep selections as well as empty selections. The raw alternative mode
runs remain archived without inflating canonical problem counts.

The optional final view `data/native-training-plans-v1` adds reviewed compact
artificial plans to 5,004 CLI training problems, replacing their normal views.
Original traces and all dev examples stay preserved. See
[artificial-plan admission](docs/artificial_action_plans.md),
[native production configurations](configs/README.md), and
[checkpoint transfer](docs/checkpoint_workflow.md). These are data and plumbing
results, not evidence that a trained model succeeds on the tasks.

## Inference

```bash
python -m picoagent 'Use Python to compute and verify 173 * 289' \
  --model runs/smol360m-full-v1/final-model \
  --image picoagent-sandbox:v1 --runtime docker
```

Add `--search-url http://127.0.0.1:8080/search` for a configured SearXNG endpoint.
The sample SearXNG settings enable JSON; replace its local secret before serving.
Each inference episode preserves its trace and uses a fresh KV file. Long
conversations support `--compaction-mode full|half|manual`: summarize all history,
replace the oldest token-half, or let the model select atomic groups to keep.
System instructions remain pinned and original traces are preserved. See
[mode contracts and qualification](docs/compaction_modes.md). For the 4K
compaction curriculum, use `--max-new-tokens 768` to reserve structured-summary
output space.

## Evaluation contract

See [reproducibility](docs/reproducibility.md). All benchmark material stays out
of training and checkpoint selection. Score the trained policy's actual
environment outcomes, not teacher answers. Publish failures, invalid tool calls,
runtime/tool budgets, and multiple seeds; compare against a 1B baseline under
the same harness and budgets. There is no guarantee this model will match it.

Run paired internal dev evaluation of an actual checkpoint with:

```bash
python -m picoagent.evaluation --model runs/smol360m-full-v1/final-model \
  --tasks data/compaction-v1/dev.tasks.jsonl --output runs/eval-dev-v1 \
  --image picoagent-sandbox:v1 --runtime docker
```

This executes all three compaction modes in fresh workspaces. Reports expose
per-mode and family-balanced scores; checkpoint qualification prioritizes the
weakest mode, not an average that hides it. No such learned-model run has been
completed yet.

Documentation-only unfamiliar-language experiments come last. Invented languages
and randomized APIs can establish novelty; C/JS/HTML cannot honestly be called
unseen by a web/code-pretrained base.

## Preserve work

Generated task data and raw synthetic traces are committed or archived with
verified durable URLs and SHA-256 manifests. Credentials and model weights are
not committed as ordinary Git objects. Checkpoints need separate verified
off-runtime storage before pruning local copies; a file on the same Colab VM is
not a backup. Restore tests are part of the training workflow.
