# External dataset review — 2026-10-01

No external corpus is admitted by this review. A matching subject or a
"decontaminated" label is not sufficient evidence. Do not download and merge a
whole repository by default. Compact artificial reasoning remains a separately
generated augmentation; none of these is an exact match for the requested format.

## NVIDIA Terminal-Corpus — investigate skill-only subsets

The [card](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus)
describes approximately 140,000 skill-generated trajectories and a separate
226,000-example adaptation stream, under CC-BY-4.0. The
[paper](https://arxiv.org/html/2602.21193v1) identifies benchmark-related sources
in the adapter stream and describes independently composed primitive-skill tasks.
Exclude dataset adapters and seed-derived material. Investigate only documented
skill-based partitions, preserving attribution and an immutable revision. Their
stateful terminal protocol differs from picoagent's per-call shells, so direct
relabeling would be unsafe. Published overlap filtering is not proof against
every benchmark or semantic duplicate. No rows have been added to training.

## NL2Shell — quarantine, not verified execution

[AryaYT/nl2shell-terminal-bench](https://huggingface.co/datasets/AryaYT/nl2shell-terminal-bench)
lists 7,509 Apache-2.0 synthetic examples. Its card explicitly says solutions
were not individually executed end-to-end. It reports canary and 8-gram checks
against Terminal-Bench, not all benchmarks. These are candidate plans with
expected results, not trusted observations. They would require provenance checks,
real execution and protocol adaptation before use.

## microagent-train-v3 — do not automatically admit

[prometheus04/microagent-train-v3](https://huggingface.co/datasets/prometheus04/microagent-train-v3)
lists 30,578 Apache-2.0 trajectories, mixing inherited terminal data and synthetic
recovery examples. Recovery examples use simulated observations and were motivated
by benchmark-evaluation failures. That combination does not meet this project's
strict admission rule without deeper source-level review. Do not use the
publisher's contamination assertion as a guarantee.

## Terminal-Lego — task source, not a drop-in trace corpus

The [Prime Intellect mirror](https://huggingface.co/datasets/PrimeIntellect/Terminal-Lego-15k)
describes StackOverflow-grounded, Docker-checked tasks and exclusions for invalid
or trivial cases. It is labeled Apache-2.0, but upstream source obligations and
each retained task's origins still need checking. Task directories are not
themselves completed agent trajectories. The primary
[SWE-Lego dataset](https://huggingface.co/datasets/SWE-Lego/Terminal-Lego-15k)
is the preferable provenance starting point; mirror availability must be checked.

## Admission requirements

Pin revisions, files and row IDs; preserve licenses and source URLs; exclude all
benchmark-derived branches; review actual execution evidence and tool semantics;
filter out later-phase programming languages; deduplicate by task lineage; and
keep evaluation artifacts entirely outside training. Treat external code as
untrusted and never execute it in the native trusted-teacher fixture runner.
