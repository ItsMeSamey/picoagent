# Picoagent training research and experiment plan

Research checked on 1 October 2026. This report proposes experiments for a small instruction-following agent with Bash, Python, documentation search, and external key-value memory. It records published findings separately from hypotheses. It does not report completed picoagent training or measured picoagent performance.

## Recommendation

Start with a reproducible 135M to 360M full-weight supervised training baseline, then improve the agent's interaction policy with verified rollouts. Make documentation retrieval, error recovery, and memory use necessary in the training tasks. Add outcome-based reinforcement learning only after the model can occasionally complete those tasks. Treat performance comparable to a roughly 1B model as a testable target on a declared task distribution, not a general capability claim.

The most valuable early question is whether the agent can read an unfamiliar specification, execute a small action, observe the result, and correct its next action. A model that reliably does this is a useful result even if its closed-book knowledge and long-form reasoning remain weak.

## What the sources establish

These are source-reported findings. The experiments have not been independently reproduced here, and results from larger models do not establish the same behavior at picoagent's scale.

1. **Executable actions are a useful design pattern.** CodeAct represents actions as executable Python and uses execution feedback for subsequent actions. This motivates a small, consistent action interface and actual execution during training rather than text that merely resembles a tool transcript. It does not establish that a million-parameter model can write general programs. [CodeAct paper](https://arxiv.org/abs/2402.01030)

2. **Training should expose the actual search interaction.** Search-R1 trains multi-turn retrieval with outcome rewards and masks retrieved tokens out of the policy objective. Its reported backbones are 3B and 7B. The transferable engineering lesson is that search results are observations supplied by the environment, not actions the model should be trained to fabricate. [Search-R1 paper](https://arxiv.org/abs/2503.09516)

3. **Verified tool rollouts can follow a supervised cold start.** ReTool combines synthetic code-augmented trajectories with subsequent outcome-driven RL using real execution. Its headline results concern 32B models and mathematical tasks. The useful precedent is the sequence of cold-start training followed by environment feedback, not an expected small-model accuracy. [ReTool paper](https://arxiv.org/abs/2504.11536)

4. **Reward design changes what tool behavior is learned.** ToolRL studies tool selection and argument rewards. A separate SLM study trains 1.5B and 3B models and distinguishes JSON validity from full call correctness. Its valid-format results are much stronger than some of its semantic-correctness results. Correct syntax alone is therefore an inadequate picoagent success metric. [ToolRL paper](https://arxiv.org/abs/2504.13958), [SLM tool-use study](https://arxiv.org/abs/2509.04518)

5. **Adaptive difficulty is promising, but not evidence for zero-data general intelligence.** Tool-R0 uses a generator/solver loop and verifiable synthetic call tasks. It estimates solver success with repeated samples and favors an intermediate difficulty band. ToolSample uses reward means and variances plus task-level curriculum because tool selection and argument prediction mature at different rates. These support measuring learnability before spending on large RL batches. [Tool-R0 paper](https://arxiv.org/abs/2602.21320), [ToolSample paper](https://arxiv.org/abs/2509.14718)

6. **There is no universally best agent RL recipe.** The STAR study varies rewards, scale, data, algorithm, and tool failures on TravelPlanner with 1.5B to 7B models. Its smaller models benefit from staged rewards and exploration support. It also finds that dense task-specific optimization can reduce out-of-domain performance. This is a reason to retain broad regression tests and compare simple GRPO against more complicated methods, not to copy its benchmark training set. [STAR paper](https://arxiv.org/abs/2603.21972)

7. **Bounded recurrent text state can be trained.** MEM1 repeatedly produces a compact state and discards old context. Its policy probabilities are computed with the same restricted information used during rollout, an important train/inference consistency requirement. Its reported models start from Qwen2.5-7B Base. This is evidence for the approach at 7B, not evidence that the same result follows at 135M. [MEM1 paper](https://arxiv.org/abs/2506.15841)

8. **Compaction is a learned action, not necessarily a fixed summarization timer.** ReSum trains summary-conditioned continuation with segmented trajectories; AgentFold learns selective folding through SFT. Their reported flagship models are substantially larger than picoagent. Their results motivate preserving recent observations and selectively compacting old work, while measuring lost information. [ReSum paper](https://arxiv.org/abs/2509.13313), [AgentFold paper](https://arxiv.org/abs/2510.24699)

9. **Memory operations can receive downstream rewards.** Memory-R1 trains memory management and answer selection with outcome-based RL, including structured add/update/delete/no-op behavior. Its evaluated scales are 3B to 14B. For picoagent, this motivates controlled memory-management tasks before attempting unconstrained long-lived memory. [Memory-R1 paper](https://arxiv.org/abs/2508.19828)

10. **Reusable code can carry capability that a small consumer lacks.** The SMITH authors' project page reports a frozen LFM2.5-350M consumer improving when supplied tools written by a separately trained 4B model. This is a system-level result: the larger writer and generated library contribute substantial work. It does not show that the 350M consumer learned to invent those tools independently. The page's paper link returned the project page during this review, so treat the detailed quantitative claims as author-reported until the full paper/artifacts are checked. [SMITH project](https://tool-use-smith.github.io/), [author repository](https://github.com/appier-research/smith)

11. **Unfamiliar-language performance may depend on metaprogramming.** A frontier-agent study finds that host-language generators, reusable helper code, and local verification materially help adaptation to unfamiliar languages; prose strategy advice alone transfers less well. Those experiments use frontier agents. Picoagent should therefore evaluate direct target-language writing separately from Python-assisted code generation and should not merge the two scores. [Unfamiliar-language study](https://arxiv.org/abs/2606.10933)

## Proposed agent architecture

Everything in this section is a design proposal to test.

- Use a single versioned action serialization for training, inference, replay, and evaluation. Start with one action per model turn. The controller validates the schema, executes the action, and appends a bounded observation.
- Expose `bash`, `python`, documentation lookup/search, and memory operations through that controller. Avoid rewarding a model merely for emitting a tool name. The environment must confirm the call actually ran.
- Make tool results explicit: exit status, bounded stdout/stderr, artifact handles, and truncation notices. Long results remain retrievable by handle instead of being silently lost.
- Begin with a frozen local documentation corpus and deterministic search. Treat a live-web evaluation as a separately labeled experiment because rankings, pages, and availability change.
- Keep external key-value memory separate from the transformer's attention KV cache. External memory stores task facts and references; attention cache is an inference optimization.
- Namespace memory by episode. Entries should carry a key, value, source or artifact reference, and version. Test overwritten values, stale facts, conflicting evidence, deletion, and no-op behavior.
- Retain the user goal and tool contracts, a compact structured task state, and the latest one or two exchanges. Keep larger evidence in immutable artifacts. Do not assume a summary is lossless.
- Use an isolated unprivileged execution environment with explicit CPU, memory, time, output, process, filesystem, and network limits. A Python subprocess or command allowlist alone is not a security boundary. Training workers must not expose credentials, checkpoint directories, hidden graders, or host mounts to generated code.
- Do not insert a large model into the inference loop without labeling that configuration separately. A teacher used to create training traces and a teacher used to solve evaluation tasks are different experimental conditions.

## Experimental ladder

The order below limits confounding variables. Each step needs a frozen development suite before selecting hyperparameters, plus a separate final test suite. Numeric budgets are starting configurations, not published guarantees.

### E0 Establish whether the smallest model can learn the protocol

**Hypothesis:** A short, stable interface makes small-model failures easier to diagnose and improves actionable output.

Train full-weight SFT on simple executable examples: inspect a file, transform supplied data, run a computation, read a manual, and return a verified result. Include tasks that require no tool and tasks that correctly end in an explicit inability to complete the request. Begin at 2K tokens per training example and small microbatches; increase context only when truncation metrics justify it.

Compare the untouched checkpoint, the SFT checkpoint, and a trivial scripted baseline. Measure valid-action rate, correct-tool rate, task success, early termination, invented observations, and recovery after an error. Constrained decoding is a separate ablation; it should not hide the raw model's formatting error rate.

**Advance when:** The model completes genuine held-out instances through actual tool calls, and masking, replay, termination, and checkpoint resume are verified. A lower training loss by itself does not satisfy this gate.

### E1 Require learning from documentation

**Hypothesis:** Counterfactual and randomized specifications teach documentation use more reliably than examples where pretrained knowledge already gives the answer.

Generate small fictional APIs with varying function names, argument order, units, indexing rules, optional defaults, return schemas, and error conditions. Give the model only a searchable manual and the task. Execute the generated call against the corresponding implementation. Use irrelevant documentation pages and near-matching functions as distractors.

Include paired tasks in which the same request has different correct answers under different manuals. Alter semantics, not just names. This detects an agent that fetches documentation ceremonially but ignores it.

Compare full docs, no docs, irrelevant docs, and a relevant excerpt supplied directly. The last condition separates retrieval failure from understanding failure. Missing or inconsistent documentation should sometimes make clarification or abstention the correct outcome.

**Advance when:** Correct manuals causally improve success over the controls, including held-out API families. Do not use a quota such as “always call search” as the success criterion.

### E2 Train recovery with verified synthetic trajectories

**Hypothesis:** Compact error-and-repair examples transfer better than exclusively polished one-shot solutions.

Use task generators with independent reference implementations and hidden tests. Produce several candidate trajectories per specification, execute them, and retain verified successful trajectories plus useful recoveries. Label every observation with the executor result that produced it. Never let a teacher invent stdout or mark its own unexecuted answer correct.

Vary paraphrases, paths, data shapes, tool schemas, and operation composition. Include quoting mistakes, missing files, empty search results, wrong API versions, and bounded transient failures. Avoid retaining endless loops as desirable SFT behavior.

Apply loss only to model-authored actions and final responses. User prompts, documentation, and tool observations are context. Verify the mask on decoded examples. Current TRL's assistant-only masking requires compatible generation spans in the template; a base tokenizer may have no chat template. [TRL SFT documentation](https://huggingface.co/docs/trl/en/sft_trainer)

Compare direct verified demonstrations against the same task mixture with repair traces, keeping token budget comparable. Track success by failure type rather than only an aggregate.

### E3 Add outcome RL after a measurable cold start

**Hypothesis:** On-policy interaction improves tool choice and recovery beyond matching teacher traces.

Start with a small GRPO experiment using fresh environments per rollout, explicit turn and token limits, and four or eight candidate rollouts per task. These group sizes are experimental choices. Log the fraction of groups with all failures, all successes, or mixed rewards. A group with identical rewards supplies no relative success-ranking signal.

Use final executable correctness as the primary reward. Compare sparse success against a staged curriculum that initially gives limited credit for valid actions and independently verified subgoals, then reduces shaping. Ensure a valid but incorrect call cannot outscore a successful solution. Introduce efficiency penalties only after correctness is established, and keep them too small to favor a cheap failure.

Sample more often from task families with useful reward variation, while retaining easy examples for regression and hard examples for measuring the frontier. Failure on every rollout should trigger easier tasks or additional verified demonstrations, not indefinitely larger RL runs.

Keep observations out of the policy-token loss. Recompute log probabilities from the exact context that was available when each action was sampled, including any compaction. Record policy version and sampling settings to detect stale-rollout errors. Current TRL supports tool/environment integration; its tool loop requires prefix-preserving templates and an explicit cap is advisable. [TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer)

**Accept the change when:** It improves the preregistered development metric across repeated runs without unacceptable regressions in no-tool instructions, unseen APIs, or safety-boundary tests. Report rollout cost as well as gradient-update cost.

### E4 Learn bounded memory and compaction

**Hypothesis:** A compact structured state plus retrievable artifacts can preserve task performance beyond the training horizon.

Create episodes where information learned early becomes necessary after intervening tool calls. Add late corrections, irrelevant facts, similar keys, exact identifiers, and conflicting observations. Begin with deterministic compaction; then compare a learned compact-state action. Suggested initial state budgets are 128, 256, and 512 tokens.

Compare full available history, recent-window truncation, deterministic extracted state, learned summary, and external memory with retrieval. Keep total context budgets visible and report separate equal-budget comparisons. Include an oracle-state condition as a diagnostic upper bound, clearly labeled as privileged.

Train continuations after realistic memory truncation and on states produced by the student, not exclusively perfect teacher summaries. Preserve exact numbers and file handles outside lossy prose. If a pointer is kept, verify that its target remains retrievable.

Test at twice and four times the training interaction horizon. Score final task success, critical-fact retention, stale-memory errors, unnecessary writes, token cost, and failure accumulation over repeated compactions. No experiment can establish indefinite reliability from a finite horizon.

### E5 Evaluate reusable helper code separately

**Hypothesis:** Small agents can gain reliable capability by selecting and composing a compact verified helper library.

First test a fixed library with concise schemas and independently tested implementations. Then, as a separate configuration, allow the agent to write reusable helpers in its workspace. Charge library construction and validation to the system's total compute budget.

Hold out helper families and task compositions. Never package benchmark solutions as “generic tools.” Distinguish success from choosing a prebuilt solver, writing a new solver, and composing general primitives. This prevents external code from being misrepresented as increased capability in model weights.

### E6 Attempt a sealed documentation-only unfamiliar language challenge

**Hypothesis:** The learned retrieve-execute-repair policy transfers to a language whose details were unavailable during posttraining.

Use four increasingly strong tests:

1. Rename syntax while preserving familiar semantics. This checks following documentation but is weak evidence of conceptual transfer.
2. Change important semantics: indexing, scope, mutation, evaluation order, or integer behavior. Keep the language small and the manual complete.
3. Hold out an entire language family or execution model during training, not merely instance seeds or keywords.
4. After final checkpoint selection, independently generate a fresh language specification and interpreter with new semantics and tasks. Seal hidden tests and reference solutions outside the agent's workspace.

Supply the manual, an interpreter, and a bounded scratch workspace. Documentation may contain elementary language examples, but no solved evaluation problems. Grade submitted target-language artifacts using hidden tests. Keep interpreter behavior and errors deterministic.

Run separate tracks for direct target-language code and Python-assisted target-code generation. In the latter, require the submitted artifact to execute in the target interpreter; otherwise the agent could simply solve the task in Python and bypass the language challenge. A language is only “unseen in our posttraining” when using a public real-world language with unknown pretraining exposure.

Report first-attempt success, success after a fixed interaction budget, compile/parse success, semantic-test success, documentation retrieval use, and performance with absent or wrong documentation. Freeze any cross-task memory policy in advance. A strict first-use track resets memory and helper files between tasks; a separate adaptation track may retain them, with ordering and compute disclosed.

## Contamination controls and reproducibility

**Project rule:** Do not train on public benchmark problems, answers, rationales, traces, or benchmark training splits. The cited papers are methodological references, not permission to import their task data.

For each example, retain generator version, seed, task family, source/specification hashes, teacher identity and prompt version if used, executable trace, verifier version, result, split, and any filtering reason. Track license/provenance for source documentation. Prevent benchmark assets from being mounted in training or synthetic-data-generation environments.

Split by task family, API implementation, grammar family, and composition pattern where possible. A different random seed within the same template is an in-distribution test, not an out-of-distribution claim. Keep development tasks separate from final test tasks; repeated tuning on a final benchmark turns it into development data.

Check exact duplicates, normalized overlap, n-gram overlap, and structural similarity. These detect some leakage, not all semantic contamination. Existing pretrained checkpoints and teachers have incompletely known exposure, so the defensible claim is “no benchmark data intentionally used in this project's posttraining, with documented checks,” not universal contamination freedom. [Evaluation-harness decontamination documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md)

Pin model/tokenizer revisions, training code, dependency versions, tool contracts, prompts, evaluator version, data manifests, random seeds, and documentation snapshots. Save optimizer and scheduler state, RNG state, sampler progress, and global step for resume tests. Publish failed runs and selection rules alongside successful ones.

The primary metric is final executable task success with a declared token/turn/time budget. Also report invalid calls, hallucinated tools, constraint violations, premature completion, error recovery, loops, memory correctness, latency, generated/observed tokens, peak memory, and training plus rollout GPU-hours. Use per-family scores, repeated seeds, and confidence intervals; bootstrap by task family when instances are correlated.

A claim of “matching a 1B model” requires a named checkpoint, actual parameter count, identical task set, tools, prompts where appropriate, context and sampling budgets, and latency/cost reporting. Compare base+tools, SFT+tools, SFT+RL+tools, and no-tool/no-doc/no-memory ablations. Some models branded “1B” have materially different actual parameter counts.

## Model and precision choices

- **Simplest initial engineering baseline:** [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), Apache 2.0, 8K configured context, standard Llama architecture. Verified revision: `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. Its base tokenizer has no chat template. Use an explicitly versioned serialization. [Revision](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/commit/f8027fd0eaeea54caa13c31d31b9fdc459c38b49), [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json), [tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/tokenizer_config.json)
- **Size-floor experiment:** [SmolLM2-135M base](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Its low cost makes it useful for discovering protocol limits; stronger general instruction-following should not be assumed.
- **Contemporary same-size challenger:** [IBM Granite 4.0 350M Base](https://huggingface.co/ibm-granite/granite-4.0-350m-base), Apache 2.0, 32K context, explicit code and fill-in-the-middle training. Its corpus includes proprietary data, so pretraining contamination cannot be fully audited from the model card.
- **Larger comparison:** [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), Apache 2.0, 32,768 context. Verified revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Use its base checkpoint when isolating this project's posttraining. [Revision](https://huggingface.co/Qwen/Qwen3-0.6B-Base/commit/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)
- **OptGear identity resolved:** [Opt.Gear-1M](https://huggingface.co/OptGear/Opt.Gear-1M) is a specialized board-command model with no general pretraining and a short context. [Opt.Gear-270M](https://huggingface.co/OptGear/Opt.Gear-270M) is already instruction-tuned, uses custom model code, and lists CC-BY-NC-SA-4.0 for its weights. Its report says general-model pretraining excluded code. These are material differences for a Bash/Python agent. [Opt.Gear report](https://arxiv.org/html/2608.01034v1)
- **Optional structured-call baseline:** [LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base). It uses a custom license with a commercial revenue threshold; the vendor discourages code/math use for the instruction model. Evaluate rather than assume suitability. [License](https://huggingface.co/LiquidAI/LFM2.5-350M-Base/blob/main/LICENSE), [vendor announcement](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)

Full fine-tuning means updating all model weights. It does not require FP32 arithmetic everywhere. BF16 mixed precision with suitable optimizer state is the preferred starting point on compatible hardware; use a checked FP16 configuration when necessary. QLoRA keeps the quantized backbone frozen and updates low-rank adapters, so it is an explicitly different experiment. Quantize the final trained model only after preserving a higher-precision evaluation reference. [QLoRA paper](https://arxiv.org/abs/2305.14314)

A conservative mixed-precision AdamW accounting is roughly 18 bytes per parameter before activations, temporary tensors, and framework overhead: approximately 2.4GB for 135M, 6.5GB for 360M, and 10.8GB for 600M. Implementations can differ. These are estimates of training tensors, not total VRAM requirements; RL adds generation and possibly reference-model overhead. Measure peak allocation on the actual GPU before committing to a run. [Transformers memory documentation](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy)

## Colab execution note

Google provides an [official Colab CLI](https://github.com/googlecolab/google-colab-cli), installed with `uv tool install google-colab-cli` or `pip install google-colab-cli`. Pin the package version in execution records. CLI authentication uses ADC or OAuth; `colab auth` separately configures GCP authentication inside a running VM. The official guide documents the required ADC scopes and read-only `colab sessions` check. GPU provisioning can consume compute units; allocation, elapsed runtime, checkpoint retrieval, and shutdown must be part of the run record. Authentication and available hardware should be verified rather than inferred from successful installation. [Official operator guide](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)

## Immediate next decisions

1. Choose the first model by an inexpensive baseline and tooling compatibility check; retain 135M as a meaningful lower-capacity experiment.
2. Validate execution, observation masking, contamination manifests, and checkpoint resume before a long run.
3. Establish E0 and E1 performance before adding RL, compaction, or helper generation.
4. Reserve the independently generated language challenge for the frozen final checkpoint. Keep all development feedback out of that test.

The working hypothesis is that good external tools and learned interaction can substantially improve a small model's usefulness. How far that transfers across domains, horizons, and unfamiliar languages remains an empirical question.
