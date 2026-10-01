# Execution boundary

Model outputs, retrieved pages, repository files, and tool results are untrusted.
Never run model-generated Bash or Python directly on the training host. Use the
container backend with a per-task workspace, no credentials, no Docker socket,
no host home mount, restricted resources, and no network by default. Container
isolation shares a kernel; hostile public inputs need a dedicated disposable VM
or stronger sandbox as well.

Keep GitHub, Colab, model-provider and storage credentials outside the repository
and outside every model-visible environment. Do not put secrets in trajectories,
KV entries, logs, prompts, or checkpoints. Check artifacts before publication.

Search is an explicit host-side tool with a configurable SearXNG endpoint. Its
results are data, never permission to run commands or transmit private files.
Production deployments must review destination and filesystem policies.

An inference assistant must not autonomously make purchases, change credentials,
contact people, or mutate external services merely because a task asks for it.
The training environment should provide synthetic accounts and disposable files.
