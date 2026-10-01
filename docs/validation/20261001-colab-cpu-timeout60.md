# Colab CPU transport with REQUEST_TIMEOUT=60

## Result

On 2026-10-01, one explicitly named standard CPU session, `picoagent-transport-cpu-v1`, completed **five independent CLI execute invocations and a verified 1 MiB upload/download roundtrip**. No GPU, TPU or high-memory runtime was requested. Status reported CPU / Standard / DEFAULT, and each executed cell found no accelerator device nodes.

The named CPU session was explicitly stopped after receipts were saved. The CLI confirmed termination, and a subsequent inventory confirmed that name absent. Total experiment duration was 195.282 seconds, inside the 840-second overall bound.

The machine-readable receipt is `20261001-colab-cpu-timeout60.json`. The executable recipe is `colab_cpu_transport_probe.py`. It uses the existing authenticated official CLI environment; it never reads authentication-file contents or persists raw CLI output. The existing launcher/wrapper was not modified.

| Operation | Verified result | Wall time, seconds |
|---|---|---:|
| Independent execute 1 | arithmetic receipt, CPU device check | 28.125 |
| Independent execute 2 | arithmetic receipt, CPU device check | 18.631 |
| Independent execute 3 | arithmetic receipt, CPU device check | 28.735 |
| Upload 1 MiB | CLI completed | 9.714 |
| Download 1 MiB | exact length and SHA256 match | 10.935 |
| Independent execute 4 | arithmetic receipt, CPU device check | 19.581 |
| Independent execute 5 | arithmetic receipt, CPU device check | 20.646 |

The transferred bytes were the sequence 0 through 255 repeated 4,096 times, with length 1,048,576 and SHA256 `fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83`. This was synthetic diagnostic data, not a dataset, checkpoint, credential or model. No training or learned-policy execution occurred.

## Interpretation and limits

This run demonstrates repeated small CPU transport under the 60-second setting, including fresh WebSocket connections after REST file transfers. It supports trying this setting for small bootstrap commands and bounded checkpoint transport.

It **does not establish that changing the timeout caused the improvement**: there was no interleaved 10-second control on the same CPU runtime. It also does not establish GPU/TPU reliability, sustained large-checkpoint throughput, or a permanent upstream race fix. Wall times include authentication/session lookup, HTTP, WebSocket startup, execution and cleanup; they are not handshake measurements.

The installed source supports the hypothesized failure mechanism:

- `jupyter_kernel_client/constants.py` reads `REQUEST_TIMEOUT` from the controller environment, defaulting to 10 seconds
- `KernelWebSocketClient.__init__` captures that value as its default timeout
- `wsclient.py:start_channels()` waits for `connection_ready` and proceeds without checking the wait result
- `stop_channels()` can close and clear the socket while the connection thread is still active
- `websocket/_app.py` later accesses `self.sock.sock`, consistent with the previously observed teardown symptom, but not proof of its cause

The environment change also affects other kernel request waits, so even a controlled improvement would not isolate WebSocket handshake timing alone. The underlying unchecked-readiness behavior remains unchanged. The JSON report records installed package versions and SHA256 values of the inspected vendor source files.

## Reproduction and boundaries

Set `REQUEST_TIMEOUT=60` in the **local CLI process environment before Python imports the client**. Setting it inside the remote execution cell is too late. Keep the existing authentication, TLS verification and proxy configuration unchanged.

For an already authorized, explicitly selected session:

```sh
REQUEST_TIMEOUT=60 HOME=/path/to/existing-colab-home \
  /path/to/colab-safe exec --session EXPLICIT_SESSION --timeout 30 --file probe.py
```

The committed recipe documents the actual CPU-only experiment, including one allocation, five small cells, the synthetic transfer, operation deadlines, process-group cleanup, and a finally-block stop of only the named session. It refuses to overwrite its existing evidence file; a new experiment needs an explicitly reviewed output path and authorization for any allocation/upload. Never reuse the recipe as an unattended allocator or retry a quota refusal by changing accounts/hardware.

Raw CLI output is held only in memory. Reports contain whitelisted operation labels, elapsed times, error class names/stack locations if present, and validated diagnostic values; no runtime URLs, tokens, headers or authentication contents are included. The vendor CLI's private state/log directory remains outside the repository. Do not publish its raw logs.
