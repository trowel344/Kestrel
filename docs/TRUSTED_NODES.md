# Trusted RPC nodes

Kestrel can extend a coordinator with accelerator devices exposed by trusted
llama.cpp RPC workers. This is experimental infrastructure for fitting a model
that is too large for one machine. It is not a general cluster scheduler and it
does not make raw llama.cpp RPC safe.

The worker is trusted with model weights and inference. Kestrel protects the
transport; it cannot attest that a remote worker returned correct results.

## Security boundary

- Bind the unauthenticated RPC worker to loopback only.
- Carry RPC through Kestrel's authenticated, host-key-pinned SSH tunnel.
- Use the same pinned llama.cpp revision on coordinator and worker.
- Never expose the worker on `0.0.0.0` or an untrusted network.
- Treat `--allow-insecure-rpc` as a developer-only escape hatch.

Managed tunnel supervision currently requires a Linux coordinator because it
uses `/proc` socket-ownership checks to prove that the forwarded listener
belongs to the SSH child Kestrel launched.

Read [SECURITY.md](../SECURITY.md) before exposing any worker.

## Worker setup

Build the same llama.cpp revision on both machines with `GGML_RPC=ON`, then
start the worker on loopback:

```bash
./build/bin/ggml-rpc-server --host 127.0.0.1 --port 50052
```

From a trusted console, copy the key type and base64 data—not the comment—from
`/etc/ssh/ssh_host_ed25519_key.pub`. Verify it out of band before registering
the node.

```bash
kestrel nodes add worker \
  --endpoint 127.0.0.1:50052 \
  --memory-mib 8192 \
  --engine-commit "$(git -C /path/to/llama.cpp rev-parse HEAD)" \
  --ssh-host worker \
  --ssh-user kestrel \
  --identity-file /absolute/path/.ssh/kestrel_worker \
  --host-key "ssh-ed25519 AAAA..." \
  --remote-rpc-port 50052
```

The identity file must be absolute, owned by the current user, and mode `0600`
or stricter. Kestrel writes a short-lived restricted `known_hosts` file and
uses non-interactive SSH with strict host-key checking. Private key paths and
host-key material are omitted from status JSON.

## Preflight and run

```bash
kestrel nodes list
kestrel nodes doctor worker
kestrel nodes plan /models/model.gguf --node worker --json
kestrel run /models/model.gguf --node worker
```

`nodes doctor` proves that the forwarded endpoint speaks a compatible RPC
protocol and reports live devices and memory. Kestrel also fails closed when a
configured worker commit differs from the coordinator engine.

`nodes plan` is a coarse weights-only accelerator fit. llama.cpp still owns
the final tensor, KV-cache, host-RAM, and disk placement, so a successful plan
is evidence rather than a guarantee that the entire working set fits.

Kestrel supervises the SSH tunnel for the complete run or serve lifetime and
tears down the process group on normal exit, failure, timeout, or Ctrl-C.

## Manual tunnels

A disposable `ssh -N -L ...` tunnel can still be paired with a direct loopback
node entry. Kestrel cannot authenticate or supervise that external process, so
managed SSH is preferred for `run`, `chat`, and `serve`.

Direct non-loopback RPC requires the explicit `--allow-insecure-rpc`
acknowledgement on both inventory and run commands. It provides no encryption,
authentication, or hostile-worker isolation and is not recommended.
