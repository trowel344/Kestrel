# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security-related problems. Report
vulnerabilities privately by email:

- **trowel344@gmail.com**

Include as much detail as you can: the Kestrel version, `kestrel doctor`
output, the steps to reproduce, and the impact. Reports are acknowledged
promptly and handled confidentially.

## Scope

The following are in scope:

- `kestrel/` package code and the `kestrel` command-line tool
- Build and install tooling (`install.sh`, `pyproject.toml`)
- Handling of model files, config, and environment variables

Out of scope:

- Third-party dependencies (report those to their own projects)
- The llama.cpp binaries Kestrel invokes (report those upstream)

## Local binary trust

Kestrel only auto-discovers llama.cpp executables and GGML shared libraries
under the current user's home directory. Temporary or system-wide builds must
be selected explicitly with `KESTREL_LLAMA_CPP_DIR` or
`KESTREL_GGML_BASE_LIB`; Kestrel does not implicitly execute or load artifacts
from world-writable `/tmp` paths. Model aliases follow the same rule and can be
overridden explicitly with their documented environment variable.

## Experimental llama.cpp RPC nodes

llama.cpp RPC does not provide authentication or encryption. Treat every RPC
worker as fully trusted: it receives model data and participates in inference.
Kestrel does not discover LAN workers, start remote daemons, install SSH keys,
or make an exposed RPC socket safe.

The supported default is a worker bound to `127.0.0.1` and a Kestrel-managed,
authenticated SSH local-forward whose coordinator endpoint is also loopback.
Managed nodes require an explicitly pinned public host key and a current-user,
0600-or-stricter identity file. Kestrel uses an argument vector (never a shell),
`BatchMode`, `IdentitiesOnly`, `StrictHostKeyChecking`, and
`ExitOnForwardFailure`; it supervises and reaps the tunnel for the complete
llama.cpp child lifetime. Private-key paths and key material are not passed to
llama.cpp or emitted in node JSON. Managed mode currently requires a Linux
coordinator so Kestrel can verify the listening socket belongs to its SSH
child through `/proc` rather than trusting a pre-existing local listener.

Verify a worker host key out of band before registering it. Never expose a
worker on `0.0.0.0`, the public internet, or an untrusted shared network.
`--allow-insecure-rpc` is an explicit experimental escape hatch, not a
security control, and managed SSH is not a hostile-worker solution: the worker
receives model data and can return incorrect or malicious protocol responses.

The repository includes a starting-point
[`contrib/kestrel-rpc-worker.service`](contrib/kestrel-rpc-worker.service)
unit. Install it only after replacing the paths and allowing the exact GPU
device nodes for the worker. Create a dedicated account with no sudo access,
disable SSH agent forwarding, and keep the private key on the coordinator.
The unit is sandbox guidance, not a universal production profile; validate it
with the installed CUDA/ROCm driver and your local llama.cpp build.

Restrict the coordinator's authorized key as well. Obtain the worker host key
from a trusted console—not an unauthenticated `ssh-keyscan` over the network.
For OpenSSH, use an entry along the following lines (replace the key and
account) and configure `sshd` with `AllowTcpForwarding local`:

```text
restrict,port-forwarding,permitopen="127.0.0.1:50052" ssh-ed25519 AAAA... kestrel-coordinator
```

This prevents the key used by Kestrel from becoming a general-purpose SSH
forwarding credential. Keep the worker account free of sudo and do not enable
agent forwarding.

Before launch, Kestrel verifies a bounded RPC HELLO, protocol compatibility,
device enumeration, live device memory, and the worker's configured pinned
llama.cpp commit. These checks reject random TCP listeners and known mismatches;
they do not authenticate a malicious peer or attest the remote binary. A lost
node fails the llama.cpp run—Kestrel does not silently move the workload back
to local execution.

## Coding-agent integrations

`kestrel agents` runs a dedicated llama-server on `127.0.0.1` and enables
llama.cpp API-key authentication with a random credential stored in a
current-user 0600 file. Generated Pi, Oh My Pi, Codex, Claude Code, and
OpenCode profiles refer to a command or environment variable that supplies
the key; Kestrel does not embed the credential in their configuration.
Managed process state is also private and includes an ownership token checked
against `/proc` before Kestrel signals a process. A stale or foreign PID fails
closed.

Authentication protects inference routes from unrelated local clients that do
not have the key. Upstream llama-server deliberately leaves health, model
metadata, and UI assets public; Kestrel therefore keeps the whole listener on
loopback. Authentication does not sandbox the coding agent. The launched agent
can read, edit, execute, and sometimes access the network with the permissions
of your user account, subject to that client's own approval and sandbox
settings. Use version control, keep approval prompts enabled for risky work,
and do not treat a local model as trusted merely because it runs on-device.

Kestrel-owned Codex, Claude Code, and OpenCode files are separate overlays
that do not rewrite global defaults, and are removed only when their ownership
digest still matches. The clients may still layer their global approval,
sandbox, and tool settings around those overlays.
Pi and Oh My Pi both use mutable provider inventories, so Kestrel modifies
only the `providers.kestrel` entry in each format and uses a sidecar digest to
prevent overwriting or removing a pre-existing or user-modified entry.

Direct `kestrel serve` remains loopback-first. A non-loopback bind is rejected
without a private API-key file. An API key is not transport encryption: put a
TLS reverse proxy in front of any intentionally remote HTTP endpoint, or use
an authenticated SSH tunnel.

Managed coding-agent start/stop currently requires Linux because ownership is
verified through `/proc/<pid>/cmdline`; other platforms fail before spawning a
managed process. Direct `kestrel serve` remains available there.

## Supported versions

Security fixes are applied to the latest release on the `main` branch.
