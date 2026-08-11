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

The supported default is a worker bound to `127.0.0.1` and an authenticated
SSH local-forward whose coordinator endpoint is also loopback. Keep ordinary
SSH host-key checking enabled. Never expose a worker on `0.0.0.0`, the public
internet, or an untrusted shared network. `--allow-insecure-rpc` is an explicit
experimental escape hatch, not a security control.

Before launch, Kestrel verifies a bounded RPC HELLO, protocol compatibility,
device enumeration, live device memory, and the worker's configured pinned
llama.cpp commit. These checks reject random TCP listeners and known mismatches;
they do not authenticate a malicious peer or attest the remote binary. A lost
node fails the llama.cpp run—Kestrel does not silently move the workload back
to local execution.

## Supported versions

Security fixes are applied to the latest release on the `main` branch.
