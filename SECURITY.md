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

## Supported versions

Security fixes are applied to the latest release on the `main` branch.
