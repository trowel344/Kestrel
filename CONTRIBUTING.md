# Contributing to Kestrel

Thanks for taking the time to contribute. This document covers how to report
issues and get in touch.

## Reporting an issue

Before opening a new issue, please:

1. **Search** the existing issues to avoid duplicates.
2. **Check the docs** — `kestrel doctor` output and `kestrel <command> --help`
   often clarify whether something is a bug or a usage question.
3. Read the [README](README.md#command-reference) to confirm you are using the
   expected invocation.

### What to include

A useful bug report includes:

- **Kestrel version** — `kestrel --version`
- **Environment** — full output of `kestrel doctor` (Python version, GPU, RAM,
  and llama.cpp capabilities). This is the single most useful diagnostic.
- **The exact command** you ran and any relevant `--help` flags.
- **Expected vs. actual** behavior.
- **Steps to reproduce**, ideally with a model name or GGUF path.

Security-sensitive findings should be reported by email (see below) rather than
in a public issue.

## Getting in touch

- **Project maintainer:** trowel344@gmail.com
- For private or security-related reports, use the email address above instead
  of opening a public issue.

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[convert]'   # add conversion extras if you need torch
```

Run the linter and test suite before submitting changes:

```bash
pip install -e '.[dev,convert]'   # pytest + ruff + conversion extras
ruff check kestrel/ tests/
python -m pytest                   # hermetic; runs fully offline
```

## Submitting changes

- Keep the package dependency-light: do not add imports at module top level if
  they pull in optional heavy dependencies (`torch`, etc.). Use lazy imports.
- Open a pull request against the default branch and reference any related issue.