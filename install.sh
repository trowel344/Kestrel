#!/usr/bin/env bash
#
# Kestrel installer — creates a virtual environment and installs the package.
#
# Usage:
#   ./install.sh                 install Kestrel into .venv
#   ./install.sh --convert       also install the NVFP4 conversion extras (torch)
#   ./install.sh --dir ~/.kestrel
#   ./install.sh --help
#
# Kestrel is installed into an isolated virtual environment so it never
# conflicts with your system Python packages.

set -euo pipefail

INSTALL_DIR=".venv"
WITH_CONVERT=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --convert) WITH_CONVERT=1 ;;
    --dir) INSTALL_DIR="${2:?--dir requires a path}"; shift ;;
    --help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required but was not found on PATH" >&2
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
MAJOR="${PYTHON_VERSION%%.*}"
MINOR="${PYTHON_VERSION##*.}"
if (( MAJOR < 3 )) || (( MAJOR == 3 && MINOR < 11 )); then
  echo "error: Kestrel requires Python 3.11+; found $PYTHON_VERSION" >&2
  exit 1
fi

echo "==> Creating virtual environment: $INSTALL_DIR"
python3 -m venv "$INSTALL_DIR"

PIP="$INSTALL_DIR/bin/pip"
KESTREL="$INSTALL_DIR/bin/kestrel"
echo "==> Upgrading pip"
"$PIP" install --quiet --upgrade pip

echo "==> Installing Kestrel"
if [[ $WITH_CONVERT -eq 1 ]]; then
  "$PIP" install --quiet ".[convert]"
else
  "$PIP" install --quiet "."
fi

echo "==> Verifying installed kestrel launcher"
VERSION_OUT="$("$KESTREL" --version 2>&1)" || true
if ! printf '%s\n' "$VERSION_OUT" | grep -E '^kestrel [0-9]+\.[0-9]+\.[0-9]+' >/dev/null 2>&1; then
  echo "error: 'kestrel --version' did not report a 'kestrel <semver>' line; got:" >&2
  printf '%s\n' "$VERSION_OUT" >&2
  echo "The pip install may have failed. Check the pip output above." >&2
  exit 1
fi

echo
echo "Kestrel installed successfully."
echo
echo "To use it:"
echo "  $KESTREL doctor"
echo "  $KESTREL setup --model /path/to/model.gguf"
echo "  $KESTREL chat"
echo
echo "Tip: add the Kestrel bin to your PATH for the short name:"
echo "  export PATH=\"$(cd "$(dirname "$INSTALL_DIR")" && pwd)/$(basename "$INSTALL_DIR")/bin:\$PATH\""
