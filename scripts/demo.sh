#!/usr/bin/env bash
# One-command local demo launcher (macOS / Linux).
#
# Creates a virtualenv, installs the dashboard dependencies, seeds the local
# database from the bundled Starlink snapshot (so it works even with no
# network), and launches the Streamlit dashboard at http://localhost:8501.
#
# Usage:
#   ./scripts/demo.sh
#
# Safe to re-run: the venv and seeded DB are reused if they already exist.

set -euo pipefail

# Move to the project root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv)"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies (this may take a minute the first time)"
python -m pip install --upgrade pip >/dev/null
pip install -e ".[dashboard]" >/dev/null

echo "==> Seeding the local database from the bundled Starlink snapshot"
python scripts/seed_demo.py

echo ""
echo "==> Launching the dashboard at http://localhost:8501  (Ctrl+C to stop)"
echo ""
spacetrack dashboard
