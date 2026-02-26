#!/usr/bin/env bash
# ============================================================
#  Spanish Podcast Difficulty Analyzer – Setup & Run Script
#  Creates a virtualenv, installs dependencies, and runs
#  the analyzer with any arguments you pass in.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
PYTHON="${PYTHON:-python3}"

# ----------------------------------------------------------
#  1. Find Python
# ----------------------------------------------------------
if ! command -v "$PYTHON" &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON="python"
    else
        echo "[ERROR] Python not found. Install Python 3.10+ and ensure it's on PATH."
        exit 1
    fi
fi

echo "[SETUP] Using Python: $($PYTHON --version 2>&1)"

# ----------------------------------------------------------
#  2. Create virtualenv if it doesn't exist
# ----------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[SETUP] Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ----------------------------------------------------------
#  3. Activate virtualenv
# ----------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ----------------------------------------------------------
#  4. Install / upgrade dependencies
# ----------------------------------------------------------
STAMP="$VENV_DIR/.deps_installed"

needs_install=0
if [ ! -f "$STAMP" ]; then
    needs_install=1
elif [ "$REQUIREMENTS" -nt "$STAMP" ]; then
    needs_install=1
fi

if [ "$needs_install" -eq 1 ]; then
    echo "[SETUP] Installing Python dependencies ..."
    python -m pip install --upgrade pip >/dev/null 2>&1
    python -m pip install -r "$REQUIREMENTS"

    echo "[SETUP] Downloading spaCy Spanish model ..."
    python -m spacy download es_core_news_lg || {
        echo "[WARNING] spaCy model download failed. Structural analysis may not work."
    }

    touch "$STAMP"
    echo "[SETUP] Dependencies installed successfully."
else
    echo "[SETUP] Dependencies up to date."
fi

# ----------------------------------------------------------
#  5. Run the analyzer (pass through all script arguments)
# ----------------------------------------------------------
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage:  ./analyze.sh [OPTIONS] FEED_URL [FEED_URL ...]"
    echo ""
    echo "Run \"./analyze.sh --help\" for full options."
    echo ""
    exit 0
fi

echo ""
echo "[RUN] Starting analyzer ..."
echo ""
python "$SCRIPT_DIR/main.py" "$@"
