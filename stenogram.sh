#!/usr/bin/env bash
# Обёртка: запускает stenogram из локального venv, из любой директории.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/venv/bin/python" -m stenogram "$@"
