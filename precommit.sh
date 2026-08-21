#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    "")
        pytest_args=()
        ;;
    --quick)
        pytest_args=(--ignore=tests/models)
        ;;
    *)
        echo "Usage: $0 [--quick]" >&2
        exit 2
        ;;
esac

flake8 mrna_bench --ignore=D100,D104,E203
mypy mrna_bench --ignore-missing-imports

pytest "${pytest_args[@]}"
