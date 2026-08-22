#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mamba="/data1/morrisq/dalalt1/miniforge3/bin/mamba"
python="/data1/morrisq/dalalt1/miniforge3/envs/mrnabench/bin/python"
port="${PORT:-4321}"

echo "Building the website on $(hostname) for http://localhost:${port}/"
SITE_BASE=/ SITE_URL="http://localhost:${port}" \
  "$mamba" run -n mrnabench npm --prefix "$repository_root/website" run build

echo "Serving the website on http://127.0.0.1:${port}/"
exec "$python" -m http.server "$port" \
  --bind 127.0.0.1 \
  --directory "$repository_root/website/dist"
