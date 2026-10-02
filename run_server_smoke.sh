#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WL_PYTHON:-/tatu/venv/gob5/bin/python}
if [[ ! -x "$python_bin" ]]; then
  printf 'Python not found: %s\nSet WL_PYTHON to the original scientific interpreter.\n' "$python_bin" >&2
  exit 1
fi
exec "$python_bin" -B -u "$repo_dir/tools/server_smoke.py" --paths "$repo_dir/config_paths.server.toml" "$@"
