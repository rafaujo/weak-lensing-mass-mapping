#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WL_PYTHON:-/tatu/venv/gob5/bin/python}
exec "$python_bin" -B -u "$repo_dir/tools/run_pipeline.py" --paths "$repo_dir/config_paths.server.toml" --environments "$repo_dir/config_environments.server.toml" "$@"
