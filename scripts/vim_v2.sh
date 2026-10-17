#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
export PYTHONPATH="$project/src:$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
runtime=${VIM_V2_PYTHON:-/root/miniconda3/envs/clip/bin/python}
if [[ ! -x "$runtime" ]]; then printf '%s\n' "Python runtime missing: $runtime" >&2; exit 2; fi
exec "$runtime" -u -m vim_v2 "$@"
