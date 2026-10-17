#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 1 ]]; then printf '%s\n' 'Usage: bash scripts/resume_vim_v2.sh rafdb__native_ce' >&2; exit 2; fi
case "$1" in *[!a-z0-9_]*|'') printf '%s\n' 'Invalid config ID' >&2; exit 2;; esac
bash "$project/scripts/vim_v2.sh" resume --config "$project/configs/vision_mamba_clip/v2/$1.json" --plan "$project/configs/vision_mamba_clip/v2/plan.json"
