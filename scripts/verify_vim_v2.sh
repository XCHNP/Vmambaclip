#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bash "$project/scripts/vim_v2.sh" verify --plan "$project/configs/vision_mamba_clip/v2/plan.json" "$@"
