#!/usr/bin/env bash
set -euo pipefail
python run_ablation_grid.py --base-config config_v2.json --out-root results_ablation
