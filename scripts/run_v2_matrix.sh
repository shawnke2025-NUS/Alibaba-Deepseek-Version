#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for w in W1-chat W2-chat-heavy W3-mixed W4-long W5-bursty W6-flashcrowd; do
  for seed in 11 12 13; do
    echo "===== START $w seed=$seed $(date -Is) ====="
    python -u scripts/09_v2_followup.py --workload "$w" --seed "$seed" --main-only
    echo "===== DONE $w seed=$seed $(date -Is) ====="
  done
done
# W7-Adversarial is an optional stress test; long whale prompts and outputs make it costly.
for seed in 11 12 13; do
  echo "===== ABLATION seed=$seed $(date -Is) ====="
  python -u scripts/09_v2_followup.py --ablations-only --seed "$seed"
done
