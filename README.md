# Alibaba DeepSeek Version — Single-instance LLM Inference Simulator

**Current status:** V2.0 follow-up experiments completed; V3.0 is a research design proposal.

> **Evidence boundary:** all latency, throughput and Goodput values are simulator outputs. The target platform (2×RTX 3090, TP=2, Qwen3-32B W4A16) is an assumed configuration, not a completed real-hardware benchmark. The bundled reference profiling source comes from a different GPU platform.

## V2.0 result snapshot

- 720 paired main-matrix runs: 6 workloads × 4 load points × 3 seeds × 10 policies.
- 87 additional runs: SLOA ablation, fixed-chunk sweep, risk-threshold × gamma sweep.
- 221 corrected profiling grid points (46 decode / 60 prefill / 115 mixed), 19,890 synthetic profiling samples.
- 6 regression tests pass in the complete working package.

| Policy | Macro Goodput (req/s) | TTFT P99 (ms) | TPOT P99 (ms) |
|---|---:|---:|---:|
| SLOA-V1-emulated | 0.2188 | 69,436 | 670 |
| SLOA-V2 | 0.2179 | 67,844 | 602 |
| SLOWEAVE | 0.2131 | 81,027 | 653 |
| Fixed-64+SPF | 0.1797 | 100,784 | 896 |

The paired 95% bootstrap intervals for SLOA-V2 vs V1-emulated and SLOWEAVE include zero. V2 has **not** demonstrated a statistically significant Goodput improvement over those two stronger baselines. It does outperform Fixed-64+SPF in the macro paired comparison.

## Key documents

- [V2.0 follow-up experiment report](V2.0_FOLLOWUP_README.md)
- [V3.0 SLOA-MPC design proposal](V3_DESIGN.md)
- [V2.0 task checklist](V2.0_TODO.md)
- [V2.0 change log](V2.0_CHANGELOG.md)

## Repository contents

- `scripts/09_v2_followup.py`: V2.0 paired experiment runner.
- `scripts/10_finalize_v2.py`: aggregation, paired bootstrap and figure generation.
- `scripts/run_v2_matrix.sh`: reproducible W1–W6 matrix and ablation runner.
- `tests/test_v2_feedback.py`: V2.0 regression tests.
- `patches/V2.0_core.patch`: core source changes relative to the original V1 source tree.
- `results/v2_followup/`: macro summary CSVs, paired confidence intervals, rho audit and ablation/chunk/parameter summaries.

**Source-tree note:** this GitHub repository was initially empty. It now contains the V2.0 patch, experiment harness, tests, reports and result summaries; it does not yet mirror every unchanged V1 source module or the full raw run-level dataset. The complete working source tree, raw CSVs, figures and reference materials are included in the accompanying project ZIP provided in this conversation.

## Reproduce from the complete project bundle

After extracting the complete project ZIP (which contains the V1 base source plus V2.0 modifications):

```bash
pip install -r requirements.txt
python -m compileall -q src scripts tests
python -m pytest -q tests
bash scripts/run_v2_matrix.sh
python scripts/10_finalize_v2.py
```

W7-Adversarial is an optional stress test: 40k-token whales and long outputs make it expensive, and small traces make P99 estimates less stable.

## Main findings motivating V3

1. The TPOT hard constraint is the most effective current SLOA component; removing it materially worsens TPOT and Goodput.
2. The current linear priority score and discrete dual tilt hurt the W3 ablation result; V3 should replace them with request-level deadline slack and candidate-aware MPC.
3. V2 almost never enters TPOT-first because its risk estimate uses Decode-only runtime and underestimates Prefill interference.
4. Fixed-64 wins the W3 chunk sweep, showing V2 has not beaten a tuned static policy in every workload.

V3 is a proposal, not an implemented or benchmarked result.
