# V2.0 Follow-up Results

Committed summary artifacts:

- `summary_policy_macro.csv`: equal-cell macro policy comparison.
- `policy_by_workload.csv`: workload-level policy mean Goodput, TTFT P99, TPOT P99 and SLO attainment.
- `paired_delta_summary.csv`: paired Goodput differences and bootstrap 95% CI.
- `realized_rho_audit.csv`: target vs realized offered load, request counts and trace horizons.
- `sloa_ablation_summary.csv`: SLOA component ablation.
- `chunk_sweep_summary.csv`: fixed Chunk sweep.
- `sloa_parameter_sweep_summary.csv`: risk threshold and gamma sweep.

The main matrix contains 720 run rows across 72 paired traces. Full run-level CSVs, all figures (PNG/SVG), per-seed files and the complete working source tree are in the full project ZIP supplied in the conversation; they are not all mirrored in this GitHub repository.

All results are simulator outputs; the target 2×RTX 3090 hardware has not been benchmarked directly. W7-Adversarial is not included in the main matrix because 40k-token whales/long outputs are expensive and small traces weaken P99 confidence.
