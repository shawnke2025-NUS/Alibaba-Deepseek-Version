# V2.0 Follow-up Results

- `run_level_all.csv`: 720 main-matrix run rows (6 workloads × 4 rho × 3 seeds × 10 policies).
- `summary_by_workload_load.csv`: per-workload/load/policy mean and standard deviation.
- `summary_policy_macro.csv`: equal-cell macro policy comparison.
- `paired_delta_summary.csv`: paired Goodput differences and bootstrap 95% CI.
- `sloa_ablation_summary.csv`: SLOA ablation.
- `chunk_sweep_summary.csv`: fixed Chunk sweep.
- `sloa_parameter_sweep_summary.csv`: risk threshold and gamma sweep.
- `realized_rho_audit.csv`: target vs realized offered load, request counts and trace horizons.
- `figures/`: PNG and SVG charts.

The main matrix has 72 paired traces. All results are simulator outputs; the target 2×RTX 3090 hardware has not been benchmarked directly. W7-Adversarial is not included in the main matrix because 40k-token whales/long outputs are expensive and small traces weaken P99 confidence.
