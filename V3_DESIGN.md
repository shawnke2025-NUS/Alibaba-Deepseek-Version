# V3.0 Design Proposal: SLOA-MPC

> Proposal only; not yet implemented or benchmarked.

## Motivation from V2

- SLOA-V2 vs V1-emulated Goodput delta: -0.0009 req/s, paired bootstrap 95% CI [-0.0129, 0.0067].
- SLOA-V2 vs SLOWEAVE: +0.0048 req/s, CI [-0.0127, 0.0281].
- W3-Mixed/rho=0.85: Fixed-64 Goodput 0.185 vs SLOA Full 0.162.
- Removing current linear Priority Score improves Goodput from 0.162 to 0.186; removing current Dual Tilt improves it to 0.173.
- Removing TPOT hard constraint drops Goodput to 0.113 and worsens TPOT P99 from 645ms to 1,720ms.
- TPOT-first is triggered on only ~0.4% of SLOA ticks; Decode-only risk prediction underestimates Prefill interference.

## Proposed architecture

1. **Request-level deadline slack.** For Decode request i:

   `Slack_TPOT_i = last_token_time + TPOT_SLO_i - now - predicted_next_mixed_iteration`

   For Waiting/Prefilling request i, estimate queue wait plus remaining multi-chunk prefill completion time to obtain `Slack_TTFT_i`.

2. **P99 upper-confidence runtime.**

   `T_safe(a) = T_hat_P99(a) + kappa * prediction_sigma(a) + positive_residual_bound`

   Residual calibration must be trained on independent profiling data, not on the same synthetic data used to generate predictions.

3. **Joint action search.** Enumerate a small discrete action set `a=(DecodeBatch, PrefillTokenBudget, Chunk)`. Reject actions violating KV capacity or hard TPOT constraints. Do not assume runtime is strictly monotonic near GPU tile boundaries.

4. **Request selection.** Sort Decode by TPOT Slack and Prefill by predicted TTFT Slack; use bounded top-K candidate selection and Aging to prevent starvation. Replace the current hand-weighted progress score.

5. **Objective.** Maximize estimated request-level SLO success plus useful token throughput, minus preemption and starvation cost. Report TTFT/TPOT Pareto, not Goodput alone.

6. **Continuous risk controller.** Blend TTFT and TPOT weights from their risk/slack difference, apply EMA, deadband and minimum hold time. If both risks are low, remain Balanced.

7. **Bottleneck-aware feedback.** Track compute saturation, memory bandwidth saturation, KV pressure, queue pressure and SLO pressure separately. Dense FLOPs utilization is not a substitute for NVML SM utilization.

## Pseudocode

```text
observe queues, KV blocks and last iteration telemetry
update P99 residual calibration and bottleneck estimates
compute per-request TTFT/TPOT slack
build Decode candidates by TPOT slack and Prefill candidates by TTFT slack
for each candidate (DecodeBatch, PrefillBudget, Chunk):
    predict safe mixed-iteration P99 runtime
    reject KV-infeasible or hard-TPOT-infeasible actions
    score expected SLO-goodput + utilization - preemption/starvation cost
choose best feasible action, execute, and update residual feedback
```

## Validation requirements

- Compare V1-emulated / V2 / V3, SLOWEAVE, SLAI, RAD, Fixed-64 and per-workload best-fixed.
- Use common traces and at least 5 seeds; tune only on training seeds and report held-out seeds.
- Report paired bootstrap 95% CI, SLO attainment, reject/admission rate, TTFT/TPOT P50/P90/P99, KV/GPU metrics.
- Do not claim improvement if Goodput increases by rejecting more requests or if latency Pareto regresses unacceptably.
- Verify simulation findings with target-GPU profiling and real inference-framework replay.

The central V3 hypothesis is a lightweight online MPC scheduler that jointly selects requests and Prefill/Decode resources under uncertainty-aware P99 runtime constraints.
