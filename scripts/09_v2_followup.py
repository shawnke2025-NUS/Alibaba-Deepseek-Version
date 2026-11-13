"""V2.0 follow-up experiments: multi-load/multi-workload paired comparison.

All measurements are simulator outputs. No real GPU benchmark is implied.
"""
from __future__ import annotations
import os, sys, time, json, platform, argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.config import RTX3090_TP2, QWEN3_32B_INT4, ENGINE, WORKLOADS, SLO_POLICY
from src.experiment import build_ctx, make_trace, run_one, capacity_tok_s, mean_context, SLOA_Variant
from src.schedulers import FCFSFull, FixedChunk, SlowEave, SLAI, RAD, SLOA

class SLOA_V1_Emulated(SLOA):
    """Emulate the previously audited V1 control path for paired comparison."""
    name = "SLOA-V1-emulated"
    def reset(self):
        super().reset()
        self.budget_target = 4096.0
    def _update_knobs(self, st):
        self.util_ema = 0.5
        if self.util_ema < self.alpha:
            self.chunk_target = min(self.chunk_max, self.chunk_target * (1 + self.gamma))
            self.budget_target = min(8192.0, self.budget_target * (1 + self.gamma / 2))
        else:
            self.chunk_target = max(self.chunk_min, self.chunk_target * (1 - self.gamma / 2))
            self.budget_target = max(1024.0, self.budget_target * (1 - self.gamma / 4))
        if st.kv_util > 0.90:
            self.budget_target = max(1024.0, self.budget_target * 0.8)
    def _tilt(self, st):
        self._last_risk = (0.0, 0.0)
        return "balanced"

OUT = Path(ROOT) / "results" / "v2_followup"
OUT.mkdir(parents=True, exist_ok=True)

def make_trace_calibrated(spec, rho, cm, seed, target_n=160, dur_min=180.0, dur_max=6000.0):
    """Create a finite trace whose realized token-load ratio equals target rho."""
    spec2, reqs, rate, desc = make_trace(spec, rho, cm, seed, target_n=target_n,
                                         dur_min=dur_min, dur_max=dur_max)
    if not reqs:
        raise RuntimeError(f"empty generated trace: {spec.name}, rho={rho}, seed={seed}")
    total_tokens = sum(r.prompt_len + r.output_len for r in reqs)
    cap = capacity_tok_s(cm, mean_context(reqs))
    duration_target = total_tokens / max(rho * cap, 1e-9)
    scale = duration_target / max(spec2.duration_s, 1e-9)
    for r in reqs:
        r.arrival *= scale
        r.queue_since = r.arrival
    spec2 = replace(spec2, duration_s=duration_target, warmup_s=spec2.warmup_s * scale)
    desc["duration_s"] = duration_target
    desc["offered_rate_req_s"] = len(reqs) / duration_target
    desc["offered_tokens_per_s"] = total_tokens / duration_target
    rate = len(reqs) / duration_target
    actual = desc["offered_tokens_per_s"] / max(cap, 1e-9)
    return spec2, reqs, rate, desc, actual, 1

POLICIES = [
    ("FCFS+Full", lambda: FCFSFull()),
    ("Fixed-64+SPF", lambda: FixedChunk(64, spf=True)),
    ("Fixed-256+SPF", lambda: FixedChunk(256, spf=True)),
    ("Fixed-512+SPF", lambda: FixedChunk(512, spf=True)),
    ("Fixed-1024+SPF", lambda: FixedChunk(1024, spf=True)),
    ("SLOWEAVE", lambda: SlowEave(max_chunk=4096, margin_us=0.0)),
    ("SLAI", lambda: SLAI()),
    ("RAD", lambda: RAD(b_col=32, b_lcm=128, cycle_B=16)),
    ("SLOA-V1-emulated", lambda: SLOA_V1_Emulated()),
    ("SLOA-V2", lambda: SLOA()),
]

def run_main(quick=False, workload_filter=None, seed_filter=None):
    ctx = build_ctx(with_predictor="analytic", margin=0.10)
    workloads = [workload_filter] if workload_filter else (["W3-mixed"] if quick else ["W1-chat", "W2-chat-heavy", "W3-mixed", "W4-long", "W5-bursty", "W6-flashcrowd"])
    rhos = [0.65, 0.85] if quick else [0.25, 0.65, 0.85, 1.20]
    seeds = [seed_filter] if seed_filter is not None else ([11] if quick else [11, 12, 13])
    policy_list = POLICIES
    if workload_filter == "W7-adversarial":
        workloads = ["W7-adversarial"]
        workloads_extra = []
        rhos = [0.85, 1.20] if not quick else [0.85]
        seeds = [seed_filter if seed_filter is not None else 11]
        policy_list = [p for p in POLICIES if p[0] in {"Fixed-64+SPF", "RAD", "SLOA-V1-emulated", "SLOA-V2"}]
    elif not quick and workload_filter is None:
        workloads_extra = [("W7-adversarial", [0.85, 1.20])]
    else:
        workloads_extra = []
    rows = []
    run_tag = workload_filter or "all"
    run_path = OUT / (f"run_level_{run_tag}_seed{seed_filter}.csv" if seed_filter is not None else f"run_level_{run_tag}.csv")
    if run_path.exists():
        run_path.unlink()
    total = (len(workloads)*len(rhos)+sum(len(x[1]) for x in workloads_extra))*len(seeds)*len(policy_list)
    idx, t0 = 0, time.time()
    combos = [(w, rho) for w in workloads for rho in rhos]
    for w, rs in workloads_extra:
        combos.extend((w, rho) for rho in rs)
    for w, rho in combos:
        spec = WORKLOADS[w]
        for seed in seeds:
            max_duration = 10000.0 if w == "W4-long" else 6000.0
            trace_target_n = 40 if w == "W7-adversarial" else 160
            spec2, reqs, rate, desc, realized_rho, rho_cal_iters = make_trace_calibrated(
                spec, rho, ctx.cm_truth, seed, target_n=trace_target_n, dur_min=180.0, dur_max=max_duration)
            for pname, factory in policy_list:
                m = run_one(ctx, spec2, reqs, factory(), seed)
                m.update({
                    "policy": pname, "workload": w, "rho_target": rho,
                    "rho_realized": realized_rho, "rho_calibration_iters": rho_cal_iters, "seed": seed,
                    "offered_rate_req_s_target": rate,
                    "offered_rate_req_s_realized": desc["n_requests"] / spec2.duration_s,
                    "offered_tokens_s": desc["offered_tokens_per_s"],
                    "mean_context": mean_context(reqs), "n_requests": desc["n_requests"],
                    "prompt_p50": desc["prompt_p50"], "prompt_p99": desc["prompt_p99"],
                    "output_p50": desc["output_p50"], "duration_s": spec2.duration_s,
                })
                rows.append(m); idx += 1
                pd.DataFrame(rows).to_csv(run_path, index=False)
                print(f"[{idx}/{total}] {w} rho={rho:.2f} seed={seed} policy={pname} gp={m.get('goodput_req_s', float('nan')):.3f}", flush=True)
    df = pd.DataFrame(rows)
    metrics = ["goodput_req_s", "throughput_req_s", "throughput_tok_s", "ttft_p50_ms", "ttft_p99_ms",
               "tpot_reqp99_p50_ms", "tpot_reqp99_p99_ms", "slo_attainment", "reject_rate",
               "avg_kv_frac", "avg_util", "preemptions", "iterations"]
    agg = df.groupby(["workload", "rho_target", "policy"])[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else c for c in agg.columns]
    agg.to_csv(OUT / "summary_mean_std.csv", index=False)
    winners = df.groupby(["workload", "rho_target", "policy"], as_index=False)["goodput_req_s"].mean()
    winners["rank"] = winners.groupby(["workload", "rho_target"])["goodput_req_s"].rank(method="min", ascending=False)
    winners.to_csv(OUT / "goodput_ranking.csv", index=False)
    meta = {"version":"V2.0-followup","evidence_type":"simulator-generated","real_gpu_measurement":False,
            "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "workloads":sorted(df.workload.unique().tolist()),"rhos":rhos,"seeds":seeds,
            "policies":[p[0] for p in policy_list],"n_runs":len(df),
            "duration":"finite-trace time-axis scaled to target rho; target_n=160 before scaling",
            "model":"Qwen3-32B W4A16 analytic cost model; TP=2; 2xRTX3090 assumed","margin":0.10}
    (OUT/"experiment_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    return df

def run_ablation_and_chunk(seed_filter=None):
    ctx = build_ctx(with_predictor="analytic", margin=0.10)
    spec = WORKLOADS["W3-mixed"]
    ab_rows, ch_rows, sens_rows = [], [], []
    for seed in ([seed_filter] if seed_filter is not None else [11, 12, 13]):
        spec2, reqs, rate, desc, realized_rho, rho_cal_iters = make_trace_calibrated(
            spec, 0.85, ctx.cm_truth, seed, target_n=160, dur_min=180, dur_max=6000)
        variants = [
            SLOA_Variant(label="SLOA-V2-full"),
            SLOA_Variant(tilt=False, label="no-dual-tilt"),
            SLOA_Variant(knobs=False, label="no-util-knobs"),
            SLOA_Variant(score=False, label="no-priority-score"),
            SLOA_Variant(constraint=False, label="no-TPOT-constraint"),
        ]
        for v in variants:
            m=run_one(ctx,spec2,reqs,v,seed); m.update({"variant":v.name,"seed":seed,"rho_target":0.85,"rho_realized":realized_rho}); ab_rows.append(m)
        for ch in [64,128,192,256,384,512,768,1024,1536,2048,3072,4096]:
            m=run_one(ctx,spec2,reqs,FixedChunk(ch,spf=True),seed); m.update({"chunk":ch,"seed":seed,"rho_target":0.85,"rho_realized":realized_rho}); ch_rows.append(m)
        for threshold in [0.60,0.70,0.80,0.90]:
            for gamma in [0.15,0.30,0.45]:
                policy=SLOA(risk_threshold=threshold,gamma=gamma)
                m=run_one(ctx,spec2,reqs,policy,seed); m.update({"risk_threshold":threshold,"gamma":gamma,"seed":seed,"rho_realized":realized_rho}); sens_rows.append(m)
    suffix=f"_seed{seed_filter}" if seed_filter is not None else ""
    pd.DataFrame(ab_rows).to_csv(OUT/f"sloa_ablation{suffix}.csv",index=False)
    pd.DataFrame(ch_rows).to_csv(OUT/f"chunk_sweep{suffix}.csv",index=False)
    pd.DataFrame(sens_rows).to_csv(OUT/f"sloa_parameter_sweep{suffix}.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--main-only",action="store_true")
    ap.add_argument("--ablations-only",action="store_true")
    ap.add_argument("--workload",choices=list(WORKLOADS.keys()))
    ap.add_argument("--seed",type=int,choices=[11,12,13])
    args=ap.parse_args()
    if not args.ablations_only: run_main(quick=args.quick,workload_filter=args.workload,seed_filter=args.seed)
    if not args.main_only: run_ablation_and_chunk(seed_filter=args.seed)

if __name__=="__main__": main()
