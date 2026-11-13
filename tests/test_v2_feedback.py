import numpy as np
from types import SimpleNamespace

from src.config import ENGINE, QWEN3_32B_INT4, RTX3090_TP2, SLO_POLICY, WORKLOADS
from src.costmodel import CostModel
from src.engine import Engine
from src.schedulers import SLOA
from src.workload import generate


class RecordingSLOA(SLOA):
    def __init__(self):
        super().__init__()
        self.seen_feedback = []

    def plan(self, st):
        self.seen_feedback.append((st.last_iter_us, st.last_util, st.last_tokens))
        return super().plan(st)


def test_engine_carries_previous_iteration_telemetry():
    cm = CostModel(RTX3090_TP2, QWEN3_32B_INT4, ENGINE)
    spec = WORKLOADS["W1-chat"]
    spec = spec.__class__(**{**spec.__dict__, "duration_s": 120.0, "warmup_s": 0.0})
    reqs, _ = generate(spec, rate=0.15, slo=SLO_POLICY, seed=7)
    scheduler = RecordingSLOA()
    eng = Engine(cm, cm, ENGINE, SLO_POLICY, seed=9, margin=0.05)
    eng.run(reqs, spec.duration_s, spec.warmup_s, scheduler)
    assert len(scheduler.seen_feedback) > 1
    assert scheduler.seen_feedback[0] == (0.0, 0.0, 0)
    assert any(t > 0 and u >= 0 and n > 0 for t, u, n in scheduler.seen_feedback[1:])


def test_sloa_reset_restores_controller_state():
    s = SLOA()
    s.chunk_target = 2048
    s.budget_target = 7000
    s.util_ema = 0.9
    s.tilt = "TPOT-first"
    s.tilt_log.append((1, "TPOT-first", 0.1, 0.8, 2048, 7000))
    s.reset()
    assert s.chunk_target == 512
    assert s.budget_target == 1536
    assert s.util_ema == 0.5
    assert s.tilt == "balanced"
    assert not s.tilt_log


def test_profile_mixed_loop_keeps_each_context_point():
    from pathlib import Path
    text = Path("src/profile.py").read_text(encoding="utf-8-sig")
    expected = '''for ctx in m["mixed_ctx"]:
                    _p = self._point("mixed", n, c, ctx, ctx * 0.6)
                    if _p is not None:
                        out.append(_p)'''
    assert expected in text


def test_zero_risk_tilt_is_balanced_not_unconditionally_ttft_first():
    s = SLOA()
    st = SimpleNamespace(decode=[], waiting=[], prefilling=[], now=0.0,
                         cfg=SimpleNamespace(max_num_seqs=128))
    assert s._tilt(st) == "balanced"


def test_step_arrival_trace_continues_after_original_profile_horizon():
    from dataclasses import replace
    spec = replace(WORKLOADS["W6-flashcrowd"], duration_s=1200.0, warmup_s=0.0)
    reqs, desc = generate(spec, rate=0.2, slo=SLO_POLICY, seed=17)
    assert reqs
    assert max(r.arrival for r in reqs) > 600.0


def test_followup_trace_load_is_exactly_normalized():
    import importlib.util
    spec_mod = importlib.util.spec_from_file_location("v2_followup", "scripts/09_v2_followup.py")
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    from src.experiment import build_ctx
    ctx = build_ctx(margin=0.10)
    _, reqs, _, _, realized, _ = mod.make_trace_calibrated(
        WORKLOADS["W3-mixed"], 0.85, ctx.cm_truth, seed=23,
        target_n=80, dur_min=180, dur_max=6000)
    assert reqs
    assert abs(realized - 0.85) < 1e-8
