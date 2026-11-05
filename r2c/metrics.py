"""Metrics: REC (Eq. 5), COV (Eq. 7), AND, CDD — and the E1 run protocol."""
import numpy as np

from .config import R2CParams
from .controller import R2CController
from .environment import SwarmEnvironment


def algebraic_connectivity(adj: np.ndarray) -> float:
    lap = np.diag(adj.sum(1)) - adj
    evals = np.linalg.eigvalsh(lap)
    return float(evals[1]) if len(evals) > 1 else 0.0


def run_trial(p: R2CParams, failure_ratio: float, deployment: str,
              controller: R2CController | None = None,
              use_ise: bool = True) -> dict:
    """One trial of the paper protocol (guide §4): deploy, run to t1,
    inject failures, run to T, record recovery and topology metrics."""
    env = SwarmEnvironment(p, p.n_random if deployment == "random" else p.n_grid,
                           deployment)
    ctrl = controller if controller is not None else R2CController(p)
    t_rec, recovered = None, False
    cov_sum, steps = 0.0, 0

    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(failure_ratio)
        v = ctrl.velocity(env, t) if use_ise else np.zeros_like(env.pos)
        env.step(v)
        cov_sum += env.coverage()
        steps += 1
        if t > p.t1 and not recovered and env.connected():
            t_rec = t - p.t1
            recovered = True

    adj = env.active_adjacency()
    deg = adj.sum(1)
    deg_hist = np.histogram(deg, bins=np.arange(0, 41))[0] \
        / max(env.active.sum(), 1)     # fixed 0..40 support so trials align
    return {
        # REC (Eq. 5): T_rec = (t_r - t1)/T; 1.0 if never recovered
        "REC": (t_rec / p.T) if recovered else 1.0,
        "recovered": float(recovered),
        "COV": cov_sum / steps,                 # time-averaged coverage (Eq. 7)
        "COV_final": env.coverage(),
        "AND": float(deg.mean()),               # average node degree
        "CDD": np.cumsum(deg_hist),             # cumulative degree distribution
        "lambda2_final": algebraic_connectivity(adj),
    }


def run_metrics(p: R2CParams, failure_ratio: float, deployment: str,
                trials: int = 50, ablation: str | None = None) -> dict:
    """Paper protocol: `trials` independent trials, report mean +- std."""
    keys = ["REC", "COV", "COV_final", "AND", "lambda2_final", "recovered"]
    stats = {k: [] for k in keys}
    for i in range(trials):
        pi = R2CParams(**{**p.__dict__, "seed": p.seed + i})
        use_ist = ablation != "no_ist"
        use_awf = ablation != "no_awf"
        use_ise = ablation != "no_ise"
        ctrl = R2CController(pi, use_ist=use_ist, use_awf=use_awf)
        r = run_trial(pi, failure_ratio, deployment, ctrl, use_ise=use_ise)
        for k in keys:
            stats[k].append(r[k])
    out = {}
    for k in keys:
        a = np.array(stats[k])
        out[k] = (float(a.mean()), float(a.std()))
    out["CDD_example"] = r["CDD"]
    return out
