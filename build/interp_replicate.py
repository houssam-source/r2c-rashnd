"""Replication: weight-interpolation midpoint gain, guardrail scale.

3 init pairs x alphas {0, 0.4, 0.6, 1.0} x 20 paired seeds.
Pool(6). Writes interpolation_replication.json to hpo_out/agent_study/.
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PAIRS = [(0, 1), (2, 3), (4, 5)]
ALPHAS = [0.0, 0.4, 0.6, 1.0]
SEEDS = list(range(20))


def _init_state(seed):
    from r2c.config import R2CParams
    from r2c.hybrid import HybridController
    from r2c.rf import RFParams
    p = R2CParams(**{**R2CParams().__dict__, "seed": 0})
    torch.manual_seed(seed)
    c = HybridController(p, rf_p=RFParams(jammer_mode="constant", seed=0),
                         variant="v2")
    return {k: v.detach().clone().cpu() for k, v in
            c.ist.model.state_dict().items()}


def _blend(w0, w1, a):
    return {k: (1 - a) * w0[k].float() + a * w1[k].float() for k in w0}


def _episode(args):
    seed, w = args
    torch.set_num_threads(1)
    from r2c.config import R2CParams
    from r2c.environment import SwarmEnvironment
    from r2c.hybrid import HybridController
    from r2c.rf import RFLayer, RFParams
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    env = SwarmEnvironment(p, p.n_random, "random")
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    rf = RFLayer(p, rf_p)
    ctrl = HybridController(p, rf_p=rf_p, variant="v2")
    ctrl.ist.model.load_state_dict(w)
    t_rec, recovered = None, False
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(0.5)
        env.step(ctrl.velocity(env, t))
        if t > p.t1 and not recovered and env.connected() and \
                rf.mean_quality(env.pos, env.active, t) > 0.5:
            t_rec = t - p.t1
            recovered = True
    return float(recovered), ((t_rec / p.T) if recovered else 1.0)


def main():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    ws = {s: _init_state(s) for s in (0, 1, 2, 3, 4, 5)}
    results = {}
    for (a, b) in PAIRS:
        for al in ALPHAS:
            w = _blend(ws[a], ws[b], al)
            tasks = [(s, w) for s in SEEDS]
            with ctx.Pool(6) as pool:
                res = pool.map(_episode, tasks)
            rec = np.mean([r[0] for r in res])
            recs = [r[0] for r in res]
            rec_std = np.std(recs)
            rec_i = [int(r) for r in recs]
            # paired McNemar vs endpoint alpha=0 of the same pair
            results[f"{a}-{b}_a{al}"] = {
                "pair": [a, b], "alpha": al, "recovery": float(rec),
                "std": float(rec_std), "recovered_flags": rec_i}
            print(f"pair({a},{b}) alpha={al}: recovery={rec:.2f} "
                  f"+-{rec_std:.2f}", flush=True)
    # paired comparison midpoint vs endpoint per pair
    from scipy.stats import binomtest
    print("\npaired midpoint vs endpoint (McNemar, per pair):")
    mids = {}
    for (a, b) in PAIRS:
        ep = results[f"{a}-{b}_a0.0"]["recovered_flags"]
        for al in (0.4, 0.6):
            mid = results[f"{a}-{b}_a{al}"]["recovered_flags"]
            d10 = sum(1 for x, y in zip(mid, ep) if x == 1 and y == 0)
            d01 = sum(1 for x, y in zip(mid, ep) if x == 0 and y == 1)
            p = binomtest(max(d10, d01), d10 + d01, 0.5).pvalue \
                if d10 + d01 else 1.0
            print(f"  pair({a},{b}) a={al} vs a=0: mid-only {d10}, "
                  f"ep-only {d01}, p={p:.3f}")
            mids[(a, b, al)] = (d10, d01)
    out = {"pairs": PAIRS, "alphas": ALPHAS, "seeds": SEEDS,
           "results": results,
           "mids": {f"{k[0]}-{k[1]}_a{k[2]}": v for k, v in mids.items()}}
    path = os.path.join(REPO, "hpo_out", "agent_study",
                        "interpolation_replication.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("REPLICATION_DONE", flush=True)


if __name__ == "__main__":
    main()
