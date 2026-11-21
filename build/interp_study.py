"""Weight-space matrix interpolation study (self-healing path scan).

Question: does blending the GCN weight matrices between two inits
W(a) = (1-a) W0 + a W1 (same architecture 512x8, baseline training)
find a BETTER network than either endpoint?

Endpoints: torch.manual_seed(0) and (1) inits of the baseline ChebNet.
Both are evaluated by the SYSTEM metric (guardrail recovery, 10 paired
seeds) AND the surrogate (val loss) at each alpha. This tests whether
matrix interpolation traces an optimizing path or hits a loss barrier
(linear mode connectivity).
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from r2c.config import R2CParams          # noqa: E402
from r2c.environment import SwarmEnvironment  # noqa: E402
from r2c.hybrid import HybridController   # noqa: E402
from r2c.rf import RFLayer, RFParams      # noqa: E402

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = list(range(10))


def init_state(seed, p, rf_p):
    torch.manual_seed(seed)
    c = HybridController(p, rf_p=rf_p, variant="v2")
    return {k: v.detach().clone().cpu() for k, v in
            c.ist.model.state_dict().items()}


def blend(w0, w1, a):
    return {k: (1 - a) * w0[k].float() + a * w1[k].float()
            for k in w0}


def episode_with_weights(seed, w):
    p = R2CParams(**{**R2CParams().__dict__, "seed": seed})
    env = SwarmEnvironment(p, p.n_random, "random")
    rf_p = RFParams(jammer_mode="constant", seed=seed)
    rf = RFLayer(p, rf_p)
    ctrl = HybridController(p, rf_p=rf_p, variant="v2")
    ctrl.ist.model.load_state_dict(w)
    val_losses = []
    ist = ctrl.ist
    orig = ist.train_step
    calls = {"n": 0}

    def hooked(pos, active, epochs=5, adj_override=None):
        calls["n"] += 1
        if calls["n"] > 1:
            ist.model.eval()
            with torch.no_grad():
                pos_t = torch.tensor(pos, dtype=torch.float32,
                                     device=ist.device)
                act_t = torch.tensor(active, dtype=torch.float32,
                                     device=ist.device)
                labels, lnorm = ist._graph_parts(pos, active, adj_override)
                n_parts = len(set(labels[active]))
                if n_parts > 0:
                    d = ist.model(ist._features(pos, active).to(ist.device),
                                  lnorm.to(ist.device),
                                  torch.tensor(labels, device=ist.device),
                                  n_parts)
                    vl = ist._loss(
                        pos_t + d[torch.tensor(
                            [labels[i] if labels[i] < n_parts else 0
                             for i in range(len(pos))],
                            device=ist.device)] * p.v_max * p.T_ist,
                        d, labels, act_t, n_parts)
                    val_losses.append(float(vl.item()))
            ist.model.train()
        return orig(pos, active, epochs=epochs, adj_override=adj_override)
    ist.train_step = hooked

    t_rec, recovered = None, False
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(0.5)
        env.step(ctrl.velocity(env, t))
        if t > p.t1 and not recovered and env.connected() and \
                rf.mean_quality(env.pos, env.active, t) > 0.5:
            t_rec = t - p.t1
            recovered = True
    return {"recovered": float(recovered),
            "REC": (t_rec / p.T) if recovered else 1.0,
            "val_loss": float(np.mean(val_losses)) if val_losses
            else float("nan")}


def main():
    p0 = R2CParams(**{**R2CParams().__dict__, "seed": 0})
    w0 = init_state(0, p0, RFParams(jammer_mode="constant", seed=0))
    w1 = init_state(1, p0, RFParams(jammer_mode="constant", seed=0))
    out = []
    for a in ALPHAS:
        w = blend(w0, w1, a)
        recs, vls = [], []
        for s in SEEDS:
            r = episode_with_weights(s, w)
            recs.append(r["recovered"])
            vls.append(r["val_loss"])
        row = {"alpha": a, "recovery": float(np.mean(recs)),
               "recovery_std": float(np.std(recs)),
               "val_loss": float(np.nanmean(vls))}
        out.append(row)
        print(f"alpha={a:.1f} recovery={row['recovery']:.2f} "
              f"+-{row['recovery_std']:.2f} val={row['val_loss']:.3f}",
              flush=True)
    os.makedirs(os.path.join(REPO, "hpo_out", "agent_study"),
                exist_ok=True)
    path = os.path.join(REPO, "hpo_out", "agent_study",
                        "interpolation_curve.json")
    with open(path, "w") as f:
        json.dump({"alphas": ALPHAS, "seeds": SEEDS, "curve": out,
                   "endpoints": "ChebNet 512x8 inits seed0/seed1, "
                   "baseline training, guardrail recovery + val loss"},
                  f, indent=2)
    print("INTERP_DONE", flush=True)


if __name__ == "__main__":
    main()
