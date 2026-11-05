"""E5: bidirectional-loop convergence/stability (guide E5).

For each jammer mode (constant / sweep / reactive) and controller
(r2c / hybrid): 3 trials of T=400 with 50% failure injection at t1=100.
Records mean link quality and throughput every 10 steps after t1, then
computes mean/std of each metric over the last 100 steps across trials as
a convergence proxy (low relative std = stable loop).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.hybrid import HybridController
from r2c.rf import RFLayer, RFParams

MODES = ["constant", "sweep", "reactive"]
CONTROLLERS = ["r2c", "hybrid"]


def run_e5_trial(p: R2CParams, rf_p: RFParams, failure_ratio: float,
                 controller: str, record_every: int = 10) -> list:
    """Mirror run_jamming.run_jamming_trial; returns trace rows
    (t, mean_quality, throughput_mbps) every `record_every` steps after t1."""
    env = SwarmEnvironment(p, p.n_random, "random")
    rf = RFLayer(p, rf_p)
    ctrl = HybridController(p, rf) if controller == "hybrid" \
        else R2CController(p)
    trace = []
    for t in range(p.T):
        if t == p.t1:
            env.inject_failures(failure_ratio)
        env.step(ctrl.velocity(env, t))
        if t > p.t1 and t % record_every == 0:
            trace.append((t,
                          rf.mean_quality(env.pos, env.active, t),
                          rf.throughput(env.pos, env.active, t)))
    return trace


def _read_trace(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["row_type"] == "trace":
                rows.append(r)
    return rows


def cmd_run(args):
    """Run a single trial and append its trace rows (chunked driver)."""
    p = R2CParams()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out) or \
        os.path.getsize(args.out) == 0
    pi = R2CParams()
    for trial in range(args.trial, args.trial + args.ntrials):
        p = R2CParams(**{**pi.__dict__, "seed": pi.seed + trial})
        ri = RFParams(jammer_mode=args.mode, seed=pi.seed + trial)
        t0 = time.time()
        trace = run_e5_trial(p, ri, args.failures, args.controller)
        last = [(q, thr) for t, q, thr in trace if t >= p.T - 100]
        print(f"[{args.mode} | {args.controller} | trial {trial}] "
              f"last100 q={np.mean([q for q, _ in last]):.4f} "
              f"thr={np.mean([thr for _, thr in last]):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        write_header = not os.path.exists(args.out) or \
            os.path.getsize(args.out) == 0
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["row_type", "jammer", "controller", "trial", "t",
                            "metric", "value", "std", "rel_std"])
            for t, q, thr in trace:
                w.writerow(["trace", args.mode, args.controller, trial, t,
                            "mean_quality", f"{q:.6f}", "", ""])
                w.writerow(["trace", args.mode, args.controller, trial, t,
                            "throughput_mbps", f"{thr:.6f}", "", ""])


def cmd_summarize(args):
    """Recompute summary rows from trace rows (cheap; no simulation)."""
    p = R2CParams()
    rows = _read_trace(args.out)
    combos = sorted({(r["jammer"], r["controller"]) for r in rows})
    summary = []
    for mode, controller in combos:
        for metric in ["mean_quality", "throughput_mbps"]:
            per_trial = {}
            for r in rows:
                if r["jammer"] == mode and r["controller"] == controller \
                        and r["metric"] == metric and int(r["t"]) >= p.T - 100:
                    per_trial.setdefault(int(r["trial"]), []).append(
                        float(r["value"]))
            win = [np.mean(v) for _, v in sorted(per_trial.items())]
            if not win:
                continue
            a = np.array(win)
            mean, std = a.mean(), a.std()
            rel = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
            summary.append(["summary", mode, controller, "all", "last100",
                            metric, f"{mean:.6f}", f"{std:.6f}", f"{rel:.6f}"])
            print(f"[{mode} | {controller}] {metric}: {mean:.4f} +- "
                  f"{std:.4f} (rel {rel:.4f}, n={len(win)} trials)", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "jammer", "controller", "trial", "t",
                    "metric", "value", "std", "rel_std"])
        for r in rows:
            w.writerow([r["row_type"], r["jammer"], r["controller"],
                        r["trial"], r["t"], r["metric"], r["value"],
                        r["std"], r["rel_std"]])
        w.writerows(summary)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("run", cmd_run), ("summarize", cmd_summarize)]:
        sp = sub.add_parser(name)
        sp.add_argument("--failures", type=float, default=0.5)
        sp.add_argument("--out", default="results/e5_convergence.csv")
        if name == "run":
            sp.add_argument("--mode", required=True, choices=MODES)
            sp.add_argument("--controller", required=True,
                            choices=CONTROLLERS)
            sp.add_argument("--trial", type=int, required=True)
            sp.add_argument("--ntrials", type=int, default=1)
        sp.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
