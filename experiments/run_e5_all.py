"""E5 driver: run all jammer x controller combos (chunked trial loop).

Thin wrapper around run_e5.cmd_run so the Kaggle runner can launch the
whole E5 job as a single command (run_e5.py itself only accepts one
mode/controller per invocation via its subcommand CLI).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import run_e5

MODES = ["constant", "sweep", "reactive"]
CONTROLLERS = ["r2c", "hybrid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3,
                    help="trials per mode/controller combo")
    ap.add_argument("--failures", type=float, default=0.5)
    ap.add_argument("--out", default="results/e5_convergence.csv")
    args = ap.parse_args()

    ns = argparse.Namespace(failures=args.failures, out=args.out,
                            mode=None, controller=None, trial=0,
                            ntrials=args.trials)
    for mode in MODES:
        for controller in CONTROLLERS:
            ns.mode, ns.controller = mode, controller
            print(f"=== E5 {mode}/{controller} ===", flush=True)
            run_e5.cmd_run(ns)
    run_e5.cmd_summarize(ns)
    print("E5 done.", flush=True)


if __name__ == "__main__":
    main()
