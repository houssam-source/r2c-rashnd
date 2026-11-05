"""Kaggle runner for the R2C + RASHND experiment program.

Attach the project as a Kaggle dataset (r2c_project.zip -> dataset), then run
this as a script kernel with GPU enabled. Jobs run SEQUENTIALLY (Kaggle gives
~4 CPU cores; parallelism does not help this workload). Every job writes CSVs
incrementally to /kaggle/working, so a session that hits the time limit still
leaves all completed cells behind. Re-run with --start N to continue where a
previous session stopped (N = index of the next unfinished job below).

Usage on Kaggle:  python run_all_experiments.py [--start 0]

After completion: download /kaggle/working, place the CSVs into results/ on
this machine, then run  python experiments/analyze_results.py  for the
paper-comparison validation report and figures.
"""
import glob
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
JOBS = [
    # (description, script, args, out_csv)
    ("E3 CEN+HERO+SIDR", "experiments/run_baselines.py",
     ["--trials", "50", "--controllers", "CEN", "HERO", "SIDR"], "e3_cen_hero_sidr.csv"),
    ("E3 MDSG-APF+DEMD+MDSG-GC", "experiments/run_baselines.py",
     ["--trials", "50", "--controllers", "MDSG-APF", "DEMD", "MDSG-GC"], "e3_apf_demd_gc.csv"),
    ("E3 GCN", "experiments/run_baselines.py",
     ["--trials", "50", "--controllers", "GCN"], "e3_gcn.csv"),
    ("E3 CR-MGC", "experiments/run_baselines.py",
     ["--trials", "50", "--controllers", "CR-MGC"], "e3_cr_mgc.csv"),
    ("E1 random", "experiments/run_experiment.py",
     ["--trials", "50", "--deployments", "random", "--failures", "20", "40", "50", "60", "80"],
     "e1_random.csv"),
    ("E1 grid", "experiments/run_experiment.py",
     ["--trials", "50", "--deployments", "grid", "--failures", "20", "40", "50", "60", "80"],
     "e1_grid.csv"),
    ("E2 no_IST", "experiments/run_experiment.py",
     ["--trials", "50", "--ablation", "no_ist", "--failures", "20", "40", "50", "60", "80"],
     "e2_no_ist.csv"),
    ("E2 no_ISE", "experiments/run_experiment.py",
     ["--trials", "50", "--ablation", "no_ise", "--failures", "20", "40", "50", "60", "80"],
     "e2_no_ise.csv"),
    ("E2 no_AWF", "experiments/run_experiment.py",
     ["--trials", "50", "--ablation", "no_awf", "--failures", "20", "40", "50", "60", "80"],
     "e2_no_awf.csv"),
    ("E4 constant", "experiments/run_jamming.py",
     ["--modes", "constant", "--trials", "50"], "e4_constant.csv"),
    ("E4 sweep", "experiments/run_jamming.py",
     ["--modes", "sweep", "--trials", "50"], "e4_sweep.csv"),
    ("E4 reactive", "experiments/run_jamming.py",
     ["--modes", "reactive", "--trials", "50"], "e4_reactive.csv"),
    ("E5 loop stability", "experiments/run_e5.py",
     ["--trials", "3", "--failures", "0.5"], "e5_convergence.csv"),
]


def find_project():
    hits = []
    for base in ["/kaggle/input", os.getcwd(), os.path.dirname(os.getcwd())]:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            depth = root.count(os.sep)
            if depth > 8:
                dirs[:] = []
            if "__init__.py" in files and root.endswith("r2c"):
                hits.append(os.path.dirname(root))
    print("project candidates:", hits, flush=True)
    if hits:
        return hits[0]
    raise SystemExit("r2c package not found -- attach the r2c project dataset")


def main():
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    root = find_project()
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    print(f"project: {root}\noutput:  {OUT}\nstarting at job {start}", flush=True)
    for i, (desc, script, args, out_csv) in enumerate(JOBS):
        if i < start:
            continue
        cmd = [sys.executable, "-u", os.path.join(root, script)] + args + \
              ["--out", os.path.join(OUT, out_csv)]
        print(f"\n===== [{i}] {desc} =====", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=root, env=env)
        print(f"----- [{i}] exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("\nALL JOBS DONE -- download /kaggle/working and run "
          "experiments/analyze_results.py locally", flush=True)


if __name__ == "__main__":
    main()
