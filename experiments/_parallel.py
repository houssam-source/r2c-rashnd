"""Deterministic process-pool trial runner shared by the experiment drivers.

Each trial is independent and carries its own seed, so parallel execution
produces numerically identical results to the serial loop. When workers <= 1
the serial path is used (no process overhead, useful for single trials).
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def map_trials(worker, tasks, workers, on_result):
    """Run worker(task) per task; on_result(index, result) as each finishes.

    tasks[i] is the argument tuple for trial index i. Results stream back in
    completion order; the caller reassembles them by index, so downstream
    aggregation and CSV row order are identical to the serial loop.

    With workers > 1 each child process is limited to one BLAS/OpenMP thread:
    trial-level parallelism uses the cores, so per-trial threading would only
    oversubscribe (spawned children inherit this environment).
    """
    if workers <= 1:
        for i, task in enumerate(tasks):
            on_result(i, worker(task))
        return
    prev = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        n_proc = max(1, min(workers, os.cpu_count() or 1, len(tasks)))
        with ProcessPoolExecutor(max_workers=n_proc) as ex:
            futs = {ex.submit(worker, task): i for i, task in enumerate(tasks)}
            for fut in as_completed(futs):
                on_result(futs[fut], fut.result())
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev
