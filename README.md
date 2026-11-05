# R2C + RASHND Hybrid Self-Healing UAV Swarm

Reproduction of the R2C paper (Peng et al., AAAI-26: *Resilient UAV Swarm with
Fast Connectivity Recovery and Extensive Coverage*, DOI 10.1609/aaai.v40i2.37060)
extended with a RASHND-style spectrum-sensing layer (Hegde et al., arXiv:2601.13418).
No public code exists for either paper; everything here is re-implemented from the
papers, with assumptions marked in docstrings.

## Layout
- `r2c/` — simulator + algorithms: environment (path-loss links, failures),
  ISE (buffered virtual force, paper Eq. 9-10), IST (first-order Chebyshev GCN,
  multipartite graph, joint connectivity loss), AWF (time-aware fusion, Eq. 22-23),
  RF layer (jammers/SINR/quality maps), hybrid controller, 8 baselines, metrics
  (REC Eq. 5, COV Eq. 7, AND, CDD)
- `experiments/` — runners: E1 reproduction, E2 ablations, E3 baselines,
  E4 jamming (R2C vs hybrid), E5 loop stability, `analyze_results.py`
  (PASS/FAIL validation against the paper tables + figures)
- `kaggle/` — Kaggle kernel runner (`run_all_experiments.py`) that clones this
  repo and executes the full experiment matrix
- `api/` — FastAPI model-state endpoint (`uvicorn api.main:app`)

## Run
```
pip install -r requirements.txt
python experiments/run_experiment.py --trials 50 --out results/e1_random.csv
python experiments/analyze_results.py
```
