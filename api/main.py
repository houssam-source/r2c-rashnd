"""FastAPI inference endpoint for the R2C model state (guide §3 stack).

Serves the trained IST/Chebyshev-GCN controller:

- GET  /health          liveness
- GET  /model           architecture, parameter stats, optimizer, AWF history
- POST /predict         translation vectors + connectivity for a swarm state
- POST /env/reset       (optional) spawn a demo simulation environment
- POST /env/step        advance the demo env, watching the model train/act

Run:  uvicorn api.main:app --reload --port 8000
"""
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from r2c.config import R2CParams
from r2c.controller import R2CController
from r2c.environment import SwarmEnvironment
from r2c.metrics import algebraic_connectivity

app = FastAPI(title="R2C Model State API", version="0.1.0")


def _adjacency(pos: np.ndarray, active: np.ndarray, r_c: float) -> np.ndarray:
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    adj = ((dist <= r_c) & active[:, None] & active[None, :]).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj


class ServiceState:
    def __init__(self):
        self.p = R2CParams()
        self.controller = R2CController(self.p)
        self.train_invocations = 0
        self.last_loss = None
        self.env = None

    def param_stats(self):
        rows = []
        for name, t in self.controller.ist.model.named_parameters():
            g = t.grad
            rows.append({
                "name": name,
                "shape": list(t.shape),
                "numel": t.numel(),
                "norm": float(t.norm()),
                "mean": float(t.mean()),
                "std": float(t.std()),
                "grad_norm": float(g.norm()) if g is not None else None,
            })
        return rows


state = ServiceState()


class PredictRequest(BaseModel):
    positions: list[list[float]] = Field(..., description="n x 2 coordinates [m]")
    active: list[bool] | None = Field(None, description="defaults to all active")
    train: bool = Field(False, description="run a training step on this state first")


class ResetRequest(BaseModel):
    deployment: str = "random"
    failure_ratio: float = 0.0
    failure_time: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "t": state.env.t if state.env else None}


@app.get("/model")
def model_info():
    m = state.controller.ist.model
    n_params = sum(t.numel() for t in m.parameters())
    return {
        "architecture": "8-layer Chebyshev GCN (d=3->512) + tanh MLP",
        "total_params": n_params,
        "cheb_order": state.p.cheb_order,
        "lr": state.p.lr,
        "optimizer": type(state.controller.ist.opt).__name__,
        "train_invocations": state.train_invocations,
        "last_loss": state.last_loss,
        "awf_history": state.controller.w_history[-20:],
        "parameters": state.param_stats(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    p = state.p
    pos = np.asarray(req.positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise HTTPException(400, "positions must be n x 2")
    if len(pos) < 2 or len(pos) > 1000:
        raise HTTPException(400, "n must be in [2, 1000]")
    if np.any(pos < 0) or np.any(pos > p.area):
        raise HTTPException(400, f"positions must lie in [0, {p.area}]^2")
    active = (np.ones(len(pos), dtype=bool) if req.active is None
              else np.asarray(req.active, dtype=bool))
    if len(active) != len(pos):
        raise HTTPException(400, "active length must match positions")

    adj_before = _adjacency(pos, active, p.r_c)
    loss = None
    if req.train:
        loss = state.controller.ist.train_step(pos, active)
        state.train_invocations += 1
        state.last_loss = loss

    translation = state.controller.ist.translation(pos, active)
    lam_before = algebraic_connectivity(adj_before)
    adj_after = _adjacency(pos + translation * p.T_ist, active, p.r_c)
    return {
        "translation": translation.tolist(),
        "lambda2_before": lam_before,
        "lambda2_after": algebraic_connectivity(adj_after),
        "loss": loss,
        "train_invocations": state.train_invocations,
    }


@app.post("/env/reset")
def env_reset(req: ResetRequest):
    p = state.p
    if req.deployment not in ("random", "grid"):
        raise HTTPException(400, "deployment must be random|grid")
    n = p.n_random if req.deployment == "random" else p.n_grid
    state.env = SwarmEnvironment(p, n, req.deployment)
    state.env.failure_ratio = req.failure_ratio
    state.env.failure_time = req.failure_time if req.failure_time is not None else p.t1
    state.controller = R2CController(p)
    return env_summary()


@app.post("/env/step")
def env_step(steps: int = 1):
    if state.env is None:
        raise HTTPException(409, "call /env/reset first")
    if steps < 1 or steps > p_max_steps():
        raise HTTPException(400, f"steps must be in [1, {p_max_steps()}]")
    loss = None
    for _ in range(steps):
        if state.env.t == state.env.failure_time:
            state.env.inject_failures(state.env.failure_ratio)
        v = state.controller.velocity(state.env, state.env.t)
        if state.env.t % state.p.T_ist == 0:
            state.train_invocations += 1
        state.env.step(v)
    adj = state.env.active_adjacency()
    return {
        **env_summary(),
        "mean_speed": float(np.linalg.norm(v[state.env.active], axis=1).mean()),
        "lambda2": algebraic_connectivity(adj),
        "train_invocations": state.train_invocations,
    }


def p_max_steps():
    return 1000


def env_summary():
    env = state.env
    return {
        "t": env.t,
        "active": int(env.active.sum()),
        "connected": env.connected(),
        "coverage": env.coverage(),
        "avg_degree": float(env.adjacency().sum(1).mean()),
        "awf_last": state.controller.w_history[-1] if state.controller.w_history else None,
    }
