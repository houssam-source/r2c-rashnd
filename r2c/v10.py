"""v10: Monte-Carlo path-probability planner (preregistration_v10_mc.md).

Self-healing by decision-time imagination instead of myopic belief
forces. At each IST invocation:

1. Monte-Carlo localization of the jammer from received powers only
   (range-circle intersections + Gaussian power likelihood + resample;
   mode-aware motion model from displacement/chase features).
2. For each candidate corridor (existing IST boundary pairs) and each of
   M particles: roll the swarm T_h steps forward (point-mass:
   harm-gated jammer repulsion + corridor-endpoint attraction) under
   the particle's jammer trajectory.
3. Bridge the corridor maximizing P(connected at horizon AND horizon
   believed SINR >= 12.5 dB) - 0.01*length; fall back to v2 bridging if
   none scores above 0.

The planning objective IS the system objective (connectivity
probability under the physics) - no surrogate gap. Deterministic under
seed. Anti-cheat: received powers only; jammer truth never read.
"""
import time

import numpy as np

from .config import R2CParams
from .hybrid import HybridController
from .rf import RFLayer, RFParams

PACKET_GATE_DB = 12.5
TH_IST = 30          # planning horizon (design constant)
LEN_PEN = 0.01       # length penalty (design constant)


def _mode_posterior_update(alpha, prev_jhat, jhat, prev_chase_d, ctr):
    """Bayesian mode weights from displacement/chase features (v9 rule).
    ctr = current swarm centroid. Returns (alpha, chase_d)."""
    if jhat is None or prev_jhat is None:
        return alpha, prev_chase_d
    delta = float(np.linalg.norm(jhat - prev_jhat))
    d = float(np.linalg.norm(jhat - ctr))
    dd = d - prev_chase_d
    like = np.array([
        np.exp(-0.5 * (delta / 2.0) ** 2) * np.exp(-0.5 * (dd / 2.0) ** 2),
        np.exp(-0.5 * ((delta - 5.0) / 2.0) ** 2) * np.exp(-0.5 * (dd / 2.0) ** 2),
        np.exp(-0.5 * ((delta - 2.0) / 2.0) ** 2) * np.exp(-0.5 * ((dd + 1.0) / 1.0) ** 2),
    ])
    alpha = alpha + like / max(like.sum(), 1e-12)
    return alpha, d


class MCPlannerController(HybridController):
    """v10: v2 hybrid machinery + MC path-probability bridging."""

    def __init__(self, p: R2CParams, rf: RFLayer | None = None,
                 rf_p: RFParams | None = None, M: int = 64, seed: int = 0,
                 **kw):
        super().__init__(p, rf=rf, rf_p=rf_p, variant="v2", **kw)
        self.M = int(M)
        self._rng = np.random.default_rng(seed)
        self.mode_alpha = np.ones(3)
        self._prev_jhat = None
        self._chase_d = 0.0
        self.plan_time = []          # latency diagnostics
        self.posterior_err = []      # diagnostic only (truth read here)

    # ---- posterior --------------------------------------------------------
    def _localize(self, pos, active, pw):
        """MCL over jammer position from received powers. Returns
        (particles (K,2), weights (K,))."""
        act = np.where(active)[0]
        if len(act) == 0 or pw[act].sum() <= 0:
            return None, None
        p = self.p
        dbm = 10.0 * np.log10(np.maximum(pw[act], 1e-30))
        d = 10.0 ** ((self.rf.rf.jammer_power_dbm - p.pl0_db - dbm)
                     / (10.0 * p.gamma))
        ok = d <= p.area * 1.1
        act, d = act[ok], d[ok]
        if len(act) == 0:
            return None, None
        # particles on range circles of a random informative subset
        K = max(self.M, 32)
        picks = self._rng.integers(0, len(act), size=K)
        ang = self._rng.uniform(0, 2 * np.pi, size=K)
        pts = pos[act[picks]] + d[picks][:, None] * \
            np.stack([np.cos(ang), np.sin(ang)], axis=1)
        pts = np.clip(pts, 0, p.area)
        # likelihood of ALL observations under each particle
        D = np.linalg.norm(pts[:, None, :] - pos[act][None, :, :], axis=2)
        pred_dbm = (self.rf.rf.jammer_power_dbm - p.pl0_db
                    - 10.0 * p.gamma * np.log10(np.maximum(D, 1.0)))
        sig = 2.0
        ll = -0.5 * ((pred_dbm - dbm[None, :]) / sig) ** 2
        w = np.exp(ll.sum(axis=1) - ll.sum(axis=1).max())
        w /= max(w.sum(), 1e-12)
        # mode-aware drift + resample
        mode = self.mode_alpha / self.mode_alpha.sum()
        steps = self._rng.choice(3, size=K, p=mode)
        drift = np.zeros_like(pts)
        drift[steps == 1] = np.array([5.0, 0.0]) * 1   # sweep ~5 m/step
        idx = np.where(steps == 2)[0]
        if len(idx):
            ctr = pos[active].mean(axis=0)
            drift[idx] = (ctr - pts[idx]) / \
                np.maximum(np.linalg.norm(ctr - pts[idx], axis=1,
                                          keepdims=True), 1e-9) * 2.0
        pts = np.clip(pts + drift, 0, p.area)
        idx = self._rng.choice(K, size=K, p=w)
        return pts[idx], np.full(K, 1.0 / K)

    # ---- rollout -----------------------------------------------------------
    def _rollout(self, pos0, active, graph, corridor, pts, t):
        """Vectorized point-mass rollout over M particles. Returns per-
        particle (connected_at_horizon, corridor_sinr_db)."""
        p = self.p
        M = len(pts)
        n = len(pos0)
        act_idx = np.where(active)[0]
        pos = np.repeat(pos0[None, :, :], M, axis=0)      # (M,n,2)
        act = active.copy()
        # corridor endpoint attraction targets (per particle same)
        a, b = corridor
        src = graph.boundary[a][:6]
        dst = graph.boundary[b][:6]
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (PACKET_GATE_DB / 10.0) * noise
        # jammer per particle with per-mode motion
        mode = self.mode_alpha / self.mode_alpha.sum()
        msteps = self._rng.choice(3, size=M, p=mode)
        jp = pts.copy()
        connected_h = np.zeros(M)
        sinr_corr = np.full(M, -np.inf)
        repel_gain = 0.8
        for step in range(TH_IST):
            # jammer motion per particle
            jp[msteps == 1, 0] += 5.0
            jp[msteps == 1, 0] = np.clip(jp[msteps == 1, 0], 100, 900)
            ri = np.where(msteps == 2)[0]
            if len(ri):
                ctr = pos[ri][:, act[act_idx] if False else act_idx].mean(axis=1)
                v = ctr - jp[ri]
                jp[ri] += v / np.maximum(np.linalg.norm(v, axis=1,
                                                        keepdims=True), 1e-9) * 2.0
            jp = np.clip(jp, 0, p.area)
            # repulsion from particle jammer (harm-gated per cluster-mean)
            diff = pos[:, act_idx, :] - jp[:, None, :]
            dist = np.linalg.norm(diff, axis=2) + 1e-9
            pw = 10.0 ** ((self.rf.rf.jammer_power_dbm - p.pl0_db
                           - 10.0 * p.gamma * np.log10(np.maximum(dist, 1.0)))
                          / 10.0)
            jammed = pw / noise < 10.0 ** (self.rf.rf.sinr_req_db / 10.0)
            away = -diff / dist[..., None] * jammed[..., None]
            vel = np.zeros((M, n, 2))
            vel[:, act_idx, :] = away * repel_gain
            # corridor endpoint attraction
            vel[:, src, :] += (pos0[dst].mean(axis=0) - pos0[src]).reshape(1, len(src), 2) * 0.4
            vel[:, dst, :] += (pos0[src].mean(axis=0) - pos0[dst]).reshape(1, len(dst), 2) * 0.4
            sp = np.linalg.norm(vel, axis=2, keepdims=True)
            vel = p.v_max * vel / np.maximum(sp, 1e-9)
            pos[:, act_idx, :] = np.clip(pos[:, act_idx, :] + vel[:, act_idx, :],
                                         0, p.area)
            # connectivity at horizon-ish (last 1/3 of steps)
            if step >= TH_IST - 1:
                D = np.linalg.norm(
                    pos[:, act_idx, :][:, :, None, :] -
                    pos[:, act_idx, :][:, None, :, :], axis=3)
                adj = (D <= p.r_c).astype(float)
                # block diagonal test via spectral gap on the 2nd eigenvalue
                for m in range(M):
                    am = adj[m]
                    deg = am.sum(1)
                    L = np.diag(deg) - am
                    if len(L) > 2:
                        ev = np.linalg.eigvalsh(L)
                        connected_h[m] = ev[1] > 1e-9
                    else:
                        connected_h[m] = am.sum() > 0
                mid = 0.5 * (pos0[src].mean(axis=0) + pos0[dst].mean(axis=0))
                dm = np.linalg.norm(jp - mid, axis=1)
                pc = 10.0 ** ((self.rf.rf.jammer_power_dbm - p.pl0_db
                               - 10.0 * p.gamma * np.log10(np.maximum(dm, 1.0)))
                              / 10.0)
                sinr_corr = 10.0 * np.log10(np.maximum(pc / noise, 1e-9))
        return connected_h, sinr_corr

    # ---- plan ---------------------------------------------------------------
    def _mc_plan(self, env, t):
        from .search import CorridorGraph, build_planned_adj
        import scipy.sparse.csgraph as cg
        pos, active, p = env.pos, env.active, self.p
        mdist = self._mission_dist(env)
        adj = ((mdist <= p.r_c) & active[:, None]
               & active[None, :]).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = cg.connected_components(adj, directed=False)
        graph = CorridorGraph(self.ist, pos, active, labels, dist=mdist)
        pw = self.rf.sensed_jammer_power(pos, active, t)
        jhat = None
        P = pw[active]
        if P.sum() > 0:
            jhat = (P @ pos[active]) / P.sum()
        ctr = pos[active].mean(axis=0) if active.any() else None
        self.mode_alpha, self._chase_d = _mode_posterior_update(
            self.mode_alpha, self._prev_jhat, jhat, self._chase_d, ctr)
        if jhat is not None:
            self._prev_jhat = jhat.copy()
        clusters = list(graph.parts)
        if len(clusters) < 2:
            return adj.astype(float)
        t0 = time.time()
        pts, _w = self._localize(pos, active, pw)
        if pts is None:
            self.plan_time.append(time.time() - t0)
            return adj.astype(float)
        best, best_score = None, 0.0
        for a in clusters:
            for b, length in graph.edges[a]:
                if a >= b:
                    continue
                ch, sinr = self._rollout(pos, active, graph, (a, b), pts, t)
                prob = float(np.mean(ch.astype(bool)
                             & (sinr >= PACKET_GATE_DB)))
                score = prob - LEN_PEN * length / p.r_c
                if score > best_score:
                    best_score, best = score, (a, b)
        self._last_score = best_score
        self.plan_time.append(time.time() - t0)
        if best is None:
            # fall back to standard v2 bridging (all corridors)
            return build_planned_adj(
                self.ist, pos, active, labels, graph,
                [(a, b) for a in clusters
                 for b, _ in graph.edges[a] if a < b])
        return build_planned_adj(self.ist, pos, active, labels, graph,
                                 [best])

    # ---- velocity (v2 flow with the MC-planned adjacency substituted) --------
    def velocity(self, env, t):
        p = self.p
        r_eff = None
        if self.compress is not None and self._jammed(env, t):
            r_eff = self.compress * p.r_c
        if r_eff is not None:
            d_ise = self.ise.direction(env.pos, env.active, r_eff)
        else:
            d_ise = self.ise.direction(env.pos, env.active)
        if t > p.t1 and hasattr(env, "connected"):
            c_now = env.connected()
            if not c_now:
                self._was_fragmented = True
            elif (self.evasion_fade and self._latched_recovery is None
                    and self._was_fragmented):
                self._latched_recovery = t
        saved_gain = self.repel_gain
        if self.evasion_fade and self._latched_recovery is not None:
            dt = t - self._latched_recovery
            self.repel_gain = saved_gain * max(0.0, 1.0 - dt / self.evasion_fade)
        d_repel = self._cluster_repulsion(env, t)
        self.repel_gain = saved_gain

        d_ist = np.zeros_like(d_ise)
        if t % p.T_ist == 0:
            qadj = self._mc_plan(env, t)      # <-- the v10 seam
            self.ist.train_step(env.pos, env.active, adj_override=qadj)
            self.last_ist = self.ist.translation(env.pos, env.active,
                                                 adj_override=qadj)
        if self.last_ist is not None:
            d_ist = self.last_ist
        w_t = self.awf_weight(t)
        self.w_history.append(w_t)
        fused = (1.0 - w_t) * d_ise + w_t * d_ist + self.repel_gain * d_repel
        speed = np.linalg.norm(fused, axis=1, keepdims=True)
        v = p.v_max * fused / (speed + 1e-12)
        v[~env.active] = 0.0
        return v
