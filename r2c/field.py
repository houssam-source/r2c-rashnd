"""FieldController: decentralized, zero-delay, recovery-first swarm control.

Design goals (project spec 2026-11-08): decentralized (1-hop gossip only, no
global graph, no leader), zero delay (every decision from this step's local
state; no invocation cadence, no training loop, no planner), novel control
paradigm (coupled diffused fields), recovery probability first, coverage
second, survivability third; open field and urban (local LOS).

Mechanism (all quantities local to node i and its 1-hop neighborhood):

1. Connectivity field v (distributed Fiedler estimate): one mean-deflated
   power-iteration step per control step. Weak cuts appear as large |v_i-v_j|
   across i's links. Symmetry broken by a fixed per-node sign drawn from the
   node id (no seeding needed, deterministic).
2. Repair force: descend the local Rayleigh numerator
       d_i = sum_j w_ij (v_i - v_j) u_ij,  u_ij = (x_i - x_j)/|x_i - x_j|
   which pulls the two sides of a weak cut toward each other (lambda2 grows).
3. Jammer field p: 1-hop gossip of sensed jammer power; local gradient gives
   a flee direction, harm-gated on the cluster-mean power (no evasion when
   the jammer does not degrade local links).
4. Coverage (secondary): boundary nodes (local neighbor-direction cone has a
   gap) spread outward only when local cut-stress q_i is below a margin, so
   coverage rebuilds after repair, locally decided.
5. Urban: w_ij = 0 when the LOS between i and j is blocked at the mission
   altitude (each endpoint pair checks its own link; no shared map).

Messages per step per node: one (v, p) pair per neighbor. O(degree).
"""
import numpy as np

from .config import R2CParams
from .ise import ISE
from .rf import RFParams, RFLayer


class FieldController:
    """Fully decentralized field controller. velocity(env, t) -> (n, 2)."""

    def __init__(self, p: R2CParams, rf_p: RFParams | None = None,
                 beta: float = 0.35, w_repair: float = 1.0,
                 w_flee: float = 0.6, w_cover: float = 0.5,
                 cover_margin: float = 0.15, altitude: float = 30.0,
                 terrain: tuple | None = None, seed: int = 0,
                 eta: float = 0.15, hops: int = 2):
        self.p = p
        self.rf = RFLayer(p, rf_p or RFParams(seed=seed))
        self.ise = ISE(p)                       # base spacing potential
        self.deg_max = None                     # per-node degree running max
        self.beta = beta
        self.w_repair = w_repair
        self.w_flee = w_flee
        self.w_cover = w_cover
        self.cover_margin = cover_margin
        self.altitude = altitude
        # deterministic per-node symmetry-breaking signs from node ids
        rng = np.random.default_rng(1234567)
        self.signs = rng.choice([-1.0, 1.0], size=max(p.n_random, p.n_grid))
        # local LOS checker (terrain grid) shared per endpoint pair
        self._terrain = terrain          # (hmap, area)
        self._V = None
        if terrain is not None:
            from .visibility import los_cell_matrix
            self._V = los_cell_matrix(terrain[0], terrain[1], altitude)
        # field state
        self.v = None
        self.m = None          # consensus global mean of v (constant under
                               # symmetric diffusion, so gossip converges to it)
        self.eta = eta         # persistent injection of the symmetry-breaking
                               # mode -> sustained bottleneck field
        self.hops = hops       # gossip TTL: relayed multi-hop neighborhood
        self._rng = np.random.default_rng(seed)

    # ---- local adjacency with LOS -----------------------------------------
    def _weights(self, pos: np.ndarray, active: np.ndarray) -> np.ndarray:
        p = self.p
        n = len(pos)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        w = ((d <= p.r_c) & active[:, None] & active[None, :]).astype(float)
        if self._V is not None:
            from .visibility import vis_mask
            w *= vis_mask(pos, self._V, self._terrain[1])
        np.fill_diagonal(w, 0.0)
        return w

    # ---- field update ------------------------------------------------------
    def _step_field(self, w: np.ndarray) -> np.ndarray:
        """Diffusion + persistent injection + consensus-mean deflation, over
        the TTL-relayed gossip adjacency. Fully local message passing."""
        deg = w.sum(1)
        if self.v is None:
            self.v = self.signs[: len(w)].copy()
            self.m = self.v.copy()
        # TTL-2 relay: 2-hop neighborhood adjacency (local via relayed gossip)
        w2 = ((w + w @ w) > 0).astype(float)
        np.fill_diagonal(w2, 0.0)
        w2 *= w.shape[0] / max(w2.sum(), 1e-9)      # scale-neutral
        deg2 = w2.sum(1)
        lv = deg2 * self.v - w2 @ self.v            # (L v) on the gossip graph
        u = self.v - self.beta * lv + self.eta * self.signs[: len(w)]
        # consensus tracking of the (constant) global mean
        self.m = self.m + 0.3 * (np.where(deg > 0, w @ self.m, self.m)
                                 / np.maximum(deg, 1.0) - self.m)
        self.v = np.clip(u - self.m, -1.5, 1.5)
        return deg

    # ---- forces -------------------------------------------------------------
    def velocity(self, env, t: int) -> np.ndarray:
        p = self.p
        pos, active = env.pos, env.active
        n = len(pos)
        w = self._weights(pos, active)
        deg = self._step_field(w)

        v = np.zeros((n, 2))
        # TTL-relayed adjacency for forces (reaches fragments beyond r_c)
        w2 = ((w + w @ w) > 0).astype(float)
        np.fill_diagonal(w2, 0.0)
        # 1) repair: pull the two sides of a weak cut together
        diff = self.v[:, None] - self.v[None, :]
        dmat = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            uhat = (pos[:, None] - pos[None]) / dmat[..., None]
        rep = -np.nansum(w2[..., None] * diff[..., None] * uhat, axis=1)
        # 2) flee: descend the relayed jammer-power gradient (harm-gated)
        pw = self.rf.sensed_jammer_power(pos, active, t)
        pdiff = pw[:, None] - pw[None, :]
        fle = -np.nansum(w2[..., None] * pdiff[..., None] * uhat, axis=1)
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (self.rf.rf.sinr_req_db / 10.0) * noise
        jammed = (pw / noise) < gate
        fle[~jammed] = 0.0
        # failure gating + cut-calm, needed by coverage and search
        q = w * diff ** 2
        if self.deg_max is None:
            self.deg_max = deg.copy()
        self.deg_max = np.maximum(self.deg_max, deg)
        degraded = deg < 0.6 * self.deg_max
        calm = q.sum(1) < self.cover_margin * np.maximum(deg, 1.0)
        # 3) coverage: boundary gap spreading, only when cuts are calm                     # local cut-stress per link
        calm = q.sum(1) < self.cover_margin * np.maximum(deg, 1.0)
        for i in np.where(active)[0]:
            if deg[i] < 1 or not calm[i]:
                continue
            if not degraded[i]:
                continue
            nbrs = np.where(w[i] > 0)[0]
            ang = np.arctan2(pos[nbrs, 1] - pos[i, 1],
                             pos[nbrs, 0] - pos[i, 0])
            ang = np.sort(ang)
            gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
            g = gaps.max()
            if g > np.radians(130):           # boundary cone gap
                k = gaps.argmax()
                bisector = (ang[k] + g / 2.0) + np.pi
                v[i] += self.w_cover * np.array([np.cos(bisector),
                                                 np.sin(bisector)])
        # 4) failure-gated exploration: only degraded nodes search
        sparse = deg < 2
        sweep = np.zeros((n, 2))
        ids = np.arange(n)
        a0 = (ids * 2654435761 % 6283) / 1000.0          # per-node heading
        ang = a0 + t * 0.02                              # slow rotation
        sweep[:, 0] = np.cos(ang)
        sweep[:, 1] = np.sin(ang)
        search = np.where((sparse & calm & degraded & active)[:, None],
                          0.5 * sweep, 0.0)
        # fuse
        d_spring = self.ise.direction(pos, active)
        fused = (d_spring + self.w_repair * rep + self.w_flee * fle
                   + v + search)
        speed = np.linalg.norm(fused, axis=1, keepdims=True)
        vel = p.v_max * fused / (speed + 1e-12)
        vel[~active] = 0.0
        return vel
