"""WP-B: RASHND-style RF-layer simulation (guide §5.1).

Python SINR model standing in for the ns-3 spectrum channel (guide lists a
SINR-grid model as an acceptable fallback). Provides:

- jammers: constant (fixed position), sweep (periodic traverse), reactive
  (chases the swarm centroid)
- background interferers: static Poisson point process emitters
- per-link SINR -> packet-delivery ratio q_ij in [0,1] (the quality map)
- quality-gated adjacency: edge iff d <= r_c AND q_ij >= theta
- throughput: sum over active links of B*log2(1+SINR)*PDR

Assumption: RASHND internals are not public; this approximates its
sensing -> link-quality-map function as described in the guide.
"""
from dataclasses import dataclass, field
import numpy as np

from .config import R2CParams


@dataclass
class RFParams:
    noise_dbm: float = -90.0        # thermal noise floor at receiver
    sinr_req_db: float = 10.0       # SINR needed for reliable delivery
    bandwidth_mhz: float = 20.0     # channel bandwidth for throughput
    theta: float = 0.5              # quality-map admission threshold
    n_interferers: int = 5          # background interferers
    interferer_power_dbm: float = 10.0
    jammer_power_dbm: float = 30.0
    jammer_mode: str = "constant"   # constant | sweep | reactive
    jammer_speed: float = 2.0       # m/step (reactive)
    sweep_period: int = 200         # steps per sweep traverse
    seed: int = 0


def _dbm_to_mw(dbm):
    return 10.0 ** (dbm / 10.0)


def _mw_to_dbm(mw):
    return 10.0 * np.log10(np.maximum(mw, 1e-30))


class RFLayer:
    """Spectrum environment producing per-link SINR and quality maps."""

    def __init__(self, p: R2CParams, rf: RFParams):
        self.p = p
        self.rf = rf
        rng = np.random.default_rng(rf.seed)
        self.interferers = rng.uniform(0, p.area, size=(rf.n_interferers, 2))
        self.jammer_pos = np.array([p.area / 2.0, p.area / 2.0])
        # per-step memo: rx_power/interferer noise and distances depend only
        # on (pos, active); the jammer term is recomputed on every call so
        # reactive-jammer move semantics are preserved exactly.
        self._static_key = None
        self._static = None
        self._dist_key = None
        self._dist = None

    def _distances(self, pos: np.ndarray) -> np.ndarray:
        key = pos.tobytes()
        if self._dist_key != key:
            self._dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
            self._dist_key = key
        return self._dist

    def _static_parts(self, pos: np.ndarray, active: np.ndarray):
        """rx_power[i, j] (j -> i) and interferer noise, per (pos, active)."""
        key = (pos.tobytes(), active.tobytes())
        if self._static_key != key:
            p = self.p
            n = len(pos)
            rx_power = np.zeros((n, n))
            for j in np.where(active)[0]:
                rx_power[:, j] = self._rx_mw(pos[j], pos, p.p_tx_dbm)
            noise = _dbm_to_mw(self.rf.noise_dbm)
            for itx in self.interferers:
                noise = noise + self._rx_mw(itx, pos,
                                            self.rf.interferer_power_dbm)
            self._static = (rx_power, noise)
            self._static_key = key
        return self._static

    def _rx_mw(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
               tx_power_dbm: float) -> np.ndarray:
        """Received power (mW) under the free-space path-loss model."""
        p = self.p
        dist = np.maximum(np.linalg.norm(rx_pos - tx_pos, axis=-1), 1e-9)
        dbm = tx_power_dbm - p.pl0_db - 10.0 * p.gamma * np.log10(dist / p.d0)
        return _dbm_to_mw(dbm)

    def _jammer_position(self, t: int, swarm_pos: np.ndarray,
                         active: np.ndarray) -> np.ndarray:
        rf, p = self.rf, self.p
        if rf.jammer_mode == "constant":
            return self.jammer_pos
        if rf.jammer_mode == "sweep":
            x = p.area * (0.5 + 0.4 * np.sin(2.0 * np.pi * t / rf.sweep_period))
            return np.array([x, p.area / 2.0])
        if rf.jammer_mode == "reactive":
            tgt = swarm_pos[active].mean(axis=0) if active.any() else self.jammer_pos
            d = tgt - self.jammer_pos
            n = np.linalg.norm(d)
            if n > 1e-9:
                self.jammer_pos += d / n * min(rf.jammer_speed, n)
            return self.jammer_pos
        raise ValueError(rf.jammer_mode)

    def _jammer_position_peek(self, t: int, swarm_pos: np.ndarray,
                              active: np.ndarray) -> np.ndarray:
        """Jammer position as observable this step WITHOUT consuming a move
        (non-mutating; safe to call any number of times per step)."""
        rf, p = self.rf, self.p
        if rf.jammer_mode == "constant":
            return self.jammer_pos
        if rf.jammer_mode == "sweep":
            x = p.area * (0.5 + 0.4 * np.sin(2.0 * np.pi * t / rf.sweep_period))
            return np.array([x, p.area / 2.0])
        if rf.jammer_mode == "reactive":
            tgt = swarm_pos[active].mean(axis=0) if active.any() else self.jammer_pos
            d = tgt - self.jammer_pos
            n = np.linalg.norm(d)
            if n > 1e-9:
                return self.jammer_pos + d / n * min(rf.jammer_speed, n)
            return self.jammer_pos
        raise ValueError(rf.jammer_mode)

    def sensed_jammer_power(self, pos: np.ndarray, active: np.ndarray,
                            t: int) -> np.ndarray:
        """Per-node received jammer power [mW] at the observable jammer
        position — the raw local sensing sample each node contributes to its
        cluster's shared belief (non-mutating)."""
        jp = self._jammer_position_peek(t, pos, active)
        return self._rx_mw(jp, pos, self.rf.jammer_power_dbm)

    def sinr_db(self, pos: np.ndarray, active: np.ndarray, t: int) -> np.ndarray:
        """Directed SINR (n,n): SINR at receiver i for transmitter j."""
        rx_power, noise = self._static_parts(pos, active)
        jpos = self._jammer_position(t, pos, active)
        noise = noise + self._rx_mw(jpos, pos, self.rf.jammer_power_dbm)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinr = rx_power / noise[:, None]
        sinr_db = _mw_to_dbm(sinr)
        sinr_db[~active, :] = -np.inf
        sinr_db[:, ~active] = -np.inf
        np.fill_diagonal(sinr_db, -np.inf)
        return sinr_db

    def quality_map(self, sinr_db: np.ndarray) -> np.ndarray:
        """q_ij in [0,1]: packet-delivery ratio from SINR (sigmoid model)."""
        return 1.0 / (1.0 + np.exp(-(sinr_db - self.rf.sinr_req_db) / 2.0))

    def gated_adjacency(self, pos: np.ndarray, active: np.ndarray,
                        t: int, r_c: float | None = None) -> np.ndarray:
        """Adjacency gated by link quality (guide §5.1 interface 1)."""
        p = self.p
        r_c = r_c or p.r_c
        dist = self._distances(pos)
        adj = ((dist <= r_c) & active[:, None] & active[None, :]).astype(float)
        q = self.quality_map(self.sinr_db(pos, active, t))
        adj[q < self.rf.theta] = 0.0
        np.fill_diagonal(adj, 0.0)
        return adj

    def quality_weighted_adjacency(self, pos: np.ndarray, active: np.ndarray,
                                   t: int, r_c: float | None = None) -> np.ndarray:
        """Distance-weighted adjacency (1 - d/r_c) scaled by q_ij
        (guide §5.1 interface 2: q_ij as GCN edge feature)."""
        p = self.p
        r_c = r_c or p.r_c
        dist = self._distances(pos)
        base = (dist <= r_c) & active[:, None] & active[None, :]
        w = np.maximum(0.0, 1.0 - dist / r_c) * base
        q = self.quality_map(self.sinr_db(pos, active, t))
        return w * q

    def throughput(self, pos: np.ndarray, active: np.ndarray, t: int) -> float:
        """Aggregate achieved throughput [Mbps] over admitted links."""
        p, rf = self.p, self.rf
        adj = self.gated_adjacency(pos, active, t)
        sinr = self.sinr_db(pos, active, t)
        sinr_lin = 10.0 ** (sinr / 10.0)
        q = self.quality_map(sinr)
        return float((rf.bandwidth_mhz *
                      np.log2(1.0 + sinr_lin) * q * adj).sum())

    def mean_quality(self, pos: np.ndarray, active: np.ndarray, t: int) -> float:
        adj_mask = self.gated_adjacency(pos, active, t) > 0
        if not adj_mask.any():
            return 0.0
        q = self.quality_map(self.sinr_db(pos, active, t))
        return float(q[adj_mask].mean())
