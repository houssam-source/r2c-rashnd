"""R2C + RASHND hybrid controller (guide §5.1).

Closes the loop between the RF-layer quality map and R2C:

1. adjacency: edge iff d <= r_c AND q_ij >= theta (quality-gated)
2. GCN edge features: q_ij-scaled distance weights feed the IST Laplacian
3. ISE: effective range R_c^eff = r_c * f(mean q) keeps nodes closer when
   the spectrum is degraded (f = clip(x, 0.3, 1), Assumption)

Fusion follows the paper's time-aware AWF (Eq. 22-23). The bidirectional
loop (E5): motion changes the geometry -> re-sensing at the next step
produces a new quality map -> new effective range/weights.
"""
import numpy as np

from .config import R2CParams
from .controller import R2CController
from .rf import RFLayer, RFParams


class HybridController(R2CController):
    def __init__(self, p: R2CParams, rf: RFLayer | None = None,
                 rf_p: RFParams | None = None, use_awf: bool = True):
        super().__init__(p, use_ist=True, use_awf=use_awf)
        self.rf = rf if rf is not None else RFLayer(p, rf_p or RFParams())

    def _mean_link_quality(self, env, t: int) -> float:
        geo = (env.distances() <= self.p.r_c) & env.active[:, None] & env.active[None, :]
        geo &= ~np.eye(len(env.pos), dtype=bool)
        if not geo.any():
            return 0.0
        q = self.rf.quality_map(self.rf.sinr_db(env.pos, env.active, t))
        return float(q[geo].mean())

    def velocity(self, env, t: int) -> np.ndarray:
        p = self.p
        # interface 3: quality-dependent effective interaction range
        mean_q = self._mean_link_quality(env, t)
        r_eff = p.r_c * float(np.clip(mean_q, 0.3, 1.0))
        d_ise = self.ise.direction(env.pos, env.active, r_eff)

        d_ist = np.zeros_like(d_ise)
        # interface 2: q_ij-scaled edge weights for the IST graph
        qadj = self.rf.quality_weighted_adjacency(env.pos, env.active, t)
        if t % p.T_ist == 0:
            self.ist.train_step(env.pos, env.active, adj_override=qadj)
            self.last_ist = self.ist.translation(env.pos, env.active,
                                                 adj_override=qadj)
        if self.last_ist is not None:
            d_ist = self.last_ist
        w_t = self.awf_weight(t)
        self.w_history.append(w_t)
        fused = (1.0 - w_t) * d_ise + w_t * d_ist
        speed = np.linalg.norm(fused, axis=1, keepdims=True)
        v = p.v_max * fused / (speed + 1e-12)
        v[~env.active] = 0.0
        return v
