"""ISE: buffered virtual-force module (reactive, runs every step).

Implements the paper's Eq. 9 exactly (verified against the AAAI-26 PDF,
see results/paper_reference.md §4):

  repulsive:  -exp(-alpha * d_ij/d_lo) * u_ij      for d_ij < d_lo
  neutral:    0                                   for d_lo <= d_ij <= d_hi
  attractive:  exp(-beta  * (Rc-d_ij)/(Rc-d_hi)) * u_ij  for d_hi < d_ij <= Rc
  boundary:   F'' = sum_{k in B} exp(-delta * d_i,k) * n_k

with alpha = beta = delta = 0.01 (paper decay coefficients) and u_ij the
unit vector from j to i. Node velocity: v' = v_max * F/(||F|| + eps) (Eq. 10).
"""
import numpy as np

from .config import R2CParams


class ISE:
    def __init__(self, p: R2CParams):
        self.p = p

    def force(self, pos: np.ndarray, active: np.ndarray,
              r_c: float | None = None) -> np.ndarray:
        p = self.p
        a = p.decay  # alpha = beta = delta = 0.01
        r_eff = r_c or p.r_c
        frac = 1.0 if r_c is None else r_c / p.r_c
        d_lo, d_hi = p.d_lo * frac, p.d_hi * frac

        diff = pos[:, None, :] - pos[None, :, :]  # i - j
        dist = np.linalg.norm(diff, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = diff / np.maximum(dist, 1e-12)[..., None]

        in_range = (dist > 1e-9) & (dist <= r_eff) & active[None, :]
        rep = np.where(dist < d_lo, np.exp(-a * dist / d_lo), 0.0)
        att = np.where((dist > d_hi) & (dist <= r_eff),
                       np.exp(-a * (r_eff - dist) / (r_eff - d_hi)), 0.0)
        mag = (rep - att) * in_range              # + away from j, - toward j
        F = (unit * mag[..., None]).sum(axis=1)

        # boundary repulsion (paper: F'' = sum_{k in B} exp(-delta d_i,k) n_k,
        # n_k the inward normal; active when a node is within the margin)
        margin = 0.5 * r_eff
        d_low = np.clip(margin - pos, 1e-9, None)          # to x=0 / y=0 edges
        d_high = np.clip(pos - (p.area - margin), 1e-9, None)  # to far edges
        F += np.where(margin - pos > 0, np.exp(-a * d_low), 0.0)
        F -= np.where(pos - (p.area - margin) > 0, np.exp(-a * d_high), 0.0)
        F[~active] = 0.0
        return F

    def velocity(self, pos: np.ndarray, active: np.ndarray,
                 r_c: float | None = None) -> np.ndarray:
        """Paper Eq. 10: v' = v_max * F/(||F|| + eps)."""
        F = self.force(pos, active, r_c)
        speed = np.linalg.norm(F, axis=1, keepdims=True)
        v = self.p.v_max * F / (speed + 1e-12)
        v[~active] = 0.0
        return v

    def direction(self, pos: np.ndarray, active: np.ndarray,
                  r_c: float | None = None) -> np.ndarray:
        """Unit direction of the ISE velocity (for AWF fusion)."""
        v = self.velocity(pos, active, r_c)
        v[~active] = 0.0
        return v / max(self.p.v_max, 1e-12)
