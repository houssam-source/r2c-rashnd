"""R2C controller: fuses ISE and IST through the adaptive sigmoid (AWF).

Paper Eqs. 22-23 (verified against the AAAI-26 PDF):

  w_T(t) = 1/(1 + exp(-k(t/T - s0)))   w_E(t) = 1 - w_T(t)     (time-aware)
  v_hat  = v_max (w_E v' + w_T v'') / (||w_E v' + w_T v''|| + eps)

v' is the ISE velocity (Eq. 10), v'' the IST translation; both enter as
directions and the fused vector is renormalized to v_max. IST fires every
T_I steps; its last translation is reused in between.
"""
import numpy as np

from .config import R2CParams
from .ise import ISE
from .ist import IST


class R2CController:
    def __init__(self, p: R2CParams, use_ist: bool = True, use_awf: bool = True):
        self.p = p
        self.use_ist = use_ist
        self.use_awf = use_awf
        self.ise = ISE(p)
        self.ist = IST(p) if use_ist else None
        self.last_ist = None  # last IST translation (reused between calls)
        self.w_history = []

    def awf_weight(self, t: int) -> float:
        """w_T(t): paper Eq. 22. With AWF disabled, fixed 0.5."""
        p = self.p
        if not self.use_awf:
            return 0.5
        return float(1.0 / (1.0 + np.exp(-p.awf_k * (t / p.T - p.awf_s0))))

    def velocity(self, env, t: int) -> np.ndarray:
        p = self.p
        d_ise = self.ise.direction(env.pos, env.active)
        d_ist = np.zeros_like(d_ise)
        if self.ist is not None:
            if t % p.T_ist == 0:
                self.ist.train_step(env.pos, env.active)
                self.last_ist = self.ist.translation(env.pos, env.active)
            if self.last_ist is not None:
                d_ist = self.last_ist
        w_t = self.awf_weight(t)
        self.w_history.append(w_t)
        fused = (1.0 - w_t) * d_ise + w_t * d_ist
        speed = np.linalg.norm(fused, axis=1, keepdims=True)
        v = p.v_max * fused / (speed + 1e-12)
        v[~env.active] = 0.0
        return v
