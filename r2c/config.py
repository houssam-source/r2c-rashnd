"""Paper parameter set (Equipment guide §3)."""
from dataclasses import dataclass


@dataclass
class R2CParams:
    area: float = 1000.0          # side length [m], 1000x1000 m^2
    n_random: int = 200           # random deployment
    n_grid: int = 196             # uniform grid deployment (14x14)
    r_c: float = 120.0            # communication radius [m]
    r_cov: float = 120.0          # coverage radius [m]
    v_max: float = 1.0            # max speed [m/step]
    T: int = 400                  # simulation steps
    t1: int = 100                 # failure injection time
    d_lo: float = 0.8 * 120.0     # neutral zone lower bound (0.8 R_c)
    d_hi: float = 0.9 * 120.0     # neutral zone upper bound (0.9 R_c)
    decay: float = 0.01           # virtual-force decay coefficients
    T_ist: int = 50               # IST invocation interval
    awf_s0: float = 0.125         # AWF sigmoid center
    awf_k: float = 30.0           # AWF sigmoid steepness
    lr: float = 1e-4              # Adam lr for IST training
    seed: int = 0

    # Free-space path-loss link model (paper Eqs. 2-4).
    # Link exists iff received power >= P0. PL0 is calibrated so the
    # sensitivity-derived range equals r_c = 120 m:
    #   120 = d0 * 10^((P_tx - PL0 - P0) / (10*gamma))  ->  PL0 = 58.42 dB
    p_tx_dbm: float = 20.0
    pl0_db: float = 58.42         # path loss at d0, calibrated to r_c
    gamma: float = 2.0            # path-loss exponent (free space)
    d0: float = 1.0
    p0_dbm: float = -80.0         # receiver sensitivity

    # Connectivity tolerance for algebraic-connectivity check
    conn_eps: float = 1e-9

    # Chebyshev GCN
    gcn_layers: int = 8
    gcn_hidden: int = 512
    cheb_order: int = 3

    # IST virtual multipartite graph (Assumption: paper gives no construction)
    r_com_boost: float = 1.5      # R^_com = r_com_boost * r_c
    top_m: int = 3                # top-M virtual neighbors across subnetworks

    # Coverage grid resolution for COV metric
    cov_grid: int = 100
