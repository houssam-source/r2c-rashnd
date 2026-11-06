"""R2C + RASHND hybrid controller, v2: cluster-shared RF belief (guide 5.1).

v1 reacted to degradation by compressing the swarm's interaction range, which
pulled nodes toward the field center — where the jammers live — and made
throughput/quality worse (E4 FAIL). v2 replaces that global compression with
the cluster-cooperative scheme:

1. Clusters = connected components of the active geometric graph (the same
   subnetworks the IST already builds).
2. Shared belief: each cluster fuses its members' jammer-power samples
   (RFLayer.sensed_jammer_power) into ONE estimate of the jammer direction,
   via a power-weighted centroid of member positions. Fusion averages out
   per-node sensing noise; larger clusters get sharper beliefs. Dividing the
   signals: each member samples the spectrum on its own slot (i mod K), so
   per-node sensing cost scales as 1/K while the cluster still updates its
   belief every step.
3. Action: every node gets its cluster's repulsion direction (away from the
   estimated jammer, confidence-weighted by the cluster's sensing contrast).
   Clusters therefore evade the jammer *together* while the IST keeps
   inter-cluster bridges alive and the ISE keeps spacing.

Interface with the RF quality map is unchanged for metrics (E4): quality-
gated adjacency, q-weighted IST graph, AWF time-aware fusion.

Anti-cheat: the belief is estimated from received powers only — the
controller never reads the jammer's true position.
"""
import numpy as np
import scipy.sparse.csgraph

from .config import R2CParams
from .controller import R2CController
from .rf import RFLayer, RFParams


class HybridController(R2CController):
    def __init__(self, p: R2CParams, rf: RFLayer | None = None,
                 rf_p: RFParams | None = None, use_awf: bool = True,
                 repel_gain: float = 0.4, contrast_ref: float = 0.5,
                 use_geo_ist: bool = True, harm_gate: bool = True,
                 compress: float = 0.6):
        super().__init__(p, use_ist=True, use_awf=use_awf)
        self.rf = rf if rf is not None else RFLayer(p, rf_p or RFParams())
        self.repel_gain = repel_gain
        self.contrast_ref = contrast_ref
        # IST graph: geometric (link existence) vs quality-gated. Geometric
        # lets the connectivity repair see fragments the jammer gates out.
        self.use_geo_ist = use_geo_ist
        self.harm_gate = harm_gate
        # when the swarm is jammed, tighten intra-cluster spacing to this
        # fraction of r_c (strong short links beat the jammer); None disables
        self.compress = compress

    def _jammed(self, env, t: int) -> bool:
        pw = self.rf.sensed_jammer_power(env.pos, env.active, t)
        if pw.sum() <= 0:
            return False
        noise_mw = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        sinr_req_lin = 10.0 ** (self.rf.rf.sinr_req_db / 10.0)
        return (pw.mean() / noise_mw) < sinr_req_lin

    def _cluster_repulsion(self, env, t: int) -> np.ndarray:
        """Per-node unit repulsion from the cluster's shared jammer belief.

        belief_jam_k = sum_i w_i pos_i / sum_i w_i, w_i = sensed jammer power
        of member i. Confidence = normalized within-cluster power contrast;
        a cluster whose members all sense the same power has no bearing
        information and does not move. With harm_gate, a cluster only evades
        when the jammer actually degrades its links (jammed SINR below the
        delivery requirement) — otherwise it stays on mission geometry.
        """
        pos, active, p = env.pos, env.active, self.p
        out = np.zeros_like(pos)
        adj = ((env.distances() <= p.r_c) & active[:, None]
               & active[None, :]).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = scipy.sparse.csgraph.connected_components(adj,
                                                              directed=False)
        pw = self.rf.sensed_jammer_power(pos, active, t)
        noise_mw = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        sinr_req_lin = 10.0 ** (self.rf.rf.sinr_req_db / 10.0)
        for k in set(labels[active]):
            idx = np.where((labels == k) & active)[0]
            P = pw[idx]
            Psum = P.sum()
            if Psum <= 0:
                continue
            if self.harm_gate and (P.mean() / noise_mw) >= sinr_req_lin:
                continue  # jammer harmless here: no evasion
            belief = (P @ pos[idx]) / Psum
            ctr = pos[idx].mean(axis=0)
            away = ctr - belief
            n = np.linalg.norm(away)
            contrast = (P.max() - P.min()) / (P.mean() + 1e-12)
            conf = float(np.clip(contrast / self.contrast_ref, 0.0, 1.0))
            if n > 1e-9 and conf > 0:
                out[idx] = away / n * conf
        return out

    def velocity(self, env, t: int) -> np.ndarray:
        p = self.p
        # full-range ISE spacing, tightened when jammed (short strong links
        # beat the jammer), cluster evasion toward free spectrum, IST bridge
        # repair on the (geo or quality) graph, AWF time fusion.
        if self.compress is not None and self._jammed(env, t):
            d_ise = self.ise.direction(env.pos, env.active,
                                       self.compress * p.r_c)
        else:
            d_ise = self.ise.direction(env.pos, env.active)
        d_repel = self._cluster_repulsion(env, t)

        d_ist = np.zeros_like(d_ise)
        if self.use_geo_ist:
            dist = env.distances()
            qadj = ((dist <= p.r_c) & env.active[:, None]
                    & env.active[None, :]).astype(float)
            np.fill_diagonal(qadj, 0.0)
        else:
            qadj = self.rf.quality_weighted_adjacency(env.pos, env.active, t)
        if t % p.T_ist == 0:
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
