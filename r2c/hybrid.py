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
                 compress: float = 0.6,
                 variant: str = "v2",
                 gamma: float = 0.5, conf_min: float = 0.3,
                 age_max: int = 100, p_block: float = 1.0):
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
        # --- v3: search-guided belief dissemination -------------------------
        # v2: per-cluster divided sensing, standard bridging (baseline)
        # v3a: A* bridging order + belief-penalized costs, v2 flooding beliefs
        # v3b: v2 bridging order, beliefs carried by the search
        # v3:  both (spec §6 ablation matrix)
        assert variant in ("v2", "v3a", "v3b", "v3", "v4", "v5")
        self.variant = variant
        self.gamma = gamma
        self.p_block = p_block
        self.counters = {"sensing_events": 0, "replans": 0}
        self._last_path = None
        if variant in ("v3", "v3b", "v3a", "v4", "v5"):
            from .search import BeliefStore
            self.beliefs = BeliefStore(contrast_ref, conf_min, age_max,
                                       self.counters)
        else:
            self.beliefs = None

    # ---- v4: mission distances on the shared camera-construct --------------
    def _mission_dist(self, env):
        """Distance matrix used for clustering/bridging. v4 masks links that
        the terrain blocks (the camera-construct visibility graph); others
        use geometric range only."""
        d = env.distances()
        if self.variant == "v4":
            m = self.rf._vis_mask(env.pos)
            if m is not None:
                d = np.where(m, d, np.inf)
        return d

    def set_terrain(self, hmap, area: float, altitude: float):
        """Attach the shared camera-construct (terrain visibility map)."""
        self.rf.attach_terrain(hmap, area, altitude)

    # ---- v3 planning (runs at each IST invocation) -------------------------
    def _v3_plan(self, env, t: int):
        """A* over the cluster graph with belief-carrying expansion.

        Returns the IST adjacency override for this invocation and updates
        the belief store (origin sensing + on_open adoption + staleness).
        """
        from .search import (CorridorGraph, astar, belief_power_at,
                             build_planned_adj, norm_power)
        pos, active, p = env.pos, env.active, self.p
        mdist = self._mission_dist(env)
        adj = ((mdist <= p.r_c) & active[:, None]
               & active[None, :]).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = scipy.sparse.csgraph.connected_components(adj,
                                                              directed=False)
        graph = CorridorGraph(self.ist, pos, active, labels, dist=mdist)
        self.beliefs.age_step()
        self.beliefs.demote_stale()
        pw = self.rf.sensed_jammer_power(pos, active, t)

        clusters = list(graph.parts)
        if len(clusters) < 2:
            # nothing to bridge: geometric intra-part graph only
            qadj = adj.astype(float)
            return qadj

        # --- sensing origin: highest jammed-SINR exposure --------------------
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (self.rf.rf.sinr_req_db / 10.0) * noise

        def exposure(k):
            m = pw[graph.parts[k]].mean()
            return m / gate

        origin = max(clusters, key=exposure)
        v3_beliefs = self.variant in ("v3", "v3b", "v4")
        if v3_beliefs:
            self.beliefs.measure(self.beliefs.key(graph.parts[origin]), pos,
                                 pw, graph.parts[origin])

        # --- on_open adoption callback (v3/v3b) ------------------------------
        def on_open(k, path):
            if not v3_beliefs:
                return
            key = self.beliefs.key(graph.parts[k])
            if k == origin:
                return
            # propagated belief: power-weighted centroid of accumulated path
            # samples; confidence = min contrast along path; age = path hops
            vecs, confs = [], []
            for node in path:
                b = self.beliefs.get(self.beliefs.key(graph.parts[node]))
                if b is not None and b["vec"] is not None:
                    vecs.append(b["vec"])
                    confs.append(b["conf"])
            if vecs:
                conf = min(confs) if confs else 0.0
                age = len(path)
                # weight by confidence; fall back to origin belief
                w = np.array(confs) + 1e-9
                vec = np.average(np.vstack(vecs), axis=0, weights=w)
                self.beliefs.adopt(key, vec, conf, age)
                if self.beliefs.get(key)["conf"] == 0.0:
                    # safety valve: low propagated confidence -> local re-sense
                    self.beliefs.measure(key, pos, pw, graph.parts[k])
            else:
                self.beliefs.measure(key, pos, pw, graph.parts[k])

        # --- costs ------------------------------------------------------------
        def corridor_power(k, m):
            """Believed jammer power at the corridor midpoint."""
            if v3_beliefs:
                b = self.beliefs.get(self.beliefs.key(graph.parts[k]))
                if b is None or b["vec"] is None:
                    return 0.0
                mid = 0.5 * (graph.centroid[k] + graph.centroid[m])
                return belief_power_at(self.rf, b["vec"], mid)
            # v3a: beliefs flood locally per cluster (v2 style) — use the
            # sensing origin's belief for corridor penalization
            b = self.beliefs.get(self.beliefs.key(graph.parts[origin]))
            if b is None or b["vec"] is None:
                return 0.0
            mid = 0.5 * (graph.centroid[k] + graph.centroid[m])
            return belief_power_at(self.rf, b["vec"], mid)

        def g_cost(k, m, length):
            pen = 1.0 + self.gamma * norm_power(self.rf, corridor_power(k, m))
            return length / p.v_max * pen

        def h(k):
            if graph.largest is None:
                return 0.0
            return float(np.linalg.norm(graph.centroid[k]
                                        - graph.centroid[graph.largest])
                         / p.v_max)

        if self.variant == "v5":
            # phase 1: A* expansion disseminates beliefs (fills the store)
            _e1, closed = astar(origin, graph, g_cost, h, on_open=on_open)
            # phase 2: DP re-plan under staleness-growing coefficients
            from .dp import dp_plan
            age_map = {}
            for k in graph.parts:
                bb = self.beliefs.get(self.beliefs.key(graph.parts[k]))
                age_map[k] = bb["age"] if bb is not None else 0
            edges, _V = dp_plan(origin, graph, corridor_power,
                                lambda p: norm_power(self.rf, p),
                                p.v_max, gamma=self.gamma,
                                kappa_age=getattr(self, "planner_kappa", 1.0),
                                age_of=lambda k: age_map.get(k, 0))
        else:
            edges, closed = astar(origin, graph, g_cost, h, on_open=on_open)
        # re-plan frequency metric
        sig = tuple(edges) if edges else None
        if sig != self._last_path:
            self.counters["replans"] += 1
            self._last_path = sig

        if self.variant in ("v2", "v3b") or edges is None:
            # standard bridging: all corridors at once
            return build_planned_adj(self.ist, pos, active, labels, graph,
                                     [(a, b) for a in clusters
                                      for b, _ in graph.edges[a] if a < b])
        # v3a/v3: bridge the A* path's passable corridors this invocation
        # (closed order; blocked corridors skipped via re-plan next call)
        passable = [(a, b) for a, b in edges
                    if norm_power(self.rf, corridor_power(a, b)) < self.p_block]
        if passable:
            return build_planned_adj(self.ist, pos, active, labels,
                                     graph, passable)
        return adj.astype(float)   # all corridors believed blocked

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
        mdist = self._mission_dist(env)
        adj = ((mdist <= p.r_c) & active[:, None]
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
            # belief: v3 variants use the store (adopted beliefs, same
            # vintage across clusters); v2/v3a measure locally per cluster
            if self.beliefs is not None:
                b = self.beliefs.get(self.beliefs.key(idx))
                if b is None or b["vec"] is None or b["conf"] <= 0:
                    b = self.beliefs.measure(self.beliefs.key(idx), pos, pw,
                                             idx)
                if b["vec"] is None:
                    continue
                belief, conf = b["vec"], b["conf"]
            else:
                belief = (P @ pos[idx]) / Psum
                contrast = (P.max() - P.min()) / (P.mean() + 1e-12)
                conf = float(np.clip(contrast / self.contrast_ref, 0.0, 1.0))
                if hasattr(self, "counters"):
                    self.counters["sensing_events"] += 1   # v2 local sensing
            ctr = pos[idx].mean(axis=0)
            away = ctr - belief
            n = np.linalg.norm(away)
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
        if t % p.T_ist == 0:
            if self.variant == "v2":
                if self.use_geo_ist:
                    dist = env.distances()
                    qadj = ((dist <= p.r_c) & env.active[:, None]
                            & env.active[None, :]).astype(float)
                    np.fill_diagonal(qadj, 0.0)
                else:
                    qadj = self.rf.quality_weighted_adjacency(
                        env.pos, env.active, t)
            else:
                qadj = self._v3_plan(env, t)
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
