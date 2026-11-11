"""v9: information-gain controller on a shared belief state (spec 2026-11-11).

Preregistered as results/preregistration_v9_ig.md (locked before any run).

v9 composes the existing v3 hybrid (cluster beliefs + A* belief
dissemination) and replaces the fixed bridging/evasion policy with net
information-gain scoring:

    S(a) = H(B_t) - E[H(B_{t+1} | a, obs)] - lambda * C_dis(a)

over the action set {construct relay, flee/reposition, switch channel,
hold & observe}. S(relay) is injected into the A* edge cost as a discount;
hard constraints gate the search (geometric connectivity AND the 12.5 dB
PHY-measured packet gate). lambda is the only tuning knob and is fixed by
preregistered tuning before any confirmatory run.

IG is the controller's internal objective only. Headline metrics remain
the v1-v8 pipeline (recovery rate, corrected throughput Mbps); belief
error, C_dis and per-action IG are diagnostic logs, never results.

Anti-cheat: beliefs come from received powers / beacons / SINR samples /
camera visibility only. Ground-truth jammer position is read exclusively
for the diagnostic belief-error log.
"""
import numpy as np

from .config import R2CParams
from .hybrid import HybridController
from .rf import RFLayer, RFParams

DELTA_BYTES = 48.0        # belief delta: vec 2xf64 + conf f64 + age f64 + mode post 3xf32
PACKET_GATE_DB = 12.5     # GNU Radio-measured packet-delivery gate (PHY_VALIDATION.md)
S_REF = 2.0               # constant IG scale for the edge-cost discount (not data-derived)
KB = 1024.0

# predicted entropy-reduction coefficients (preregistration §2 table)
K_HOLD_DIR, K_HOLD_MODE = 0.05, 0.02
K_FLEE_DIR = 0.10
K_SWITCH_DIR, K_SWITCH_MODE = 0.50, 0.10
K_RELAY_DIR = 0.15


def _h2(x):
    """Binary entropy (nats), 0 at the boundaries."""
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


class IGController(HybridController):
    """HybridController (v3 machinery) + IG action selection and IG-priced
    A* edge costs. variant is pinned to 'v3' internally; v1-v8 code paths
    are untouched."""

    def __init__(self, p: R2CParams, rf: RFLayer | None = None,
                 rf_p: RFParams | None = None, lam: float = 1.0,
                 seed: int = 0, **kw):
        super().__init__(p, rf=rf, rf_p=rf_p, variant="v3", **kw)
        assert "variant" not in kw, "v9 pins variant='v3'"
        self.lam = float(lam)
        self._rng = np.random.default_rng(seed)
        # jammer-mode posterior (Dirichlet pseudo-counts over the 3 modes)
        self.mode_alpha = np.ones(3)
        self._prev_jhat = None
        self._prev_chase_d = None
        # node health (beacon silence)
        self._misses = None
        # action state and IG accounting
        self._pending_action = "hold & observe"
        self._hold_skip = False
        self.ig_sum = 0.0            # sum of S(a*) over the episode
        self.steps = 0
        self.counters["dis_bits"] = 0.0
        self._dis_log = []           # measured dissemination events (bytes)
        self._c_pred = 0.0           # predicted C_dis accumulator (bits)
        self.diag_err = []           # ||believed - true jammer|| per step
        self.diag_mode_err = []      # 1 - p_posterior[true mode] per step

    # ---- belief update (phase 1, every control step) -----------------------
    def _believed_jammer(self, pos, active, pw):
        P = pw[active]
        s = P.sum()
        if s <= 0:
            return None
        return (P @ pos[active]) / s

    def _update_posteriors(self, env, t):
        pw = self.rf.sensed_jammer_power(env.pos, env.active, t)
        jhat = self._believed_jammer(env.pos, env.active, pw)
        # --- mode posterior from displacement + chase features -------------
        if jhat is not None and self._prev_jhat is not None:
            delta = float(np.linalg.norm(jhat - self._prev_jhat))
            c = env.pos[env.active].mean(axis=0)
            d = float(np.linalg.norm(jhat - c))
            dd = d - self._prev_chase_d if self._prev_chase_d is not None else 0.0
            s2, s5 = 2.0, 5.0
            like = np.array([
                np.exp(-0.5 * (delta / s2) ** 2) * np.exp(-0.5 * (dd / s2) ** 2),
                np.exp(-0.5 * ((delta - s5) / s2) ** 2) * np.exp(-0.5 * (dd / s2) ** 2),
                np.exp(-0.5 * ((delta - 2.0) / s2) ** 2) * np.exp(-0.5 * ((dd + 1.0) / 1.0) ** 2),
            ])
            self.mode_alpha += like / max(like.sum(), 1e-12)
            self._prev_chase_d = d
        if jhat is not None:
            self._prev_jhat = jhat.copy()
        # --- node health from beacon silence (gated degree) ----------------
        deg = (self.rf.gated_adjacency(env.pos, env.active, t).sum(axis=1))
        if self._misses is None:
            self._misses = np.zeros(len(env.pos))
        self._misses = np.where(deg <= 0, self._misses + 1.0, 0.0)
        # --- diagnostics (read-only ground truth) ---------------------------
        if jhat is not None:
            self.diag_err.append(float(np.linalg.norm(jhat - self.rf.jammer_pos)))
        post = self.mode_alpha / self.mode_alpha.sum()
        try:
            true_mode = ("constant", "sweep", "reactive").index(self.rf.rf.jammer_mode)
        except ValueError:
            true_mode = 0
        self.diag_mode_err.append(1.0 - float(post[true_mode]))
        return pw

    @property
    def _p_fail(self):
        return 1.0 - np.exp(-self._misses / 3.0)

    def _H(self, confs, p_conn):
        """Belief entropy: mode (nats) + per-cluster bearing uncertainty
        (1-conf bits each) + health Bernoulli entropies + topology h2."""
        post = self.mode_alpha / self.mode_alpha.sum()
        H_mode = float(-(post * np.log(post)).sum())
        H_dir = float(sum(1.0 - float(c) for c in confs))
        H_health = float(sum(_h2(v) for v in self._p_fail)) if self._misses is not None else 0.0
        return H_mode + H_dir + H_health + _h2(p_conn)

    # ---- dissemination accounting ------------------------------------------
    def _disseminate(self, hops, n_deltas=1):
        """Measure actual belief-delta dissemination along A* paths."""
        bits = DELTA_BYTES * hops * n_deltas
        self.counters["dis_bits"] += bits
        self._dis_log.append(bits)

    # ---- IG scoring (phase 2) ----------------------------------------------
    def _corridor_sinr_db(self, vec, mid):
        p_mw = self._belief_power(vec, mid)
        if p_mw <= 0:
            return -np.inf
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        return float(10.0 * np.log10(p_mw / noise))

    def _belief_power(self, vec, point):
        from .search import belief_power_at
        if vec is None:
            return 0.0
        return belief_power_at(self.rf, vec, point)

    def _score_actions(self, pos, active, graph, pw, passable):
        """Returns {action: (S, dH, c_dis_bits)} for the 4 preregistered
        actions plus per-corridor relay scores."""
        confs = []
        for k in graph.parts:
            b = self.beliefs.get(self.beliefs.key(graph.parts[k]))
            confs.append(b["conf"] if (b is not None and b["vec"] is not None) else 0.0)
        post = self.mode_alpha / self.mode_alpha.sum()
        H_mode = float(-(post * np.log(post)).sum())
        p_conn = min(1.0, len(passable) / max(len(graph.parts) - 1, 1))
        # dissemination path-length estimate: mean A* corridor hops ~ 2
        hops = 2
        n_k = max(len(graph.parts), 1)
        d_hold = sum(1.0 - c for c in confs) * K_HOLD_DIR + H_mode * K_HOLD_MODE
        d_flee = sum(1.0 - c for c in confs) * K_FLEE_DIR
        d_switch = sum(1.0 - c for c in confs) * K_SWITCH_DIR + H_mode * K_SWITCH_MODE
        c_hold = 0.0
        c_flee = DELTA_BYTES * hops * n_k          # moved-cluster deltas
        c_switch = DELTA_BYTES * hops * n_k        # all-cluster deltas
        out = {
            "hold & observe": (d_hold - self.lam * c_hold / KB, d_hold, c_hold),
            "flee/reposition": (d_flee - self.lam * c_flee / KB, d_flee, c_flee),
            "switch channel": (d_switch - self.lam * c_switch / KB, d_switch, c_switch),
        }
        relay = {}
        for a, b in passable:
            gain_topo = _h2(p_conn) - _h2(min(1.0, (len(passable) - 1)
                                              / max(len(graph.parts) - 1, 1)))
            conf = confs[list(graph.parts).index(a)] if a in graph.parts else 0.0
            d = max(gain_topo, 0.0) + (1.0 - conf) * K_RELAY_DIR
            c = DELTA_BYTES * hops
            relay[(a, b)] = (d - self.lam * c / KB, d, c)
        return out, relay

    # ---- v3 plan override with IG-priced edges + 12.5 dB gate --------------
    def _v3_plan(self, env, t):
        from .search import (CorridorGraph, astar, belief_power_at,
                             build_planned_adj, norm_power)
        pos, active, p = env.pos, env.active, self.p
        mdist = self._mission_dist(env)
        adj = ((mdist <= p.r_c) & active[:, None]
               & active[None, :]).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = __import__("scipy.sparse.csgraph", fromlist=["connected_components"]
                               ).connected_components(adj, directed=False)
        graph = CorridorGraph(self.ist, pos, active, labels, dist=mdist)
        self.beliefs.age_step()
        self.beliefs.demote_stale()
        pw = self.rf.sensed_jammer_power(pos, active, t)

        clusters = list(graph.parts)
        if len(clusters) < 2:
            self._act(env, t, graph, pw, adj, hold_only=True)
            return adj.astype(float)

        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (self.rf.rf.sinr_req_db / 10.0) * noise

        def exposure(k):
            return pw[graph.parts[k]].mean() / gate

        origin = max(clusters, key=exposure)
        self.beliefs.measure(self.beliefs.key(graph.parts[origin]), pos,
                             pw, graph.parts[origin])

        # --- 12.5 dB hard gate on believed corridor midpoints ---------------
        passable = []
        for a in clusters:
            b_a = self.beliefs.get(self.beliefs.key(graph.parts[a]))
            for b_, _len in graph.edges[a]:
                if a >= b_:
                    continue
                b_b = self.beliefs.get(self.beliefs.key(graph.parts[b_]))
                vec_a = b_a.get("vec") if b_a else None
                vec_b = b_b.get("vec") if b_b else None
                vec = vec_a if vec_a is not None else vec_b
                mid = 0.5 * (graph.centroid[a] + graph.centroid[b_])
                if self._corridor_sinr_db(vec, mid) >= PACKET_GATE_DB:
                    passable.append((a, b_))

        def on_open(k, path):
            key = self.beliefs.key(graph.parts[k])
            if k == origin:
                return
            vecs, confs = [], []
            for node in path:
                b = self.beliefs.get(self.beliefs.key(graph.parts[node]))
                if b is not None and b["vec"] is not None:
                    vecs.append(b["vec"])
                    confs.append(b["conf"])
            if vecs:
                conf = min(confs) if confs else 0.0
                w = np.array(confs) + 1e-9
                vec = np.average(np.vstack(vecs), axis=0, weights=w)
                self.beliefs.adopt(key, vec, conf, len(path))
                self._disseminate(len(path))          # measured: delta x hops
                if self.beliefs.get(key)["conf"] == 0.0:
                    self.beliefs.measure(key, pos, pw, graph.parts[k])
            else:
                self.beliefs.measure(key, pos, pw, graph.parts[k])

        def corridor_power(k, m):
            b = self.beliefs.get(self.beliefs.key(graph.parts[k]))
            if b is None or b["vec"] is None:
                return 0.0
            mid = 0.5 * (graph.centroid[k] + graph.centroid[m])
            return belief_power_at(self.rf, b["vec"], mid)

        # --- phase 2: IG scoring, action selection, effects ------------------
        scores, relay_scores = self._score_actions(pos, active, graph, pw,
                                                   passable)
        feasible = {"hold & observe": True, "switch channel": True,
                    "flee/reposition": bool(self._jammed(env, t)),
                    "construct relay": bool(passable)}
        for k_ in list(scores):
            if not feasible[k_]:
                scores[k_] = (-np.inf,) + scores[k_][1:]
        best = max(scores, key=lambda k_: (scores[k_][0], k_))
        # neighbor-consensus tie-break (deterministic, seeded)
        tops = [k_ for k_ in scores
                if scores[k_][0] >= scores[best][0] - 1e-12]
        if len(tops) > 1:
            best = tops[int(self._rng.integers(0, len(tops)))]
        S_best = scores[best][0]
        if S_best > -np.inf:
            self.ig_sum += max(S_best, 0.0)
            self._c_pred += scores[best][2]
        self._pending_action = best

        relay = {}
        for (a, b), (S, d, c) in relay_scores.items():
            relay[(a, b)] = S

        def g_cost(k, m, length):
            pen = 1.0 + self.gamma * norm_power(self.rf, corridor_power(k, m))
            s_bar = float(np.clip(relay.get((min(k, m), max(k, m)), 0.0)
                                  / S_REF, 0.0, 1.0))
            return length / p.v_max * pen * (1.0 - s_bar)

        def h(k):
            if graph.largest is None:
                return 0.0
            return float(np.linalg.norm(graph.centroid[k]
                                        - graph.centroid[graph.largest])
                         / p.v_max)

        edges, closed = astar(origin, graph, g_cost, h, on_open=on_open)
        sig = tuple(edges) if edges else None
        if sig != self._last_path:
            self.counters["replans"] += 1
            self._last_path = sig

        # --- action effects on the plan --------------------------------------
        if best == "hold & observe":
            for k_ in clusters:                        # local re-measure all
                self.beliefs.measure(self.beliefs.key(graph.parts[k_]),
                                     pos, pw, graph.parts[k_])
            return build_planned_adj(self.ist, pos, active, labels, graph,
                                     [(a, b) for a in clusters
                                      for b, _ in graph.edges[a] if a < b])
        if best == "switch channel":
            for k_ in clusters:                        # demote -> force re-sense
                key = self.beliefs.key(graph.parts[k_])
                b = self.beliefs.get(key)
                if b is not None:
                    b["age"] = self.beliefs.age_max + 1
        if edges is None:
            return adj.astype(float)
        pass_now = [(a, b) for a, b in edges
                    if (min(a, b), max(a, b)) in dict.fromkeys(passable)
                    and norm_power(self.rf, corridor_power(a, b)) < self.p_block]
        if pass_now:
            return build_planned_adj(self.ist, pos, active, labels,
                                     graph, pass_now)
        return adj.astype(float)

    def _act(self, env, t, graph, pw, adj, hold_only=False):
        """Fallback action hook when there is nothing to bridge (K < 2)."""
        self._pending_action = "hold & observe"

    # ---- velocity with per-step belief update + flee boost ------------------
    def velocity(self, env, t):
        self._update_posteriors(env, t)
        self.steps += 1
        saved = self.repel_gain
        if t % self.p.T_ist == 0 and self._pending_action == "flee/reposition":
            self.repel_gain = saved * 1.5
        v = super().velocity(env, t)
        self.repel_gain = saved
        return v

    # ---- diagnostic bundle (never headline metrics) -------------------------
    def diagnostics(self):
        T = max(self.steps, 1)
        return {
            "ig_net": self.ig_sum / T,
            "c_dis_bits": self.counters["dis_bits"],
            "c_dis_pred_bits": self._c_pred,
            "belief_err": float(np.mean(self.diag_err)) if self.diag_err else np.nan,
            "mode_err": float(np.mean(self.diag_mode_err)) if self.diag_mode_err else np.nan,
            "lambda": self.lam,
        }
