"""v9: information-gain controller with a spatio-temporal probability map.

Preregistered as results/preregistration_v9_ig.md (rev 2, locked before any
run). The belief source is a (3, 200, 200) log-odds grid (world frame,
5 m cells): channel 0 P(jammer present), channel 1 P(link up | cell),
channel 2 node health. Log-odds stored, probability on read, robust
clip(±2) updates, per-mode temporal decay. Net-IG scoring S(a) = dH_roi -
lambda*C_dis(a) over {relay, flee, switch, hold}; S injected as an A*
edge-cost discount; 12.5 dB PHY corridor gate; dissemination priced and
metered; 10-bin ECE calibration gate per episode. Pure NumPy, no learned
components, deterministic under seed.

IG is internal only — headline metrics stay the v1-v8 pipeline (recovery,
corrected throughput). Ground truth is read only for diagnostics
(belief error, ECE).
"""
import numpy as np

from .config import R2CParams
from .hybrid import HybridController
from .rf import RFLayer, RFParams

DELTA_BYTES = 6.0          # int16 cell index + float32 log-odds delta
PACKET_GATE_DB = 12.5      # GNU Radio-measured packet gate (PHY_VALIDATION.md)
S_REF = 2.0                # constant IG scale for the edge discount
KB = 1024.0
ECE_BINS = 10
ECE_GATE = 0.05
# preregistered per-mode temporal model: multiplicative log-odds
# forgetting factor (1.0 = static world; <1 relaxes stale evidence back
# toward the uniform prior, which moving jammers require)
FORGET = {"constant": 1.0, "sweep": 0.995, "reactive": 0.99}
# preregistered per-mode "interior impossibility" weights: a MOVING
# jammer violates the static-source assumption, so interior negative
# evidence is weighted down by mobility (constant = stationary -> full)
NEAR_WEIGHT = {"constant": -2.0, "sweep": -0.05, "reactive": -0.2}
ANNULUS_GAIN = 2.0   # within the preregistered clip(+/-2) bound
# preregistered predicted-observation multipliers for B_pred
PRED_OBS = {"hold & observe": 1.0, "flee/reposition": 1.0,
            "switch channel": 2.0}


def _shannon(p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def ece_score(pred, truth, n_bins=ECE_BINS):
    """10-bin expected calibration error. pred in (0,1), truth in {0,1}.
    Hand-checkable: ECE = sum_m (|B_m|/N) * |acc_m - conf_m|."""
    pred = np.asarray(pred, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    n = len(pred)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for m in range(n_bins):
        if m == n_bins - 1:
            mask = (pred >= edges[m]) & (pred <= edges[m + 1])
        else:
            mask = (pred >= edges[m]) & (pred < edges[m + 1])
        if not mask.any():
            continue
        conf = pred[mask].mean()
        acc = truth[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


class GridFilter:
    """Layer 1: spatio-temporal log-odds probability map, world frame."""

    def __init__(self, area: float = 1000.0, cell: float = 5.0,
                 n_ch: int = 3):
        self.area = float(area)
        self.cell = float(cell)
        self.g = int(round(self.area / self.cell))
        self.L = np.zeros((n_ch, self.g, self.g), dtype=np.float32)
        self.version = 0
        ij = np.indices((self.g, self.g)).reshape(2, -1).T
        self._centers = ((ij + 0.5) * self.cell)[:, ::-1]  # world (x, y)

    # ---- read ---------------------------------------------------------------
    def prob(self):
        # clip avoids exp overflow; |L|>60 is 0/1 to machine precision
        L = np.clip(self.L.astype(np.float64), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-L))

    def cell_of(self, pos):
        """World (x, y) -> clipped (row, col)."""
        ij = np.floor(np.asarray(pos, dtype=float) / self.cell).astype(int)
        return np.clip(ij, 0, self.g - 1)

    def map_position(self):
        """MAP estimate of the jammer: center of the argmax ch0 cell."""
        p0 = self.prob()[0]
        ij = np.unravel_index(int(np.argmax(p0)), p0.shape)
        return (np.array(ij, dtype=float) + 0.5) * self.cell

    def expected_jammer_power(self, rf: RFLayer, point):
        """Believed jammer power at a world point: sum over cells of
        P(jammer at cell) x free-space rx power cell->point."""
        idx = np.indices((self.g, self.g)).reshape(2, -1).T   # (G, 2) rows,cols
        centers = ((idx + 0.5) * self.cell)[:, ::-1]          # world (x, y)
        p_cell = self.prob()[0].ravel()
        if p_cell.sum() <= 0:
            return 0.0
        rx = rf._rx_mw(centers, np.asarray(point, dtype=float).reshape(2),
                       rf.rf.jammer_power_dbm)
        return float(p_cell @ rx)

    # ---- update -------------------------------------------------------------
    def apply(self, ch, cells, increments):
        """Robust log-odds update; clip bounds one corrupted reading."""
        cells = np.asarray(cells)
        inc = np.clip(np.asarray(increments, dtype=np.float32), -2.0, 2.0)
        flat = np.ravel_multi_index((np.full(len(cells), ch), cells[:, 0],
                                     cells[:, 1]), self.L.shape)
        np.add.at(self.L.reshape(-1), flat, inc)

    def observe(self, pos, active, gated_adj, pw, gate_mw, mult=1):
        """One 10 Hz observation round for all three channels."""
        act = np.where(active)[0]
        if len(act) == 0:
            self.version += 1
            return
        cells = self.cell_of(pos[act])
        # ch0: sensed jammer power at own cell (mult: re-sense repeats)
        for _ in range(int(mult)):
            self.apply(0, cells, np.where(pw[act] >= gate_mw, 2.0, -2.0))
        # ch1: gated pairs + -> both cells; gated-out in-range pairs -
        adj = gated_adj.copy()
        np.fill_diagonal(adj, 0.0)
        rng_pairs = (np.linalg.norm(pos[:, None] - pos[None], axis=-1)
                     <= 120.0)
        up = np.argwhere((adj > 0) & active[:, None] & active[None, :])
        up = up[up[:, 0] < up[:, 1]]
        if len(up):
            c = self.cell_of(0.5 * (pos[up[:, 0]] + pos[up[:, 1]]))
            self.apply(1, c, np.full(len(c), 1.0))
        down = np.argwhere((adj <= 0) & rng_pairs & active[:, None]
                           & active[None, :])
        down = down[down[:, 0] < down[:, 1]]
        if len(down):
            c = self.cell_of(0.5 * (pos[down[:, 0]] + pos[down[:, 1]]))
            self.apply(1, c, np.full(len(c), -1.0))
        # ch2: presence +0.5 (active), absence -2.0 (fast saturation)
        allc = self.cell_of(pos)
        for _ in range(int(mult)):
            self.apply(2, allc, np.where(active, 0.5, -2.0))
        self.version += 1

    def decay(self, mode):
        self.L = (self.L * np.float32(FORGET[mode])).astype(np.float32)

    TOMO_SLOTS = 20   # divided sensing: slot i % K senses each step
                      # (design constant, v2 pattern; not a knob)

    def observe_ch0_tomography(self, pos, active, pw, rf, t=0):
        """Range-based RF tomography: observed jammer power at drone i
        implies the jammer lies on a circle of radius d_i (free-space
        model, anti-cheat: received power only). Cells in the annulus
        get +1, cells inside (would imply higher power than observed)
        and outside the local window get -2."""
        act = np.where(active)[0]
        act = act[act % self.TOMO_SLOTS == t % self.TOMO_SLOTS]
        if len(act) == 0:
            return
        p_dbm = 10.0 * np.log10(np.maximum(pw[act], 1e-30))
        d = rf.p.d0 * 10.0 ** ((rf.rf.jammer_power_dbm - rf.p.pl0_db
                                - p_dbm) / (10.0 * rf.p.gamma))
        # detections beyond the arena carry no spatial information
        # (jammer out of informative range): skip, do not flood negatives
        d = np.where(d <= self.area * 1.1, d, np.nan)
        half = self.cell                          # annulus half-width, 1 cell
        win = d + 6.0 * self.cell                 # local window radius
        for k, i in enumerate(act):
            if not np.isfinite(d[k]):
                continue
            diff = self._centers - pos[i]
            dist = np.einsum("ij,ij->i", diff, diff) ** 0.5
            ann = np.abs(dist - d[k]) <= half
            near = (dist < d[k] - half) & (dist <= win[k])
            if ann.any():
                self.apply(0, self.cell_of(self._centers[ann]), ANNULUS_GAIN)
            if near.any():
                self.apply(0, self.cell_of(self._centers[near]),
                           NEAR_WEIGHT[rf.rf.jammer_mode])
        self.version += 1

    # ---- IG support ----------------------------------------------------------
    def entropy_roi(self, roi):
        y0, y1, x0, x1 = roi
        return float(_shannon(self.prob()[:, y0:y1, x0:x1]).sum())

    def predicted_entropy_roi(self, roi, ch, cells, increments, mult=1.0):
        """H of the ROI if the given observation were applied (not committed)."""
        y0, y1, x0, x1 = roi
        L = np.clip(self.L[:, y0:y1, x0:x1].astype(np.float64), -60.0, 60.0).copy()
        cells = np.asarray(cells)
        if len(cells):
            inc = np.clip(np.asarray(increments, dtype=float) * mult,
                          -2.0, 2.0)
            sub = (cells[:, 0] - y0, cells[:, 1] - x0)
            inside = (sub[0] >= 0) & (sub[0] < L.shape[1]) \
                     & (sub[1] >= 0) & (sub[1] < L.shape[2])
            np.add.at(L[ch], (sub[0][inside], sub[1][inside]),
                      inc[inside])
        return float(_shannon(1.0 / (1.0 + np.exp(-L))).sum())

    def deltas_bytes(self, n_changed, hops=1):
        return float(DELTA_BYTES) * n_changed * hops


def _roi_of(pos, active, gf: GridFilter, margin=2):
    cells = gf.cell_of(pos[active])
    y0 = max(int(cells[:, 0].min()) - margin, 0)
    y1 = min(int(cells[:, 0].max()) + margin + 1, gf.g)
    x0 = max(int(cells[:, 1].min()) - margin, 0)
    x1 = min(int(cells[:, 1].max()) + margin + 1, gf.g)
    return (y0, y1, x0, x1)


class IGController(HybridController):
    """v9: HybridController (v3 machinery) with the grid map as belief
    source for IG scoring. v1-v8 code untouched."""

    def __init__(self, p: R2CParams, rf: RFLayer | None = None,
                 rf_p: RFParams | None = None, lam: float = 1.0,
                 seed: int = 0, **kw):
        super().__init__(p, rf=rf, rf_p=rf_p, variant="v3", **kw)
        assert "variant" not in kw, "v9 pins variant='v3'"
        self.lam = float(lam)
        self._rng = np.random.default_rng(seed)
        self.grid = GridFilter(area=p.area)
        self.ig_sum = 0.0
        self.steps = 0
        self.counters["dis_bits"] = 0.0
        self._dis_log = []
        self._c_pred = 0.0
        self._belief_err = []
        self._ece = []            # per-episode (ece0, ece1, ece2), filled at end
        self._pending_action = "hold & observe"
        self._pending_version = 0
        self._passable = []
        self._switch_boost = False

    # ---- Layer 4a: metering ---------------------------------------------------
    def _disseminate(self, n_changed, hops):
        bits = DELTA_BYTES * n_changed * hops
        self.counters["dis_bits"] += bits
        self._dis_log.append(bits)

    # ---- Layer 2: net-IG scoring ----------------------------------------------
    def _score_actions(self, pos, active, graph, roi_cells):
        """Returns {action: (S, dH, c_pred_bits)} using ROI entropy on the
        grid, and per-corridor relay scores keyed (a, b)."""
        gf = self.grid
        roi = _roi_of(pos, active, gf)
        cells, incs = roi_cells
        H_before = gf.entropy_roi(roi)
        scores = {}
        for a, mult in PRED_OBS.items():
            cells_a, incs_a, ch_a = self._action_observation(a, pos, active,
                                                             graph, cells,
                                                             incs)
            H_after = gf.predicted_entropy_roi(roi, ch_a, cells_a, incs_a,
                                               mult=mult)
            dH = max(H_before - H_after, 0.0)
            n_changed = len(np.atleast_1d(cells_a)) or 1
            c = gf.deltas_bytes(n_changed, hops=2)
            scores[a] = (dH - self.lam * c / KB, dH, c)
        relay = {}
        passable = getattr(self, "_passable", [])
        for (a, b) in passable:
            rc = self._corridor_cells(pos, graph, a, b)
            inc = np.full(len(rc), 1.0)
            H_after = gf.predicted_entropy_roi(roi, 0, rc, inc, mult=1.0)
            dH = max(H_before - H_after, 0.0) + 0.15  # new-edge topology info
            c = gf.deltas_bytes(max(len(rc), 1), hops=2)
            relay[(a, b)] = (dH - self.lam * c / KB, dH, c)
        return scores, relay

    def _action_observation(self, action, pos, active, graph, cells, incs):
        """Predicted observation cells for B_pred (channel, cells, incs)."""
        if action == "hold & observe":
            return cells, incs, 0
        if action == "switch channel":
            return cells, incs, 0          # same cells, doubled via mult
        if action == "flee/reposition":
            ctr = pos[active].mean(axis=0)
            b = self.beliefs.get(self.beliefs.key(
                np.where(active)[0])) if self.beliefs else None
            away = ctr - (b["vec"] if b and b["vec"] is not None else ctr)
            n = np.linalg.norm(away)
            step = away / n * self.p.v_max if n > 1e-9 else np.zeros(2)
            new_pos = pos + np.where(active[:, None], step, 0.0)
            c2 = self.grid.cell_of(new_pos[active])
            return c2, np.full(len(c2), 2.0), 0
        # construct relay: corridor endpoint cells
        cells_r, incs_r = [], []
        for (a, b) in getattr(self, "_passable", []):
            rc = self._corridor_cells(pos, graph, a, b)
            cells_r.append(rc)
            incs_r.append(np.full(len(rc), 1.0))
        if cells_r:
            return (np.vstack(cells_r), np.concatenate(incs_r), 0)
        return np.empty((0, 2), dtype=int), np.empty(0), 0

    def _corridor_cells(self, pos, graph, a, b):
        ba = np.atleast_1d(graph.boundary[a])
        bb = np.atleast_1d(graph.boundary[b])
        return self.grid.cell_of(pos[np.concatenate([ba, bb])])

    # ---- Layer 3: plan with IG-priced edges + 12.5 dB gate --------------------
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
        # observation round for the grid (Layer 1)
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (self.rf.rf.sinr_req_db / 10.0) * noise

        clusters = list(graph.parts)
        if len(clusters) < 2:
            self._pending_action = "hold & observe"
            return adj.astype(float)

        def exposure(k):
            return pw[graph.parts[k]].mean() / gate

        origin = max(clusters, key=exposure)
        self.beliefs.measure(self.beliefs.key(graph.parts[origin]), pos,
                             pw, graph.parts[origin])

        # --- 12.5 dB hard gate on believed corridor midpoints (grid belief) ---
        passable = []
        for a in clusters:
            for b_, _len in graph.edges[a]:
                if a >= b_:
                    continue
                mid = 0.5 * (graph.centroid[a] + graph.centroid[b_])
                p_mid = self.grid.expected_jammer_power(self.rf, mid)
                sinr_db = (10.0 * np.log10(p_mid / noise)
                           if p_mid > 0 else -np.inf)
                if sinr_db >= PACKET_GATE_DB:
                    passable.append((a, b_))
        self._passable = passable

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
                self._disseminate(1, len(path))     # one delta per adoption
                if self.beliefs.get(key)["conf"] == 0.0:
                    self.beliefs.measure(key, pos, pw, graph.parts[k])
            else:
                self.beliefs.measure(key, pos, pw, graph.parts[k])

        def corridor_power(k, m):
            mid = 0.5 * (graph.centroid[k] + graph.centroid[m])
            p_mid = self.grid.expected_jammer_power(self.rf, mid)
            return max(p_mid, 0.0)

        # --- Layer 2: score actions on the current grid version ----------------
        roi_cells = self.grid.cell_of(pos[active]), np.where(
            pw[active] >= gate, 2.0, -2.0)
        scores, relay_scores = self._score_actions(pos, active, graph,
                                                   roi_cells)
        # --- Layer 4b: staleness rule (re-score once on version < v-2) --------
        if self.grid.version - self._pending_version > 2 and \
                self._pending_action != "hold & observe":
            scores, relay_scores = self._score_actions(pos, active, graph,
                                                       roi_cells)
        feasible = {"hold & observe": True, "switch channel": True,
                    "flee/reposition": bool(self._jammed(env, t)),
                    "construct relay": bool(passable)}
        for k_ in list(scores):
            if not feasible[k_]:
                scores[k_] = (-np.inf,) + scores[k_][1:]
        best = max(scores, key=lambda k_: (scores[k_][0], k_))
        tops = [k_ for k_ in scores
                if scores[k_][0] >= scores[best][0] - 1e-12]
        if len(tops) > 1:
            best = tops[int(self._rng.integers(0, len(tops)))]
        S_best = scores[best][0]
        if S_best > -np.inf:
            self.ig_sum += max(S_best, 0.0)
            self._c_pred += scores[best][2]
        self._pending_action = best
        self._pending_version = self.grid.version
        # Layer 4 sharing payload: action code + tie-break + version (10 B)
        self.counters["dis_bits"] += 10.0
        self._dis_log.append(10.0)

        def g_cost(k, m, length):
            pen = 1.0 + self.gamma * norm_power(self.rf, corridor_power(k, m))
            s_bar = float(np.clip(
                relay_scores.get((min(k, m), max(k, m)), (0.0,))[0] / S_REF,
                0.0, 1.0))
            return length / p.v_max * pen * (1.0 - s_bar)

        def h(k):
            if graph.largest is None:
                return 0.0
            return float(np.linalg.norm(graph.centroid[k]
                                        - graph.centroid[graph.largest])
                         / p.v_max)

        edges, _closed = astar(origin, graph, g_cost, h, on_open=on_open)
        sig = tuple(edges) if edges else None
        if sig != self._last_path:
            self.counters["replans"] += 1
            self._last_path = sig

        if best == "hold & observe":
            for k_ in clusters:
                self.beliefs.measure(self.beliefs.key(graph.parts[k_]),
                                     pos, pw, graph.parts[k_])
            return build_planned_adj(self.ist, pos, active, labels, graph,
                                     [(a, b) for a in clusters
                                      for b, _ in graph.edges[a] if a < b])
        if best == "switch channel":
            self._switch_boost = True  # double-increment next observation
        if edges is None:
            return adj.astype(float)
        pass_now = [(a, b) for a, b in edges
                    if (min(a, b), max(a, b)) in dict.fromkeys(passable)
                    and norm_power(self.rf, corridor_power(a, b)) < self.p_block]
        if pass_now:
            return build_planned_adj(self.ist, pos, active, labels,
                                     graph, pass_now)
        return adj.astype(float)

    def velocity(self, env, t):
        self.steps += 1
        # Layer 1 at 10 Hz: observe + temporal decay every control step
        pos, active = env.pos, env.active
        pw = self.rf.sensed_jammer_power(pos, active, t)
        noise = 10.0 ** (self.rf.rf.noise_dbm / 10.0)
        gate = 10.0 ** (self.rf.rf.sinr_req_db / 10.0) * noise
        gadj = self.rf.gated_adjacency(pos, active, t)
        self.grid.observe(pos, active, gadj, pw, gate,
                          mult=2 if self._switch_boost else 1)
        self.grid.observe_ch0_tomography(pos, active, pw, self.rf, t)
        self._switch_boost = False
        self.grid.decay(self.rf.rf.jammer_mode)
        if t % self.p.T_ist == 0:
            jmap = self.grid.map_position()
            jtrue = self.rf._jammer_position_peek(t, pos, active)
            self._belief_err.append(float(np.linalg.norm(jmap - jtrue)))
        saved = self.repel_gain
        if t % self.p.T_ist == 0 and self._pending_action == "flee/reposition":
            self.repel_gain = saved * 1.5
        v = super().velocity(env, t)
        self.repel_gain = saved
        return v

    # ---- Layer 4c: calibration (ECE) + diagnostics ----------------------------
    def _ground_truth_ece(self, env):
        gf = self.grid
        pr = gf.prob()
        # ch0: jammer cell neighborhood + strided background sample
        # (truth from the simulator's position function: the stored
        # jammer_pos attr is stale in sweep mode by design)
        jtrue = self.rf._jammer_position_peek(self.steps, env.pos, env.active)
        jc = gf.cell_of(np.asarray(jtrue).reshape(1, 2))[0]
        nb, nb_truth = [], []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                c = (int(jc[0]) + dy, int(jc[1]) + dx)
                if 0 <= c[0] < gf.g and 0 <= c[1] < gf.g:
                    nb.append(c)
                    nb_truth.append(1.0 if (dy == 0 and dx == 0) else 0.0)
        bg = np.array(np.meshgrid(np.arange(0, gf.g, 10),
                                  np.arange(0, gf.g, 10))).T.reshape(-1, 2)
        cells = np.vstack([np.asarray(nb, dtype=int), bg])
        truth = np.concatenate([np.asarray(nb_truth, dtype=float),
                                np.zeros(len(bg))])
        e0 = ece_score(pr[0][cells[:, 0], cells[:, 1]], truth)
        # ch1: deterministic sample of in-range pairs
        n = len(env.pos)
        ids = np.arange(n)
        pairs = [(i, (i * 7 + 3) % n) for i in range(0, n, max(n // 200, 1))]
        gadj = self.rf.gated_adjacency(env.pos, env.active, self.steps)
        pred1, truth1 = [], []
        pos = env.pos
        for i, j in pairs:
            if i == j or not (env.active[i] and env.active[j]):
                continue
            if np.linalg.norm(pos[i] - pos[j]) > self.p.r_c:
                continue
            c1 = gf.cell_of(pos[i].reshape(1, 2))[0]
            c2 = gf.cell_of(pos[j].reshape(1, 2))[0]
            pred1.append(0.5 * (pr[1][c1[0], c1[1]] + pr[1][c2[0], c2[1]]))
            truth1.append(1.0 if gadj[i, j] > 0 else 0.0)
        e1 = ece_score(np.asarray(pred1), np.asarray(truth1))
        # ch2: every node's cell vs active
        c_all = gf.cell_of(pos)
        e2 = ece_score(pr[2][c_all[:, 0], c_all[:, 1]],
                       env.active.astype(float))
        return e0, e1, e2

    def diagnostics(self):
        if not self._ece and self.steps > 0:
            pass  # ECE computed per episode by the runner via finalize()
        T = max(self.steps, 1)
        d = {
            "ig_net": self.ig_sum / T,
            "c_dis_bits": self.counters["dis_bits"],
            "c_dis_pred_bits": self._c_pred,
            "belief_version": float(self.grid.version),
            "belief_err": float(np.mean(self._belief_err))
            if self._belief_err else np.nan,
            "lambda": self.lam,
        }
        if self._ece:
            e0, e1, e2 = np.mean(self._ece, axis=0)
            d.update({"ece_map": float(e0), "ece_link": float(e1),
                      "ece_health": float(e2),
                      "ece_mean": float(np.mean([e0, e1, e2]))})
        return d

    def finalize_episode(self, env):
        """Runner calls this at episode end to log the ECE calibration gate."""
        self._ece.append(self._ground_truth_ece(env))
