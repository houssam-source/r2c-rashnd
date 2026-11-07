"""v3 search primitive: A* over the cluster graph with belief-carrying
expansion, plus the belief store with adopt / re-measure / staleness rules.

Design spec: Search-Guided Belief Dissemination (2026-11-07).
Everything here is anti-cheat compliant: beliefs are power-weighted
centroids derived from received-power samples only; propagated beliefs
interpolate the same measurement model at believed jammer positions.
"""
import heapq
import itertools

import numpy as np


# ---------------------------------------------------------------------------
# corridor graph
# ---------------------------------------------------------------------------
class CorridorGraph:
    """Clusters = nodes; edges = boundary pairs within R̂_com (IST machinery).

    Edge length = min distance between the clusters' convex-hull boundary
    nodes (the pair the IST would bridge).
    """

    def __init__(self, ist, pos, active, labels, dist=None):
        p = ist.p
        r_virt = p.r_com_boost * p.r_c
        self.dist = dist if dist is not None else             np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        self.parts, self.boundary, self.centroid = {}, {}, {}
        for k in set(labels[active]):
            idx = np.where((labels == k) & active)[0]
            self.parts[k] = idx
            self.boundary[k] = idx[ist._boundary_nodes(pos[idx])]
            self.centroid[k] = pos[idx].mean(axis=0)
        self.edges = {k: [] for k in self.parts}
        keys = list(self.parts)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                D = self.dist[np.ix_(self.boundary[a], self.boundary[b])]
                dmin = float(D.min()) if D.size else np.inf
                if dmin <= r_virt:
                    self.edges[a].append((b, dmin))
                    self.edges[b].append((a, dmin))
        if self.parts:
            self.largest = max(self.parts, key=lambda k: len(self.parts[k]))
        else:
            self.largest = None

    def adjacent_to_largest(self, k):
        return any(m == self.largest for m, _ in self.edges.get(k, ()))


# ---------------------------------------------------------------------------
# A* with on_open adoption callback
# ---------------------------------------------------------------------------
def astar(origin, graph, g_cost, h, on_open=None, goal=None):
    """A* over the corridor graph.

    g_cost(k, m, length) -> cost of edge k->m (belief-penalized).
    h(k) -> heuristic (admissible).
    on_open(k, path) -> called when a node is popped (belief adoption hook);
        path = nodes from origin to k.
    goal(k) -> bool; defaults to "adjacent to largest cluster".
    Returns (path_edges, closed_order) or (None, closed_order).
    """
    if goal is None:
        goal = graph.adjacent_to_largest
    counter = itertools.count()
    g, came = {origin: 0.0}, {}
    open_h = [(h(origin), next(counter), origin)]
    closed = []
    while open_h:
        _, _, k = heapq.heappop(open_h)
        if k in closed:
            continue
        closed.append(k)
        # reconstruct path origin..k
        path = [k]
        while path[-1] in came:
            path.append(came[path[-1]])
        path.reverse()
        if on_open is not None:
            on_open(k, path)
        if goal(k) and k != origin:
            edges = list(zip(path[:-1], path[1:]))
            return edges, closed
        for m, length in graph.edges.get(k, ()):
            if m in closed:
                continue
            ng = g[k] + g_cost(k, m, length)
            if ng < g.get(m, np.inf):
                g[m] = ng
                came[m] = k
                heapq.heappush(open_h, (ng + h(m), next(counter), m))
    return None, closed


# ---------------------------------------------------------------------------
# belief store
# ---------------------------------------------------------------------------
class BeliefStore:
    """Per-cluster jammer beliefs with adopt / re-measure / staleness rules.

    Keyed by the sorted member tuple so beliefs survive across IST
    invocations while membership is unchanged.
    """

    def __init__(self, contrast_ref, conf_min, age_max, counters):
        self.store = {}          # key -> dict(vec, conf, age, source)
        self.contrast_ref = contrast_ref
        self.conf_min = conf_min
        self.age_max = age_max
        self.counters = counters

    @staticmethod
    def key(members):
        return tuple(sorted(int(i) for i in members))

    def age_step(self):
        for b in self.store.values():
            b["age"] += 1

    def demote_stale(self):
        for b in self.store.values():
            if b["age"] > self.age_max and b["conf"] > 0:
                b["conf"] = 0.0     # forces local re-measure on next need

    def measure(self, key, pos, pw, members):
        """Local divided-sensing measurement for one cluster."""
        P = pw[members]
        s = P.sum()
        if s <= 0:
            self.store[key] = dict(vec=None, conf=0.0, age=0, source="local")
            return self.store[key]
        vec = (P @ pos[members]) / s
        contrast = (P.max() - P.min()) / (P.mean() + 1e-12)
        conf = float(np.clip(contrast / self.contrast_ref, 0.0, 1.0))
        self.store[key] = dict(vec=vec, conf=conf, age=0, source="local")
        self.counters["sensing_events"] += 1
        return self.store[key]

    def adopt(self, key, vec, conf, age):
        if conf >= self.conf_min:
            self.store[key] = dict(vec=vec.copy(), conf=float(conf),
                                   age=int(age), source="adopted")
        else:
            self.store[key] = dict(vec=None, conf=0.0, age=0, source="local")

    def get(self, key):
        return self.store.get(key)


# ---------------------------------------------------------------------------
# belief-penalized cost helpers
# ---------------------------------------------------------------------------
def belief_power_at(rf, belief_vec, point):
    """Jammer power [mW] the *belief* implies at a point (measurement model
    evaluated at the believed jammer position — no ground truth)."""
    return float(rf._rx_mw(np.asarray(belief_vec, dtype=float),
                           np.asarray(point, dtype=float).reshape(1, 2),
                           rf.rf.jammer_power_dbm)[0])


def norm_power(rf, p_mw):
    """Normalize by the delivery gate: P / (SINR_req * noise)."""
    noise = 10.0 ** (rf.rf.noise_dbm / 10.0)
    gate = 10.0 ** (rf.rf.sinr_req_db / 10.0) * noise
    return p_mw / gate


# ---------------------------------------------------------------------------
# planned IST adjacency: intra-part edges + virtual edges for chosen corridors
# ---------------------------------------------------------------------------
def build_planned_adj(ist, pos, active, labels, graph, corridors):
    """Clone of IST._multipartite restricted to the given corridor pairs
    (plus all intra-part edges). corridors = iterable of (a, b) labels."""
    p = ist.p
    n = len(pos)
    adj = np.zeros((n, n))
    dist = graph.dist
    same = labels[:, None] == labels[None, :]
    adj[same] = np.maximum(0.0, 1.0 - dist[same] / p.r_c)
    r_virt = p.r_com_boost * p.r_c
    pairs = set()
    for a, b in corridors:
        pairs.add((a, b))
        pairs.add((b, a))
    for a, b in pairs:
        if a not in graph.boundary or b not in graph.parts:
            continue
        src, dst_idx = graph.boundary[a], graph.parts[b]
        D = dist[np.ix_(src, dst_idx)]
        Dm = np.where(D <= r_virt, D, np.inf)
        if not np.isfinite(Dm).any():
            continue
        m_eff = min(ist.top_m, Dm.shape[1])
        top = np.argsort(Dm, axis=1)[:, :m_eff]
        rows = np.repeat(np.arange(len(src)), m_eff)
        cols = dst_idx[top.ravel()]
        w = np.maximum(0.0, 1.0 - D[rows, top.ravel()] / r_virt)
        adj[src[rows], cols] = np.maximum(adj[src[rows], cols], w)
    adj[~active, :] = 0.0
    adj[:, ~active] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj
