"""Swarm environment: positions, free-space path-loss links, failure injection."""
import numpy as np
import scipy.spatial

from .config import R2CParams


class LinkModel:
    """Free-space path loss: P_rx(d) = P_tx - PL0 - 10*gamma*log10(d/d0).

    A link exists iff P_rx >= P0. Because the path loss is monotone in d,
    this is equivalent to d <= r_c; max_link_distance() returns that radius
    derived from the sensitivity so both views stay consistent.
    """

    def __init__(self, p: R2CParams):
        self.p = p
        # link test is monotone in d, so precompute the equivalent range
        # threshold once instead of evaluating log10 over the full matrix.
        self.r_link = self.max_link_distance()

    def max_link_distance(self) -> float:
        p = self.p
        budget = p.p_tx_dbm - p.pl0_db - p.p0_dbm
        return p.d0 * 10.0 ** (budget / (10.0 * p.gamma))

    def recv_power(self, dist: np.ndarray) -> np.ndarray:
        p = self.p
        dist = np.maximum(dist, 1e-9)
        return p.p_tx_dbm - p.pl0_db - 10.0 * p.gamma * np.log10(dist / p.d0)

    def adjacency(self, dist: np.ndarray) -> np.ndarray:
        """Binary adjacency: link iff received power >= P0 (iff d <= r_c)."""
        return (dist <= self.r_link).astype(float)


class SwarmEnvironment:
    def __init__(self, p: R2CParams, n: int, deployment: str = "random"):
        self.p = p
        self.link = LinkModel(p)
        self.rng = np.random.default_rng(p.seed)
        self.n = n
        self.pos = self._deploy(n, deployment)
        self.active = np.ones(n, dtype=bool)  # failed nodes become inactive
        self.t = 0
        self._dist_cache = None   # invalidated on step()/inject_failures()
        self._adj_cache = None

    def _deploy(self, n: int, deployment: str) -> np.ndarray:
        if deployment == "random":
            return self.rng.uniform(0, self.p.area, size=(n, 2))
        if deployment == "grid":
            side = int(round(np.sqrt(n)))
            xs = np.linspace(0, self.p.area, side, endpoint=False)
            xs += self.p.area / (2 * side)
            gx, gy = np.meshgrid(xs, xs)
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
            return pts[:n]
        raise ValueError(deployment)

    def distances(self) -> np.ndarray:
        if self._dist_cache is None:
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            self._dist_cache = np.linalg.norm(diff, axis=-1)
        return self._dist_cache

    def adjacency(self) -> np.ndarray:
        if self._adj_cache is None:
            adj = self.link.adjacency(self.distances())
            adj[~self.active, :] = 0.0
            adj[:, ~self.active] = 0.0
            np.fill_diagonal(adj, 0.0)
            self._adj_cache = adj
        return self._adj_cache

    def active_adjacency(self) -> np.ndarray:
        """Adjacency restricted to active nodes (failed nodes excluded)."""
        adj = self.adjacency()
        return adj[np.ix_(self.active, self.active)]

    def connected(self) -> bool:
        n_act = int(self.active.sum())
        if n_act <= 1:
            return True
        n_comp = scipy.sparse.csgraph.connected_components(
            self.active_adjacency(), directed=False)[0]
        return n_comp == 1

    def inject_failures(self, ratio: float):
        """Fail a random subset of currently active nodes (links drop)."""
        idx = np.where(self.active)[0]
        k = int(round(ratio * len(idx)))
        victims = self.rng.choice(idx, size=k, replace=False)
        self.active[victims] = False
        self._dist_cache = None
        self._adj_cache = None

    def step(self, velocity: np.ndarray):
        """Move active nodes, clip speed and area bounds. One step = 1 s."""
        v = velocity.copy()
        v[~self.active] = 0.0
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.p.v_max / np.maximum(speed, 1e-12))
        v = v * scale
        self.pos[self.active] += v[self.active]
        np.clip(self.pos, 0.0, self.p.area, out=self.pos)
        self.t += 1
        self._dist_cache = None
        self._adj_cache = None

    def _cov_grid(self) -> np.ndarray:
        """Cached evaluation grid (depends only on area and resolution)."""
        g = getattr(self, "_cov_grid_pts", None)
        if g is None:
            axis = np.linspace(0, self.p.area, self.p.cov_grid)
            gx, gy = np.meshgrid(axis, axis)
            g = np.stack([gx.ravel(), gy.ravel()], axis=1)
            self._cov_grid_pts = g
        return g

    def coverage(self) -> float:
        """COV (Eq. 7 approximation): fraction of grid points covered by an
        active node's R_cov disk. Grid sampling of the exact area union.

        Uses a nearest-neighbour query (cKDTree) instead of the all-pairs
        grid x nodes broadcast: same distances, same metric, ~40x cheaper.
        """
        pos = self.pos[self.active]
        if len(pos) == 0:
            return 0.0
        d, _ = scipy.spatial.cKDTree(pos).query(self._cov_grid(), k=1,
                                                workers=-1)
        return float((d <= self.p.r_cov).mean())

    def degree(self) -> np.ndarray:
        return self.adjacency().sum(axis=1)
