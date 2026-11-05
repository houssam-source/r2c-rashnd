"""The 8 comparison baselines from the R2C paper (guide §3).

NOTE: none of these methods has public code; each is a faithful-in-spirit
re-implementation of the algorithm class it names, with assumptions marked.
All expose velocity(env, t) -> (n, 2), like R2CController.
"""
import numpy as np
import scipy.sparse.csgraph

from .config import R2CParams
from .ise import ISE
from .ist import IST


def _components(env):
    adj = env.adjacency()  # full matrix: failed nodes are isolated singletons
    n = scipy.sparse.csgraph.connected_components(adj, directed=False)
    return n[0], n[1], adj


def _toward(src, dst):
    d = dst - src
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(2)


class CEN:
    """Centralized self-healing (Assumption: centralized component-merge
    repair). Nodes outside the largest connected component move toward their
    nearest node in it; that component's nodes keep ISE spacing."""

    def __init__(self, p: R2CParams):
        self.ise = ISE(p)
        self.p = p

    def velocity(self, env, t):
        n_comp, labels, adj = _components(env)
        v = self.ise.velocity(env.pos, env.active)
        if n_comp <= 1:
            return v
        largest = np.bincount(labels[env.active]).argmax()
        big_pos = env.pos[(labels == largest) & env.active]
        for i in np.where(env.active & (labels != largest))[0]:
            d = np.linalg.norm(big_pos - env.pos[i], axis=1)
            v[i] = _toward(env.pos[i], big_pos[d.argmin()]) * self.p.v_max
        return v


class HERO:
    """Hierarchical repair (Assumption: cluster-head guided). Nodes with
    degree below the median follow their nearest high-degree cluster head;
    heads use ISE spacing."""

    def __init__(self, p: R2CParams):
        self.ise = ISE(p)
        self.p = p

    def velocity(self, env, t):
        _, _, adj = _components(env)
        deg = adj.sum(1)
        v = self.ise.velocity(env.pos, env.active)
        if deg[env.active].size == 0:
            return v
        med = np.median(deg[env.active])
        heads = np.where(env.active & (deg >= med))[0]
        if len(heads) == 0:
            return v
        for i in np.where(env.active & (deg < med))[0]:
            d = np.linalg.norm(env.pos[heads] - env.pos[i], axis=1)
            v[i] = _toward(env.pos[i], env.pos[heads[d.argmin()]]) * self.p.v_max
        return v


class SIDR:
    """Information-driven relocation (Assumption: degree-guided). Isolated
    nodes seek the nearest active node; low-degree nodes seek their neighbor
    centroid; otherwise buffered virtual forces."""

    def __init__(self, p: R2CParams):
        self.ise = ISE(p)
        self.p = p

    def velocity(self, env, t):
        _, _, adj = _components(env)
        v = self.ise.velocity(env.pos, env.active)
        for i in np.where(env.active)[0]:
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs) == 0:
                others = np.where(env.active)[0]
                others = others[others != i]
                if len(others):
                    d = np.linalg.norm(env.pos[others] - env.pos[i], axis=1)
                    v[i] = _toward(env.pos[i], env.pos[others[d.argmin()]]) * self.p.v_max
            elif len(nbrs) < 2:
                v[i] = _toward(env.pos[i], env.pos[nbrs].mean(axis=0)) * self.p.v_max
        return v


class MDSGAPF:
    """MDSG-APF: artificial potential field without the buffer zone.
    Attraction above r_c/2, repulsion below it (Assumption: standard APF
    threshold), no neutral band."""

    def __init__(self, p: R2CParams):
        self.p = p

    def velocity(self, env, t):
        p = self.p
        diff = env.pos[:, None, :] - env.pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        unit = diff / np.maximum(dist, 1e-12)[..., None]
        r = p.r_c / 2.0
        in_range = (dist > 1e-9) & (dist <= p.r_c) & env.active[None, :]
        mag = (np.where(dist < r, r - dist, 0.0) -
               np.where(dist > r, dist - r, 0.0)) * in_range
        F = (unit * mag[..., None]).sum(axis=1) * p.decay
        speed = np.linalg.norm(F, axis=1, keepdims=True)
        v = F / np.maximum(speed, 1e-12) * np.minimum(1.0, speed) * p.v_max
        v[~env.active] = 0.0
        return v


class GCNBaseline:
    """Plain GCN repair: IST alone at its invocation interval, no ISE,
    no AWF (fusion weight fixed to 1)."""

    def __init__(self, p: R2CParams):
        self.p = p
        self.ist = IST(p)

    def velocity(self, env, t):
        if t % self.p.T_ist == 0:
            self.ist.train_step(env.pos, env.active)
            return self.ist.translation(env.pos, env.active)
        return np.zeros_like(env.pos)


class CRMGC:
    """Connectivity-restoration multi-GCN (Assumption: ensemble of 3
    Chebyshev GCNs, translations averaged)."""

    def __init__(self, p: R2CParams):
        self.p = p
        self.members = [IST(R2CParams(**{**p.__dict__, "seed": p.seed + k}))
                        for k in range(3)]

    def velocity(self, env, t):
        if t % self.p.T_ist != 0:
            return np.zeros_like(env.pos)
        v = np.zeros_like(env.pos)
        for m in self.members:
            m.train_step(env.pos, env.active)
            v += m.translation(env.pos, env.active)
        return v / len(self.members)


class DEMD:
    """Distributed estimation of missing nodes (Assumption: nodes dead-reckon
    lost neighbors' last known positions and move to fill the gaps; otherwise
    buffered virtual forces)."""

    def __init__(self, p: R2CParams):
        self.ise = ISE(p)
        self.p = p
        self.last_seen = {}

    def velocity(self, env, t):
        adj = env.adjacency()
        v = self.ise.velocity(env.pos, env.active)
        for i in np.where(env.active)[0]:
            lost = [j for j in np.where(adj[i] == 0)[0]
                    if j != i and j in self.last_seen and not env.active[j]]
            if lost:
                tgt = np.mean([self.last_seen[j] for j in lost], axis=0)
                v[i] = _toward(env.pos[i], tgt) * self.p.v_max
        # remember neighbors seen this step
        for i in np.where(env.active)[0]:
            for j in np.where(adj[i] > 0)[0]:
                self.last_seen[int(j)] = env.pos[j].copy()
        return v


class MDSGGC:
    """MDSG-GC: MDSG-APF forces passed through one graph-convolution-like
    smoothing step (forces neighbor-averaged over the communication graph)."""

    def __init__(self, p: R2CParams):
        self.apf = MDSGAPF(p)
        self.p = p

    def velocity(self, env, t):
        v = self.apf.velocity(env, t)
        adj = env.adjacency()  # full 200x200; failed nodes are isolated
        deg = np.maximum(adj.sum(1), 1.0)
        smooth = (adj @ v) / deg[:, None]
        out = 0.5 * (v + smooth)
        out[~env.active] = 0.0
        return out


BASELINES = {
    "CEN": CEN, "HERO": HERO, "SIDR": SIDR, "MDSG-APF": MDSGAPF,
    "GCN": GCNBaseline, "CR-MGC": CRMGC, "DEMD": DEMD, "MDSG-GC": MDSGGC,
}
