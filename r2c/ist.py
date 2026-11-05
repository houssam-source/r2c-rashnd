"""IST: proactive topology-repair module (first-order spectral GCN).

Faithful to the AAAI-26 paper (results/paper_reference.md §4):

- Multipartite graph: connected subnetworks are the parts; virtual edges
  run from convex-hull boundary nodes of each part to the top-M nearest
  nodes of every OTHER part within boosted range R^_com, weighted
  A(i,j) = 1 - ||pi-pj||/R^_com. Intra-part edges use the same distance
  decay over R_c.
- GCN: 8 first-order Chebyshev-style layers H = ReLU((I - eps_l L~) H W),
  eps_l learnable (paper: "Chebyshev-first-order spectral GCN",
  H^(l) = sigma((I - eps L~) H^(l-1) W^(l))).
- Readout: average-pool per subnetwork -> 2-layer MLP -> delta_k = tanh(.)
  in (-1,1)^2, one translation per subnetwork (paper: tanh(phi(z)) + 1
  scaled by [0.5Lx, 0.5Ly]; direction is what survives AWF normalization).
- Joint loss (paper): L = L_conn + L_time + L_move with
  L_conn = C_t - 1 (component count, via a soft spectral proxy),
  L_time = max_k ||delta_k|| / v_max, L_move = mean_k ||delta_k||.

Assumptions: M = 3 top virtual neighbors; L_conn uses a soft count of
Laplacian eigenvalues below a threshold (paper gives no smooth surrogate).
"""
import numpy as np
import scipy.spatial
import scipy.sparse.csgraph
import torch
import torch.nn as nn

from .config import R2CParams


def _normalized_laplacian(adj: np.ndarray) -> torch.Tensor:
    """L~ = I - D^-1/2 A D^-1/2, spectrum in [0, 2] (paper's L~)."""
    a = torch.tensor(adj, dtype=torch.float32)
    deg = a.sum(1)
    dinv = torch.diag(torch.pow(torch.maximum(deg, torch.tensor(1e-12)), -0.5))
    return torch.eye(len(a), device=a.device) - dinv @ a @ dinv


class FirstOrderChebLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.5))  # learnable, paper's eps
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, lnorm: torch.Tensor) -> torch.Tensor:
        support = torch.eye(len(x), device=x.device) - self.eps * lnorm
        return support @ x @ self.weight + self.bias


class ChebNet(nn.Module):
    """8 first-order layers, d=3 -> 512; per-subnetwork tanh translation."""

    def __init__(self, p: R2CParams, in_dim: int = 3):
        super().__init__()
        h = p.gcn_hidden
        dims = [in_dim] + [h] * p.gcn_layers
        self.convs = nn.ModuleList(
            FirstOrderChebLayer(dims[i], dims[i + 1])
            for i in range(p.gcn_layers))
        self.head = nn.Sequential(
            nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh())
        self.n_layers = p.gcn_layers

    def forward(self, x, lnorm, labels, n_parts):
        for conv in self.convs:
            x = torch.relu(conv(x, lnorm))
        deltas = []
        for k in range(n_parts):
            mask = labels == k
            pooled = x[mask].mean(dim=0)          # average-pool per subnetwork
            deltas.append(self.head(pooled))
        return torch.stack(deltas)                # (n_parts, 2) in (-1,1)^2


class IST:
    def __init__(self, p: R2CParams):
        self.p = p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChebNet(p).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.lr)
        self.top_m = getattr(p, "top_m", 3)

    # ---- graphs -----------------------------------------------------------

    def _parts(self, pos: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Subnetwork labels from the geometric communication graph."""
        adj = ((np.linalg.norm(pos[:, None] - pos[None, :], axis=-1) <= self.p.r_c)
               & active[:, None] & active[None, :]).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
        # renumber so active parts are 0..K-1 and failed nodes share a part
        remap = {old: k for k, old in enumerate(sorted(set(labels[active])))}
        out = np.array([remap.get(l, len(remap)) for l in labels])
        return out

    def _boundary_nodes(self, part_pos: np.ndarray) -> np.ndarray:
        """Convex-hull boundary node indices within one part."""
        if len(part_pos) < 3:
            return np.arange(len(part_pos))
        try:
            return scipy.spatial.ConvexHull(part_pos).vertices
        except scipy.spatial.QhullError:
            return np.arange(len(part_pos))

    def _multipartite(self, pos: np.ndarray, active: np.ndarray,
                      labels: np.ndarray) -> np.ndarray:
        """Multipartite graph with top-M virtual edges across parts
        (A(i,j) = 1 - d/R, paper construction)."""
        p = self.p
        n = len(pos)
        adj = np.zeros((n, n))
        dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        same = labels[:, None] == labels[None, :]
        adj[same] = np.maximum(0.0, 1.0 - dist[same] / p.r_c)
        parts = sorted(set(labels[active]))
        r_virt = p.r_com_boost * p.r_c
        for i, a in enumerate(parts):
            idx_a = np.where(labels == a)[0]
            bound_a = idx_a[self._boundary_nodes(pos[idx_a])]
            for b in parts[i + 1:]:
                idx_b = np.where(labels == b)[0]
                bound_b = idx_b[self._boundary_nodes(pos[idx_b])]
                # one matrix op per part pair: distances boundary_a x part_b
                for src, dst_idx in ((bound_a, idx_b), (bound_b, idx_a)):
                    D = dist[np.ix_(src, dst_idx)]
                    Dm = np.where(D <= r_virt, D, np.inf)
                    if not np.isfinite(Dm).any():
                        continue
                    m_eff = min(self.top_m, Dm.shape[1])
                    top = np.argsort(Dm, axis=1)[:, :m_eff]
                    rows = np.repeat(np.arange(len(src)), m_eff)
                    cols = dst_idx[top.ravel()]
                    w = np.maximum(0.0, 1.0 - D[rows, top.ravel()] / r_virt)
                    adj[src[rows], cols] = np.maximum(adj[src[rows], cols], w)
        adj[~active, :] = 0.0
        adj[:, ~active] = 0.0
        np.fill_diagonal(adj, 0.0)
        return adj

    # ---- features / loss --------------------------------------------------

    def _features(self, pos: np.ndarray, active: np.ndarray) -> torch.Tensor:
        t = torch.tensor(pos / self.p.area, dtype=torch.float32)
        act = torch.tensor(active, dtype=torch.float32).unsqueeze(1)
        return torch.cat([t, act], dim=1)

    def _loss(self, pos_t, deltas, labels, active_t, n_parts):
        """L_conn + L_time + L_move (paper; see module docstring)."""
        p = self.p
        # soft component count from the soft Laplacian spectrum
        dist = torch.cdist(pos_t, pos_t)
        a = torch.sigmoid((p.r_c - dist) / 5.0)
        a = a * (1.0 - torch.eye(len(a), device=pos_t.device))
        a = a * active_t[:, None] * active_t[None, :]
        deg = a.sum(1)
        lap = torch.diag(deg) - a
        lap = 0.5 * (lap + lap.T) + 1e-6 * torch.eye(len(a), device=pos_t.device)
        evals = torch.linalg.eigvalsh(lap)
        c_soft = torch.sigmoid((1e-2 - evals) / 5e-4).sum()  # ~#components
        l_conn = c_soft - 1.0
        # per-part translation magnitudes (failed-part delta excluded by mask)
        mag = deltas.norm(dim=1)[:n_parts]
        l_time = mag.max() / p.v_max
        l_move = mag.mean()
        # keep predicted positions in the area
        over = torch.relu(pos_t - p.area).sum() + torch.relu(-pos_t).sum()
        return l_conn + l_time + l_move + 1e-3 * over

    # ---- public API -------------------------------------------------------

    def train_step(self, pos: np.ndarray, active: np.ndarray, epochs: int = 5,
                   adj_override: np.ndarray | None = None) -> float:
        self.model.train()
        dev = self.device
        pos_t = torch.tensor(pos, dtype=torch.float32, device=dev)
        act_t = torch.tensor(active, dtype=torch.float32, device=dev)
        labels = self._parts(pos, active)
        n_parts = len(set(labels[active]))
        lnorm = _normalized_laplacian(adj_override if adj_override is not None
                                      else self._multipartite(pos, active, labels)).to(dev)
        x = self._features(pos, active).to(dev)
        loss = None
        for _ in range(epochs):
            deltas = self.model(x, lnorm, torch.tensor(labels, device=dev), n_parts)
            part_of = torch.tensor(
                [labels[i] if labels[i] < n_parts else 0 for i in range(len(pos))],
                device=dev)
            full = deltas[part_of]
            loss = self._loss(pos_t + full * self.p.v_max * self.p.T_ist,
                              deltas, labels, act_t, n_parts)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return float(loss.item())

    def translation(self, pos: np.ndarray, active: np.ndarray,
                    adj_override: np.ndarray | None = None) -> np.ndarray:
        """Per-node unit translation directions (one delta per subnetwork)."""
        self.model.eval()
        dev = self.device
        labels = self._parts(pos, active)
        n_parts = len(set(labels[active]))
        lnorm = _normalized_laplacian(adj_override if adj_override is not None
                                      else self._multipartite(pos, active, labels)).to(dev)
        with torch.no_grad():
            deltas = self.model(self._features(pos, active).to(dev), lnorm,
                                torch.tensor(labels, device=dev), n_parts)
        out = np.zeros((len(pos), 2), dtype=float)
        for i in range(len(pos)):
            k = labels[i]
            if k < n_parts:
                out[i] = deltas[k].cpu().numpy()
        return out
