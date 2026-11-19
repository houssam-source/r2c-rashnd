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

Device support (environment variables, no API change):
- R2C_IST_DEVICE: "auto" (default; cuda when available) | "cuda" | "cpu".
- R2C_IST_TF32=1: allow TF32 matmul on Ampere+ GPUs (faster, slightly
  lower precision — off by default so results match CPU runs bit-for-bit).
"""
import os

import numpy as np
import scipy.spatial
import scipy.sparse.csgraph
import torch
import torch.nn as nn

from .config import R2CParams


def _pick_device() -> torch.device:
    override = os.environ.get("R2C_IST_DEVICE", "auto").lower()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("R2C_IST_DEVICE=cuda but no CUDA GPU is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_enable_tf32(dev: torch.device):
    if dev.type == "cuda" and os.environ.get("R2C_IST_TF32", "0") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True


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
    """8 first-order layers, d=3 -> 512; per-subnetwork tanh translation.
    Dropout (HPO knob) sits before the MLP head; default 0.0 reproduces the
    frozen baseline architecture exactly."""

    def __init__(self, p: R2CParams, in_dim: int = 3):
        super().__init__()
        h = p.gcn_hidden
        dims = [in_dim] + [h] * p.gcn_layers
        self.convs = nn.ModuleList(
            FirstOrderChebLayer(dims[i], dims[i + 1])
            for i in range(p.gcn_layers))
        self.dropout = float(getattr(p, "gcn_dropout", 0.0))
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
            if self.dropout > 0.0:
                pooled = torch.nn.functional.dropout(
                    pooled, p=self.dropout, training=self.training)
            deltas.append(self.head(pooled))
        return torch.stack(deltas)                # (n_parts, 2) in (-1,1)^2


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


class IST:
    def __init__(self, p: R2CParams):
        self.p = p
        # seed torch so the GCN init (and hence every metric) is reproducible
        torch.manual_seed(p.seed)
        self.device = _pick_device()
        _maybe_enable_tf32(self.device)
        # HPO overrides (training-loop knobs; absent env -> frozen defaults)
        p.lr = _env_float("R2C_IST_LR", p.lr)
        p.gcn_hidden = int(_env_float("R2C_IST_HIDDEN", p.gcn_hidden))
        p.gcn_layers = int(_env_float("R2C_IST_LAYERS", p.gcn_layers))
        self.model = ChebNet(p).to(self.device)
        init_path = os.environ.get("R2C_IST_INIT_WEIGHTS")
        if init_path:
            self.model.load_state_dict(torch.load(
                init_path, map_location=self.device))
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=p.lr,
            weight_decay=_env_float("R2C_IST_WD", 0.0))
        self.patience = int(_env_float("R2C_IST_PATIENCE", 20))
        self.val_frac = _env_float("R2C_IST_VAL_FRAC", 0.0)
        self.max_epochs = int(_env_float("R2C_IST_MAX_EPOCHS", 5))
        self.top_m = getattr(p, "top_m", 3)
        # graph cache: train_step() and the translation() call that always
        # follows it in the controllers use identical (pos, active[, adj]).
        self._graph_key = None
        self._graph = None

    def _graph_parts(self, pos: np.ndarray, active: np.ndarray,
                     adj_override: np.ndarray | None):
        if adj_override is not None:
            key = (pos.tobytes(), active.tobytes(), adj_override.tobytes())
        else:
            key = (pos.tobytes(), active.tobytes())
        if self._graph_key != key:
            labels = self._parts(pos, active)
            adj = adj_override if adj_override is not None \
                else self._multipartite(pos, active, labels)
            lnorm = _normalized_laplacian(adj)
            self._graph = (labels, lnorm)
            self._graph_key = key
        return self._graph

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

    def _split_parts(self, labels: np.ndarray, active: np.ndarray,
                     n_parts: int):
        """Deterministic train/val part split for early stopping
        (val_frac of parts, at least one val part when >= 2 parts)."""
        if self.val_frac <= 0.0 or n_parts < 2:
            return None
        n_val = max(1, int(round(self.val_frac * n_parts)))
        n_val = min(n_val, n_parts - 1)
        if n_val < 1:
            return None
        g = np.random.RandomState(self.p.seed + int(labels[active][0])
                                  + n_parts)
        perm = g.permutation(n_parts)
        return np.sort(perm[:n_val])      # val part ids

    def _masked_loss(self, pos_t, deltas, labels, active_t, n_parts,
                     train_parts, val_parts):
        """L_conn/L_time/L_move restricted to a part subset; positions of
        the other parts are held fixed (their deltas are not applied)."""
        p = self.p
        part_of = torch.tensor(
            [labels[i] if labels[i] < n_parts else 0 for i in range(len(labels))],
            device=pos_t.device)
        full = deltas[part_of]
        shift = torch.zeros_like(pos_t)
        keep = torch.zeros(len(labels), dtype=torch.bool, device=pos_t.device)
        for k in (val_parts if val_parts is not None else range(n_parts)):
            keep |= torch.tensor(labels == k, device=pos_t.device)
        if val_parts is not None:
            keep = ~keep                     # shift only NON-val parts
            shift[keep] = full[keep] * p.v_max * p.T_ist
        else:
            shift = full * p.v_max * p.T_ist
        pred = pos_t + shift
        dist = torch.cdist(pred, pred)
        a = torch.sigmoid((p.r_c - dist) / 5.0)
        a = a * (1.0 - torch.eye(len(a), device=pos_t.device))
        a = a * active_t[:, None] * active_t[None, :]
        deg = a.sum(1)
        lap = torch.diag(deg) - a
        lap = 0.5 * (lap + lap.T) + 1e-6 * torch.eye(len(a), device=pos_t.device)
        evals = torch.linalg.eigvalsh(lap)
        l_conn = torch.sigmoid((1e-2 - evals) / 5e-4).sum() - 1.0
        sel = torch.tensor(train_parts, device=pos_t.device)
        mag = deltas.norm(dim=1)[sel]
        l_time = mag.max() / p.v_max
        l_move = mag.mean()
        over = torch.relu(pred - p.area).sum() + torch.relu(-pred).sum()
        return l_conn + l_time + l_move + 1e-3 * over

    def train_step(self, pos: np.ndarray, active: np.ndarray, epochs: int = 5,
                   adj_override: np.ndarray | None = None) -> float:
        self.model.train()
        dev = self.device
        pos_t = torch.tensor(pos, dtype=torch.float32, device=dev)
        act_t = torch.tensor(active, dtype=torch.float32, device=dev)
        labels, lnorm = self._graph_parts(pos, active, adj_override)
        n_parts = len(set(labels[active]))
        if n_parts == 0:
            return 0.0   # total fleet loss: nothing to train
        lnorm = lnorm.to(dev)
        x = self._features(pos, active).to(dev)
        val_ids = self._split_parts(labels, active, n_parts)
        epochs = max(epochs, self.max_epochs) if self.max_epochs > 5             else epochs
        if val_ids is None:
            # frozen baseline path (bit-identical to the original loop)
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
        # HPO path: train/val part split, best-snapshot restore, early stop
        train_ids = np.array([k for k in range(n_parts)
                              if k not in set(val_ids.tolist())])
        best_state, best_val, since_best = None, float("inf"), 0
        labels_t = torch.tensor(labels, device=dev)
        for _ in range(epochs):
            deltas = self.model(x, lnorm, labels_t, n_parts)
            loss = self._masked_loss(pos_t, deltas, labels, act_t,
                                     n_parts, train_ids, val_ids)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.model.eval()
            with torch.no_grad():
                vd = self.model(x, lnorm, labels_t, n_parts)
                vloss = float(self._masked_loss(pos_t, vd, labels, act_t,
                                                n_parts, val_ids,
                                                None).item())
            self.model.train()
            if vloss < best_val - 1e-6:
                best_val, since_best = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return float(loss.item())

    def translation(self, pos: np.ndarray, active: np.ndarray,
                    adj_override: np.ndarray | None = None) -> np.ndarray:
        """Per-node unit translation directions (one delta per subnetwork)."""
        self.model.eval()
        dev = self.device
        labels, lnorm = self._graph_parts(pos, active, adj_override)
        n_parts = len(set(labels[active]))
        if n_parts == 0:
            return np.zeros((len(pos), 2), dtype=float)
        lnorm = lnorm.to(dev)
        with torch.inference_mode():
            deltas = self.model(self._features(pos, active).to(dev), lnorm,
                                torch.tensor(labels, device=dev), n_parts)
        # one device->host transfer for all parts (not one per node)
        d_np = deltas.cpu().numpy()
        out = np.zeros((len(pos), 2), dtype=float)
        for i in range(len(pos)):
            k = labels[i]
            if k < n_parts:
                out[i] = d_np[k]
        return out
