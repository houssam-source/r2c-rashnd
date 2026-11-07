"""Visibility/terrain layer: camera-construct proxy for the shared map.

Drones' cameras + SLAM would build a fused height/occupancy map of the area;
here the (already generated) ruins grid stands in for that shared construct.
With equal mission altitudes, link (i,j) exists iff geometric range AND the
line of sight clears the terrain skyline.

Terrain is attached to RFLayer (affects link truth: blocked -> q = 0) and to
the hybrid controller variant 'v4' (clustering, corridors and bridging run on
the visibility graph, so knowledge sharing follows open corridors).
"""
import numpy as np


def los_cell_matrix(hmap: np.ndarray, area: float, altitude: float,
                    samples: int = 12) -> np.ndarray:
    """V[a, b] = True if the LOS between cell centers a and b clears the
    terrain at the given (equal) altitude. Precomputed once per map."""
    cells = hmap.shape[0]
    step = area / cells
    xs = (np.arange(cells) + 0.5) * step
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)   # (cells^2, 2)
    n = len(centers)
    V = np.ones((n, n), dtype=bool)
    for a in range(n):
        d = centers - centers[a]
        dist = np.linalg.norm(d, axis=1)
        for b in np.where((dist > 0) & (dist < area))[0]:
            ts = np.linspace(0, 1, samples)
            pts = centers[a] + ts[:, None] * (centers[b] - centers[a])
            idx = np.clip((pts / area * cells).astype(int), 0, cells - 1)
            h_los = hmap[idx[:, 0], idx[:, 1]]
            if (h_los >= altitude).any():
                V[a, b] = False
    return V


def vis_mask(pos: np.ndarray, V: np.ndarray, area: float) -> np.ndarray:
    """(n, n) boolean: True where the link clears terrain (and is in-map)."""
    cells = int(round(np.sqrt(V.shape[0])))
    idx = np.clip((pos / area * cells).astype(int), 0, cells - 1)
    c = idx[:, 0] * cells + idx[:, 1]
    return V[np.ix_(c, c)]
