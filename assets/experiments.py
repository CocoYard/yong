"""Estimate projection points for power_diagram.html.

Edit `estimate_projections` below, then click "Estimate projections" in the page.
The page talks to `python3 assets/estimate_server.py`, which reloads this file
on every click, so edits take effect without restarting the server.

Coordinates are the normalized [-1, 1]^2 frame with y pointing up (as in
rbf-your-sdf-demo).
"""

import numpy as np


def closest_points(loops, xy, chunk=256):
    """Closest point on the closed polylines `loops` to each row of xy (N, 2)."""
    if len(xy) > chunk:  # bound the (N, M, 2) temporaries
        return np.concatenate([closest_points(loops, xy[i:i + chunk]) for i in range(0, len(xy), chunk)])
    best_d2 = np.full(len(xy), np.inf)
    best = xy.copy()
    for L in loops:
        a = np.asarray(L, dtype=float)                    # (M, 2) segment starts
        e = np.roll(a, -1, axis=0) - a                    # (M, 2) segment vectors
        len2 = np.maximum((e * e).sum(1), 1e-300)
        # t for every (sample, segment) pair, clamped to the segment
        t = ((xy[:, None, :] - a[None]) * e[None]).sum(2) / len2[None]
        t = np.clip(t, 0.0, 1.0)
        q = a[None] + t[..., None] * e[None]              # (N, M, 2)
        d2 = ((xy[:, None, :] - q) ** 2).sum(2)
        k = d2.argmin(1)
        rows = np.arange(len(xy))
        better = d2[rows, k] < best_d2
        best_d2[better] = d2[rows, k][better]
        best[better] = q[rows, k][better]
    return best


def estimate_projections(xy, sdf, loops):
    """Estimated projection point for each sample.

    xy: (N, 2) sample positions, sdf: (N,) signed distances (negative inside),
    loops: GT polylines. Returns (N, 2); a row of NaNs keeps the GT projection.
    For now this is just the exact GT projection.
    """
    return closest_points(loops, xy)
