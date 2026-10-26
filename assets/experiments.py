"""Estimate projection points for power_diagram.html.

Edit `estimate_projections` below, then tick "Estimated projection points" in the
page. The page talks to `python3 assets/estimate_server.py`, which reloads this
file on every request, so edits take effect without restarting the server.

Coordinates are the normalized [-1, 1]^2 frame with y pointing up (as in
rbf-your-sdf-demo).
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull

# Filters applied after estimation (normalized units)
ERROR_THRESHOLD = 1e-8   # drop a projection whose RANSAC error exceeds this
FEASIBILITY_TOL = 1e-3   # inside circle j means |q - x_j| < r_j - tol; slack for curved boundaries


def regular_triangulation_neighbors(xy, r):
    """Neighbors of each weighted point (center xy[i], radius r[i]) in the regular
    triangulation: the lower convex hull of the lifted points (x, y, |x|^2 - r^2).
    Hidden points (empty power cells) get no neighbors."""
    lifted = np.c_[xy, (xy ** 2).sum(1) - r ** 2]
    hull = ConvexHull(lifted)
    lower = hull.equations[:, 2] < -1e-12   # facets whose outward normal points down
    nbrs = [set() for _ in range(len(xy))]
    for tri in hull.simplices[lower]:
        for a, b in itertools.combinations(tri, 2):
            nbrs[a].add(b)
            nbrs[b].add(a)
    return [np.array(sorted(n), dtype=int) for n in nbrs]


def covered_by_other_circle(q, xy, r, tol=None):
    """(N,) bool: q[i] lies inside some circle j != i by more than tol."""
    if tol is None:
        tol = FEASIBILITY_TOL
    d = np.sqrt(((q[:, None, :] - xy[None]) ** 2).sum(2))   # (N, N)
    inside = d < r[None] - max(tol, 1e-12)
    np.fill_diagonal(inside, False)
    return inside.any(1)


def estimate_projections(xy, sdf, loops, inherit=True):
    """Estimated projection point for each sample.

    xy: (N, 2) sample positions, sdf: (N,) signed distances (negative inside),
    loops: GT polylines (unused here).
    inherit: let samples without a usable gradient inherit one (page toggle).
    Returns (N, 2); a row of NaNs drops that sample's projection point.

    RANSAC-style: the gradient at each sample is fit from pairs of its regular
    triangulation neighbors (see estimate_gradient_exhaustive), or inherited from
    a sample whose best pair included it (see estimate_gradients_RANSAC), and the
    projection is x - sdf * grad. A projection is dropped when its error exceeds
    ERROR_THRESHOLD, or when it falls strictly inside another sample's circle
    (an invisible region, so it cannot be a boundary point).
    """
    r = np.abs(sdf)
    neighbors = regular_triangulation_neighbors(xy, r)
    grads, errors, inherited = estimate_gradients_RANSAC(xy, sdf, neighbors, inherit)
    proj = xy - sdf[:, None] * grads

    no_grad = np.linalg.norm(grads, axis=1) < 0.5       # hidden or too few neighbors
    high_err = ~no_grad & (errors > ERROR_THRESHOLD)
    infeasible = ~no_grad & ~high_err & covered_by_other_circle(proj, xy, r)
    proj[no_grad | high_err | infeasible] = np.nan
    print(f"  kept {np.isfinite(proj[:, 0]).sum()}/{len(xy)} "
          f"({(inherited & np.isfinite(proj[:, 0])).sum()} with an inherited gradient); dropped: "
          f"{no_grad.sum()} no gradient, {high_err.sum()} error > {ERROR_THRESHOLD}, "
          f"{infeasible.sum()} inside another circle")
    return proj



def estimate_gradients_RANSAC(points, sdf_values, neighbors, inherit=True):
    """Robust gradient at each sample from its neighbors' SDF values.
    Inputs:
        points: (N, 2) coordinates of sample points
        sdf_values: (N,) corresponding SDF values
        neighbors: per-point neighbor index arrays (regular triangulation neighbors)
    Returns:
        gradients: (N, 2) unit gradients (zero when no gradient could be fit)
        errors: (N,) mean absolute residual of each gradient on the neighbor
            pair it was solved from (inf when no gradient could be fit)
        inherited: (N,) bool, gradient taken from another sample (see below)

    The gradient a sample c solves from its neighbor pair (n_i, n_j) fits the
    linear model through all three points, so it is also a gradient for n_i and
    n_j. Neighborhoods are not transitive, so n_i may not find a usable gradient
    from its own neighbors; such samples inherit the lowest-error gradient
    among those solved by samples that used them in their pair.
    """
    n = points.shape[0]
    gradients = np.zeros_like(points)
    errors = np.zeros(n)
    donor_grad = np.zeros_like(points)   # best gradient handed to each sample
    donor_err = np.full(n, np.inf)
    for c in range(n):
        gradients[c], errors[c], pair = estimate_gradient_exhaustive(points, sdf_values, c, neighbors[c])
        if pair is None:
            continue
        for m in pair:
            if errors[c] < donor_err[m]:
                donor_err[m] = errors[c]
                donor_grad[m] = gradients[c]

    usable = (np.linalg.norm(gradients, axis=1) >= 0.5) & (errors <= ERROR_THRESHOLD)
    inherited = ~usable & (donor_err <= ERROR_THRESHOLD) & inherit
    gradients[inherited] = donor_grad[inherited]
    errors[inherited] = donor_err[inherited]
    return gradients, errors, inherited


def estimate_gradient_exhaustive(points, sdf_values, ind, indices):
    """
    Brute-force all pairs of neighbors to find the unit gradient that minimizes
    the mean absolute residual on its own pair.
    Returns (gradient, loss, pair): pair holds the indices of the two neighbors
    that produced the gradient, or None when it did not come from a pair.
    """
    # Local coordinates relative to the center point c = `ind`: the linear model
    # sdf(n) - sdf(c) = g . (x_n - x_c) passes through (0, 0). Stacked over the
    # neighbors n_1..n_k it reads A_local @ g = b_local:
    #
    #   [ x_n1 - x_c   y_n1 - y_c ]             [ sdf(n_1) - sdf(c) ]
    #   [ x_n2 - x_c   y_n2 - y_c ]   [ g_x ]   [ sdf(n_2) - sdf(c) ]
    #   [    ...          ...     ] @ [ g_y ] = [        ...        ]
    #   [ x_nk - x_c   y_nk - y_c ]             [ sdf(n_k) - sdf(c) ]
    A_local = points[indices] - points[ind]
    b_local = sdf_values[indices] - sdf_values[ind]
    n_neighbors = len(indices)

    if n_neighbors == 1:
        # Only one neighbor: the best we can do is the direction to it,
        # flipped if the neighbor has the smaller SDF
        grad = A_local[0] / np.linalg.norm(A_local[0])
        if b_local[0] < 0:
            grad *= -1.0
        return grad, np.abs(A_local[0] @ grad - b_local[0]), None
    if n_neighbors < 2:
        return np.zeros(2), np.inf, None

    best_loss = np.inf
    best_gradient = np.zeros(2)
    best_pair = None
    for i, j in itertools.combinations(range(n_neighbors), 2):
        # Candidate from the 2x2 subsystem of neighbors n_i, n_j:
        #   [ x_ni - x_c   y_ni - y_c ]   [ g_x ]   [ sdf(n_i) - sdf(c) ]
        #   [ x_nj - x_c   y_nj - y_c ] @ [ g_y ] = [ sdf(n_j) - sdf(c) ]
        A_sub = A_local[[i, j]]
        b_sub = b_local[[i, j]]
        det = A_sub[0, 0] * A_sub[1, 1] - A_sub[0, 1] * A_sub[1, 0]
        if abs(det) < 1e-6:
            # Collinear: fall back to a 1D gradient g = slope * d along the unit
            # direction d, with slope the least-squares solution of
            #   [ (x_ni - x_c) . d ]           [ sdf(n_i) - sdf(c) ]
            #   [ (x_nj - x_c) . d ] * slope = [ sdf(n_j) - sdf(c) ]
            dir_norm = np.linalg.norm(A_sub[0])
            if dir_norm < 1e-10:
                dir_norm = np.linalg.norm(A_sub[1])
                if dir_norm < 1e-10:
                    continue  # Both vectors are zero (overlapping points)
                direction = A_sub[1] / dir_norm
            else:
                direction = A_sub[0] / dir_norm
            projections = A_sub @ direction
            slope = np.dot(projections, b_sub) / np.dot(projections, projections)
            cand_grad = slope * direction
        else:
            cand_grad = np.linalg.solve(A_sub, b_sub)
        # SDF property: |grad| ~ 1; loose tolerance for imperfect but reasonable fits
        norm = np.linalg.norm(cand_grad)
        if not (0.5 < norm < 1.5):
            continue
        cand_grad /= norm
        # Score on the pair itself: mean of |A_sub @ g - b_sub| (2,) after g
        # was normalized to unit length
        loss = np.mean(np.abs(A_sub @ cand_grad - b_sub))
        if loss < best_loss:
            best_loss = loss
            best_gradient = cand_grad
            best_pair = (indices[i], indices[j])
    return best_gradient, best_loss, best_pair
