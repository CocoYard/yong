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


def estimate_projections(xy, sdf, loops, inherit=True, refine=False, turning=0.0, turning_only=False,
                         turning_h=0.1, rope=False, rope_flat=10.0):
    """Estimated projection point for each sample.

    xy: (N, 2) sample positions, sdf: (N,) signed distances (negative inside),
    loops: GT polylines (unused here).
    inherit: let samples without a usable gradient inherit one (page toggle).
    refine: then move the points along the exposed arcs of their circles to
        minimize the power-contour length (page toggle, see refine_projections).
        Every circle with an exposed arc gets a point, including those the
        filters dropped (see refine_projections for their start).
    turning: weight of the contour turning-angle term in the refine objective
        (page input, see turning_and_grad).
    turning_only: drop the length from the refine objective and minimize the
        turning-angle term alone (page checkbox; turning is then ignored).
    rope: in the refine, use the rope length (a simplified contour through
        the projection points, see rope_length_and_grad) instead of the
        contour length (page checkbox).
    rope_flat: below this turn (degrees) two consecutive projection points
        are joined straight instead of through their tangents' corner.
    turning_h: scale h of the turning-angle term, as a fraction of the
        sample spacing (page input, see turning_and_grad and sample_spacing).
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
    grads, errors, inherited, ambiguous = estimate_gradients_RANSAC(xy, sdf, neighbors, inherit)
    proj = xy - sdf[:, None] * grads

    no_grad = np.linalg.norm(grads, axis=1) < 0.5       # hidden, too few neighbors, or ambiguous
    high_err = ~no_grad & (errors > ERROR_THRESHOLD)
    infeasible = ~no_grad & ~high_err & covered_by_other_circle(proj, xy, r)
    proj[no_grad | high_err | infeasible] = np.nan
    print(f"  kept {np.isfinite(proj[:, 0]).sum()}/{len(xy)} "
          f"({(inherited & np.isfinite(proj[:, 0])).sum()} with an inherited gradient, "
          f"{(inherited & ambiguous & np.isfinite(proj[:, 0])).sum()} of them ambiguous); dropped: "
          f"{no_grad.sum()} no gradient ({(no_grad & ambiguous).sum()} ambiguous), "
          f"{high_err.sum()} error > {ERROR_THRESHOLD}, "
          f"{infeasible.sum()} inside another circle")
    if refine:
        proj = refine_projections(xy, sdf, proj, turning, turning_only, turning_h, rope, rope_flat)
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
        ambiguous: (N,) bool, several feasible perfect gradients (see
            estimate_gradient_exhaustive)

    The gradient a sample c solves from its neighbor pair (n_i, n_j) fits the
    linear model through all three points, so it is also a gradient for n_i and
    n_j. Neighborhoods are not transitive, so n_i may not find a usable gradient
    from its own neighbors, or may find several (ambiguous); such samples
    inherit the lowest-error gradient among those solved, unambiguously, by
    samples that used them in their pair.
    """
    n = points.shape[0]
    gradients = np.zeros_like(points)
    errors = np.zeros(n)
    donor_grad = np.zeros_like(points)   # best gradient handed to each sample
    donor_err = np.full(n, np.inf)
    ambiguous = np.zeros(n, bool)
    for c in range(n):
        gradients[c], errors[c], pair, ambiguous[c] = estimate_gradient_exhaustive(points, sdf_values, c, neighbors[c])
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
    return gradients, errors, inherited, ambiguous


def estimate_gradient_exhaustive(points, sdf_values, ind, indices):
    """
    Brute-force all pairs of neighbors to find the unit gradient that minimizes
    the mean absolute residual on its own pair.
    Returns (gradient, loss, pair, ambiguous): pair holds the indices of the
    two neighbors that produced the gradient, or None when it did not come
    from a pair.

    A pair solves its 2x2 system exactly, so every pair whose gradient has
    unit length scores ~0 ("perfect"): e.g. two pairs can give mirror-image
    unit gradients. Perfect gradients (loss within ERROR_THRESHOLD) whose
    projection x - sdf * g lands inside another circle are discarded (the
    feasibility filter applied later anyway). If the feasible perfect ones
    still point in more than one direction, the sample is ambiguous: it gets
    no gradient here (zero, inf, None, True) and may inherit one later (see
    estimate_gradients_RANSAC).
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
        return grad, np.abs(A_local[0] @ grad - b_local[0]), None, False
    if n_neighbors < 2:
        return np.zeros(2), np.inf, None, False

    cands = []                                    # (loss, gradient, pair)
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
        cands.append((loss, cand_grad, (indices[i], indices[j])))
    if not cands:
        return np.zeros(2), np.inf, None, False
    cands.sort(key=lambda c: c[0])
    radii = np.abs(sdf_values)
    feasible = []
    for loss, g, pair in cands:
        if loss > ERROR_THRESHOLD:
            break
        q = points[ind] - sdf_values[ind] * g
        inside = np.linalg.norm(points - q, axis=1) < radii - max(FEASIBILITY_TOL, 1e-12)
        inside[ind] = False
        if not inside.any():
            feasible.append((loss, g, pair))
    if not feasible:
        loss, g, pair = cands[0]
        return g, loss, pair, False
    if any(np.linalg.norm(g - feasible[0][1]) > 1e-6 for _, g, _ in feasible[1:]):
        return np.zeros(2), np.inf, None, True
    loss, g, pair = feasible[0]
    return g, loss, pair, False


# =============================================================================
# Iterative refinement: minimize the length of the power contour
# =============================================================================
# Adapted from the 2D prototype power_contour_star.py.
#
# - Each sample i gives a circle (c_i, r_i = |d_i|) with sign s_i, and one
#   projection point q_i with weight 0 and the OPPOSITE sign.
# - q_i lives on an exposed arc of circle i (the part not inside any other disk).
#   An exposed set can have several disjoint arcs; one is chosen per circle
#   (Problem's theta_ref).
# - Corner circles: the exposed set degenerates to a point -> q_i fixed there.
#   Fully covered circles get no q.
# - Sites of the power diagram: circles (weight r^2) + q's (weight 0) + 4 far
#   box points. Power contour = power-diagram edges between opposite signs.
# - Objective: total contour length. DOF: angle theta_i of each free q_i.
#
# Gradient: for fixed combinatorics, the power vertex v of triangle (p0,p1,p2)
# solves
#     A v = b,  A rows = 2 (p_k - p0)^T,  b_k = |p_k|^2 - |p0|^2 - w_k + w0  (k = 1,2)
# Implicit differentiation (weights of q are constant 0):
#     dv/dp0 = A^{-1} [2(v-p0)^T ; 2(v-p0)^T]
#     dv/dp1 = A^{-1} [2(p1-v)^T ; 0]
#     dv/dp2 = A^{-1} [0 ; 2(p2-v)^T]
# Then chain rule: edge length -> vertices -> q -> theta.

TWO_PI = 2 * np.pi

# Refinement settings
REFINE_ITERS = 400       # projected gradient descent iterations
LENGTH_WEIGHT = 1.0      # weight of the contour length in the refine objective
                         # (turning_only sets it to 0)


# ----------------------------------------------------------------------------- exposed arcs
def free_arcs(i, C, R, eps=1e-9):
    """Arcs of circle i not covered by any other open disk. Returns list of (lo, hi), hi may exceed 2pi."""
    ci, ri = C[i], R[i]
    cov = []
    for j in range(len(C)):
        if j == i:
            continue
        dv = C[j] - ci
        d = np.hypot(*dv)
        if d >= ri + R[j]:
            continue
        if d + ri <= R[j]:
            return []  # fully covered
        if d + R[j] <= ri:
            continue  # disk j strictly inside disk i, doesn't reach circle i
        cb = (ri ** 2 + d ** 2 - R[j] ** 2) / (2 * ri * d)
        b = np.arccos(np.clip(cb, -1, 1)) - eps
        if b <= 0:
            continue
        phi = np.arctan2(dv[1], dv[0])
        s = np.mod(phi - b, TWO_PI)
        e = s + 2 * b
        if e > TWO_PI:
            cov += [(s, TWO_PI), (0.0, e - TWO_PI)]
        else:
            cov.append((s, e))
    if not cov:
        return [(0.0, TWO_PI)]
    cov.sort()
    merged = [list(cov[0])]
    for s, e in cov[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    free = []
    prev = 0.0
    for s, e in merged:
        if s > prev:
            free.append([prev, s])
        prev = max(prev, e)
    if prev < TWO_PI:
        free.append([prev, TWO_PI])
    # merge wrap-around arc
    if len(free) >= 2 and free[0][0] == 0.0 and free[-1][1] == TWO_PI:
        last = free.pop()
        free[0] = [last[0], free[0][1] + TWO_PI]
    return [tuple(f) for f in free]


def angle_to_arc(th, lo, hi):
    """(angular distance, clamped angle) from th to the arc [lo, hi]."""
    best = (np.inf, lo)
    for t in (th, th + TWO_PI, th - TWO_PI):
        c = min(max(t, lo), hi)
        if abs(t - c) < best[0]:
            best = (abs(t - c), c)
    return best


# ----------------------------------------------------------------------------- power diagram
def regular_triangulation(X, W):
    """Lower-hull triangles of the lifted sites, and each one's power vertex read
    off its facet plane z = 2 v.x + c (well defined even for a flat triangle
    that Qt cut out of a coplanar facet)."""
    Z = (X ** 2).sum(1) - W
    hull = ConvexHull(np.c_[X, Z], qhull_options="Qt")
    lower = hull.equations[:, 2] < 0
    eq = hull.equations[lower]
    Vplane = -eq[:, :2] / (2 * eq[:, 2:3])
    return hull.simplices[lower], Vplane


def power_vertices(X, W, T, Vplane):
    """Power vertices and the 2x2 systems A v = b they solve; triangles with a
    (near) singular A keep the facet-plane vertex. ok marks the regular ones."""
    P = X[T]
    w = W[T]
    A = 2 * (P[:, 1:] - P[:, :1])
    b = (P[:, 1:] ** 2).sum(-1) - (P[:, :1] ** 2).sum(-1) - w[:, 1:] + w[:, :1]
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    scale = np.abs(A).reshape(len(A), -1).max(1) ** 2
    ok = np.abs(det) > 1e-12 * np.maximum(scale, 1e-300)
    v = Vplane.copy()
    v[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return v, A, ok


def contour_edges(T, labels):
    emap = {}
    for t, tri in enumerate(T):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            emap.setdefault(key, []).append(t)
    out = []
    for (a, b), ts in emap.items():
        if labels[a] != labels[b] and len(ts) == 2:
            out.append((a, b, ts[0], ts[1]))
    return out


def merge_contour(V, edges):
    """The contour as a graph: power vertices joined by a zero-length contour
    edge (several triangles sharing one vertex) merged into one vertex at
    their mean. Returns ts (the triangles on the contour), c (merged vertex of
    each), size (triangles per merged vertex), Vc (merged positions), pos
    (triangle -> merged vertex) and nbrs (neighbor sets of merged vertices)."""
    ts = sorted({t for _, _, t1, t2 in edges for t in (t1, t2)})
    parent = {t: t for t in ts}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for _, _, t1, t2 in edges:
        if np.hypot(*(V[t1] - V[t2])) < 1e-6:
            parent[find(t1)] = find(t2)
    roots = sorted({find(t) for t in ts})
    cid = {r: k for k, r in enumerate(roots)}
    c = np.array([cid[find(t)] for t in ts], int)     # cluster of each vertex in ts
    k = len(roots)
    size = np.bincount(c, minlength=k).astype(float)
    Vc = np.zeros((k, 2))
    np.add.at(Vc, c, V[ts])
    Vc /= size[:, None]
    pos = {t: c[n] for n, t in enumerate(ts)}
    nbrs = [set() for _ in range(k)]
    for _, _, t1, t2 in edges:
        a, b = pos[t1], pos[t2]
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    return ts, c, size, Vc, pos, nbrs


def turning_and_grad(V, edges, Qp, h=0.0):
    """Turning-angle energy of the contour polyline and its gradients dE/dV
    and dE/dQp (Qp: the projection points).

    theta_i = the angle between the edges a -> v_i and v_i -> b at each vertex
    v_i with exactly two contour neighbors a, b (0 = straight); only the
    directions of the edges count. Power vertices joined by a zero-length
    contour edge (several triangles sharing one vertex) are merged first into
    one vertex at their mean. Junctions and ends get no term. A corner at a
    projection point is a corner of the shape (the contour is normally flat
    through q, along the tangent line of q's circle, and turns at the ends of
    that segment), so each vertex's term is scaled by w = min(1, d / h), d its
    distance to the nearest projection point: 0 on a q, full from h away.
    (A hard cutoff, 0 within 1e-6 of a q and 1 beyond, was a step the
    optimizer would get stuck on; h <= 0 still uses it.)

    Without a scale (h <= 0), E = sum_i |theta_i|. That jumps by ~pi as soon
    as a q rotates off alignment and opens a tiny step whose two ends turn
    ~90 degrees each, however short the step. So an edge shorter than h is
    contracted with probability 1 - s, s = l / h, independently per edge, and
    E is the expected sum of |net turning| over the vertices that result:
    a run v_i..v_j merged into one vertex turns by theta_i + ... + theta_j, so
        E = sum over runs [i..j] of P(i..j) * |theta_i + ... + theta_j|,
        P(i..j) = s_{i-1} * s_j * prod_{i <= e < j} (1 - s_e),
    with s = 1 for edges of length >= h (always kept). A tiny step (net turn
    0) then costs ~0 and a corner split by a tiny edge still costs its net
    turn; at l = h it is the plain sum. Runs through a junction or end cost
    0, as that vertex would absorb them; a run's w is its members' smallest.

    With e1 = v - a, e2 = b - v, theta = atan2(e1 x e2, e1 . e2), and
    perp(u) = (-u_y, u_x):
        dtheta/da = perp(e1) / |e1|^2
        dtheta/db = perp(e2) / |e2|^2
        dtheta/dv = -dtheta/da - dtheta/db
    and for a short edge p - q, ds/dq = -ds/dp = (q - p) / (l h); where
    d < h, dw/dv = -dw/dq = (v - q) / (d h).
    """
    ts, c, size, Vc, pos, nbrs = merge_contour(V, edges)
    k = len(Vc)
    # weight from the distance to the nearest projection point
    if len(Qp):
        Dq = np.linalg.norm(Vc[:, None] - Qp[None], axis=2)
        jq = Dq.argmin(1)
        dq = Dq[np.arange(k), jq]
    else:
        jq, dq = np.zeros(k, int), np.full(k, np.inf)
    wq = np.minimum(1.0, dq / h) if h > 0 else (dq >= 1e-6).astype(float)

    # turning angle and its gradient at every vertex with two neighbors
    perp = lambda u: np.stack([-u[:, 1], u[:, 0]], 1)
    two = np.array([i for i in range(k) if len(nbrs[i]) == 2], int)
    theta = np.zeros(k)                                 # signed, walking a -> v -> b
    first = np.full(k, -1)                              # a of each vertex
    null = np.ones(k, bool)                             # no term of its own
    if len(two):
        a2, b2 = np.array([sorted(nbrs[i]) for i in two]).T
        e1, e2 = Vc[two] - Vc[a2], Vc[b2] - Vc[two]
        theta[two] = np.arctan2(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0], (e1 * e2).sum(1))
        da = perp(e1) / (e1 ** 2).sum(1)[:, None]
        db = perp(e2) / (e2 ** 2).sum(1)[:, None]
        null[two] = False
        first[two] = a2

    # short edges and the runs they join
    P = np.array(sorted({(min(i, j), max(i, j)) for i in range(k) for j in nbrs[i]}), int).reshape(-1, 2)
    ell = np.linalg.norm(Vc[P[:, 1]] - Vc[P[:, 0]], axis=1)
    sh = np.where(ell < h)[0] if h > 0 else np.zeros(0, int)
    s_of = {}                                           # (i, j) -> short edge index
    adj = [[] for _ in range(k)]
    for e in sh:
        i, j = P[e]
        s_of[i, j] = s_of[j, i] = e
        adj[i].append(j)
        adj[j].append(i)
    s = np.ones(len(P))
    s[sh] = ell[sh] / h

    E = 0.0
    gth = np.zeros(k)                                   # dE/dtheta
    gs = np.zeros(len(P))                               # dE/ds
    gw = np.zeros(k)                                    # dE/dw
    seen = np.zeros(k, bool)
    for v0 in range(k):
        if seen[v0]:
            continue
        # component of v0 under short edges; ordered if it is a path
        comp, stack = [], [v0]
        seen[v0] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        n_e = sum(len(adj[u]) for u in comp) // 2
        if len(comp) > 1 and n_e == len(comp) - 1 and max(len(adj[u]) for u in comp) <= 2:
            u = next(u for u in comp if len(adj[u]) == 1)
            path = [u]
            while len(path) < len(comp):
                u = next(w for w in adj[path[-1]] if len(path) < 2 or w != path[-2])
                path.append(u)
            runs = [path]
        else:
            runs = [[u] for u in comp]                  # single vertices (or an odd tangle)
        for path in runs:
            m = len(path)
            se = [s_of[path[t], path[t + 1]] for t in range(m - 1)]
            # signs that turn each theta into the one walking along the path,
            # so a run's angles add up to its net turn
            sg = np.ones(m)
            if m > 1:
                for t, v in enumerate(path):
                    prev = path[t - 1] if t > 0 else next(iter(nbrs[v] - {path[1]}), -1)
                    sg[t] = 1.0 if first[v] == prev else -1.0
            th = theta[path] * sg
            for i in range(m):
                for j in range(i, m):
                    if null[path[i:j + 1]].any():
                        break                           # longer runs contain it too
                    keep = ([se[i - 1]] if i > 0 else []) + ([se[j]] if j < m - 1 else [])
                    cut = se[i:j]
                    base = (s[se[i - 1]] if i > 0 else 1.0) * np.prod(1 - s[cut])
                    if base < 1e-15:
                        break                           # only shrinks as j grows
                    Pr = base * (s[se[j]] if j < m - 1 else 1.0)
                    kw = path[i + int(np.argmin(wq[path[i:j + 1]]))]
                    w = wq[kw]
                    if w <= 0:
                        break                           # longer runs contain it too
                    S = th[i:j + 1].sum()
                    E += Pr * abs(S) * w
                    gth[path[i:j + 1]] += Pr * np.sign(S) * sg[i:j + 1] * w
                    gw[kw] += Pr * abs(S)
                    for e in keep:
                        gs[e] += abs(S) * Pr * w / s[e]
                    for e in cut:
                        gs[e] -= abs(S) * Pr * w / (1 - s[e])

    gVc = np.zeros((k, 2))
    if len(two):
        w = gth[two][:, None]
        np.add.at(gVc, a2, w * da)
        np.add.at(gVc, b2, w * db)
        np.add.at(gVc, two, -w * (da + db))
    if len(sh):
        u = (Vc[P[sh, 1]] - Vc[P[sh, 0]]) / ell[sh][:, None]
        w = (gs[sh] / h)[:, None]
        np.add.at(gVc, P[sh, 1], w * u)
        np.add.at(gVc, P[sh, 0], -w * u)
    gQ = np.zeros_like(Qp)
    if h > 0:
        on = (dq < h) & (dq > 1e-15) & (gw != 0)
        u = (Vc[on] - Qp[jq[on]]) / (dq[on] * h)[:, None] * gw[on][:, None]
        gVc[on] += u
        np.add.at(gQ, jq[on], -u)
    gV = np.zeros_like(V)
    gV[ts] = gVc[c] / size[c][:, None]
    return E, gV, gQ


def contour_length_and_grad(X, W, labels, lw=1.0, tw=0.0, h=0.0):
    """Contour length L, turning-angle energy E (see turning_and_grad) and the
    gradient of lw * L + tw * E w.r.t. all site positions (weights held fixed)."""
    T, Vplane = regular_triangulation(X, W)
    V, A, ok = power_vertices(X, W, T, Vplane)
    edges = contour_edges(T, labels)
    L = 0.0
    gV = np.zeros_like(V)
    for a, b, t1, t2 in edges:
        d = V[t1] - V[t2]
        l = np.hypot(*d)
        L += l
        if lw and l > 1e-14:
            u = lw * d / l
            gV[t1] += u
            gV[t2] -= u
    E = 0.0
    qsel = np.where(W == 0)[0]                          # projection points (and the far box)
    gQ = None
    if tw and edges:
        E, gE, gQ = turning_and_grad(V, edges, X[qsel], h)
        gV += tw * gE
    gX = power_vertex_backprop(X, T, V, A, ok, gV)
    if gQ is not None:
        gX[qsel] += tw * gQ
    return L, E, gX


def power_vertex_backprop(X, T, V, A, ok, gV):
    """Gradient w.r.t. the sites X from a gradient gV w.r.t. the power
    vertices (see the derivation above); singular (flat) triangles contribute
    none."""
    gX = np.zeros_like(X)
    active = np.where((np.abs(gV).sum(1) > 0) & ok)[0]
    if len(active):
        lam = np.linalg.solve(np.transpose(A[active], (0, 2, 1)), gV[active][..., None])[..., 0]
        for k, t in enumerate(active):
            p0, p1, p2 = X[T[t]]
            v = V[t]
            l0, l1 = lam[k]
            gX[T[t][0]] += 2 * (v - p0) * (l0 + l1)
            gX[T[t][1]] += 2 * (p1 - v) * l0
            gX[T[t][2]] += 2 * (p2 - v) * l1
    return gX


def contour_segments(X, W, labels):
    """Power-contour segments (K, 2, 2) of the sites."""
    T, Vplane = regular_triangulation(X, W)
    V, _, _ = power_vertices(X, W, T, Vplane)
    edges = contour_edges(T, labels)
    return np.array([(V[t1], V[t2]) for _, _, t1, t2 in edges]).reshape(-1, 2, 2)


def closest_on_segments(P, segs):
    """Closest point on any of the segments (K, 2, 2) to each row of P (N, 2)."""
    a, e = segs[:, 0], segs[:, 1] - segs[:, 0]
    t = ((P[:, None] - a[None]) * e[None]).sum(2) / np.maximum((e * e).sum(1), 1e-300)[None]
    q = a[None] + np.clip(t, 0, 1)[..., None] * e[None]          # (N, K, 2)
    k = ((P[:, None] - q) ** 2).sum(2).argmin(1)
    return q[np.arange(len(P)), k]


# ----------------------------------------------------------------------------- rope
# The rope: a simplified contour through the projection points. Walking each
# contour loop gives the order of the q's on it, and the contour's direction
# leaving / arriving at each q (the tangent line of its circle, or at a
# corner the contour edges on either side). Between two consecutive q's a, b:
#   - the straight segment ab when the tangents turn by less than the flat
#     angle, or one tangent points at the other point (within that angle);
#   - else the corner X where the two tangent lines meet, a -> X -> b;
#   - if that path cuts into a circle, or the tangents meet behind a or
#     beyond b, the shortest polyline from a to b through the contour's
#     vertices between them that cuts into no circle (the contour itself is
#     one, so there always is one).
# Gradient: segments through their endpoints; contour vertices are power
# vertices, chained back to the sites as in contour_length_and_grad; a corner
# piece has length lam + mu (a + lam ta = b - mu tb = X) and depends on the
# two angles only, differentiated by central differences, rotating each q
# together with its tangent.

def cuts_circle(p0, p1, C, R, eps=None):
    """Whether the segment p0 p1 cuts into any circle (C, R) by more than eps
    (default FEASIBILITY_TOL: slack for estimation error, as for q itself)."""
    eps = FEASIBILITY_TOL if eps is None else eps
    d = p1 - p0
    L2 = d @ d
    t = np.clip(((C - p0) @ d) / L2, 0, 1) if L2 > 1e-24 else np.zeros(len(C))
    return bool((R - np.linalg.norm(C - (p0 + t[:, None] * d), axis=1) > eps).any())


def tangent_corner(a, ta, b, tb):
    """(X, lam, mu) with X = a + lam ta = b - mu tb, or None if parallel."""
    det = ta[0] * tb[1] - ta[1] * tb[0]
    if abs(det) < 1e-12:
        return None
    d = b - a
    lam = (d[0] * tb[1] - d[1] * tb[0]) / det
    mu = (ta[0] * d[1] - ta[1] * d[0]) / det
    return a + lam * ta, lam, mu


def rope_piece(a, ta, b, tb, C, R, flat, tol):
    """("line",), ("corner", X), or None when the contour fallback is needed."""
    d = b - a
    l = np.hypot(*d)
    if l < tol:
        return ("line",)
    # flat turn, or one tangent (nearly) pointing at the other point
    cf = np.cos(flat)
    if ta @ tb >= cf or ta @ d >= cf * l or tb @ d >= cf * l:
        return None if cuts_circle(a, b, C, R) else ("line",)
    sol = tangent_corner(a, ta, b, tb)
    if sol is None or sol[1] < 0 or sol[2] < 0:
        return None
    X = sol[0]
    return None if cuts_circle(a, X, C, R) or cuts_circle(X, b, C, R) else ("corner", X)


def shortest_clear_path(pts, C, R):
    """Indices of the shortest polyline pts[0] -> pts[-1] through any of the
    points in between whose segments cut into no circle (Dijkstra);
    consecutive points (contour edges) are always allowed."""
    k = len(pts)
    dist = np.full(k, np.inf)
    dist[0] = 0.0
    prev = np.full(k, -1)
    done = np.zeros(k, bool)
    while True:
        i = int(np.argmin(np.where(done, np.inf, dist)))
        if done[i] or not np.isfinite(dist[i]):
            break
        done[i] = True
        if i == k - 1:
            break
        for j in range(k):
            if done[j] or j == i:
                continue
            dj = dist[i] + np.hypot(*(pts[j] - pts[i]))
            if dj < dist[j] and (abs(i - j) == 1 or not cuts_circle(pts[i], pts[j], C, R)):
                dist[j], prev[j] = dj, i
    path = [k - 1]
    while path[-1] > 0:
        path.append(prev[path[-1]])
    return path[::-1]


def rope_loops(Vc, nbrs):
    """Loops of the contour graph as (closed, merged vertex ids in walking
    order). Dead-end spurs are pruned first; a walk stuck at a junction keeps
    the cycle it closed there."""
    nbrs = [set(s) for s in nbrs]
    stack = [v for v in range(len(Vc)) if len(nbrs[v]) == 1]
    while stack:
        v = stack.pop()
        for w in list(nbrs[v]):
            nbrs[w].discard(v)
            if len(nbrs[w]) == 1:
                stack.append(w)
        nbrs[v].clear()
    used, loops = set(), []
    for u0 in range(len(Vc)):
        for w0 in sorted(nbrs[u0]):
            if (min(u0, w0), max(u0, w0)) in used:
                continue
            used.add((min(u0, w0), max(u0, w0)))
            seq, prev, cur = [u0], u0, w0
            while cur != u0:
                seq.append(cur)
                nxt = [w for w in nbrs[cur] if (min(cur, w), max(cur, w)) not in used]
                if not nxt:
                    break
                w = next((w for w in nxt if w != prev), nxt[0])
                used.add((min(cur, w), max(cur, w)))
                prev, cur = cur, w
            closed = cur == u0
            if not closed and cur in seq[:-2]:
                seq = seq[seq.index(cur):]
                closed = True
            loops.append((closed, seq))
    return loops


def place_on_loops(Q, Vc, loops, tol):
    """For each loop, its q's (rows of Q within tol of it) as sorted
    (parameter, q index); parameter i + t means t along edge i."""
    best = np.full(len(Q), np.inf)
    where = [None] * len(Q)
    for li, (closed, seq) in enumerate(loops):
        P = Vc[seq]
        A = P if closed else P[:-1]
        B = np.roll(P, -1, 0) if closed else P[1:]
        if not len(A):
            continue
        e = B - A
        t = np.clip(((Q[:, None] - A[None]) * e[None]).sum(2) / np.maximum((e * e).sum(1), 1e-300)[None], 0, 1)
        dist = np.linalg.norm(Q[:, None] - (A[None] + t[..., None] * e[None]), axis=2)
        j = dist.argmin(1)
        dj = dist[np.arange(len(Q)), j]
        for k in np.where(dj < best)[0]:
            best[k], where[k] = dj[k], (li, j[k] + t[k, j[k]])
    on = [[] for _ in loops]
    for k in np.where(best < tol)[0]:
        on[where[k][0]].append((where[k][1], k))
    return [sorted(o) for o in on]


def rope_length_and_grad(X, W, labels, n, qs, qc, flat_deg=10.0, tol=1e-6, with_paths=False):
    """Rope length and its gradient: (L, gq, gth) with gq (len(qs), 2) w.r.t.
    the q positions (sites qs, segments and contour vertices) and gth
    (len(qs),) w.r.t. their angles around their circles' centers qc (corner
    pieces); with_paths also the rope as polylines. The first n sites are the
    circles. A loop through fewer than two distinct q's, or not closed,
    keeps its contour polyline."""
    C, R = X[:n], np.sqrt(np.maximum(W[:n], 0))
    flat = np.radians(flat_deg)
    T, Vplane = regular_triangulation(X, W)
    V, A, ok = power_vertices(X, W, T, Vplane)
    edges = contour_edges(T, labels)
    L, gq, gth, paths = 0.0, np.zeros((len(qs), 2)), np.zeros(len(qs)), []
    if not edges:
        return (L, gq, gth, paths) if with_paths else (L, gq, gth)
    ts, c, size, Vc, _, nbrs = merge_contour(V, edges)
    gVc = np.zeros_like(Vc)
    gQ = np.zeros((len(qs), 2))
    loops = rope_loops(Vc, nbrs)
    Q = X[qs]

    def add_polyline(nodes):
        # nodes: ("q", k) / ("v", merged vertex) / ("x", point); straight segments
        nonlocal L
        pos = [Q[i] if kind == "q" else Vc[i] if kind == "v" else i for kind, i in nodes]
        for (k0, i0), (k1, i1), p0, p1 in zip(nodes, nodes[1:], pos, pos[1:]):
            d = p1 - p0
            l = np.hypot(*d)
            L += l
            if l < 1e-14:
                continue
            u = d / l
            for kind, i, sgn in ((k0, i0, -1), (k1, i1, 1)):
                if kind == "q":
                    gQ[i] += sgn * u
                elif kind == "v":
                    gVc[i] += sgn * u
        return pos

    def rot(p, ctr, h):
        cs, sn = np.cos(h), np.sin(h)
        d = p - ctr
        return ctr + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])

    for (closed, seq), on in zip(loops, place_on_loops(Q, Vc, loops, tol)):
        nP = len(seq)
        if nP < 2:
            continue
        distinct = len({tuple(np.round(Q[k] / tol)) for _, k in on})
        if not closed or distinct < 2:
            nodes = [("v", v) for v in seq] + ([("v", seq[0])] if closed else [])
            paths.append(add_polyline(nodes))
            continue
        P = Vc[seq]
        E = np.roll(P, -1, 0) - P
        El = np.linalg.norm(E, axis=1)

        def edge_dir(i, step):
            for s in range(nP):
                j = (i + step * s) % nP
                if El[j] > tol:
                    return E[j] / El[j]
            return E[i % nP] / max(El[i % nP], 1e-300)

        def dirs(p, k):
            """Contour directions arriving at / leaving q."""
            i = int(np.floor(p)) % nP
            if np.hypot(*(Q[k] - P[(i + 1) % nP])) < tol:      # at vertex i + 1
                return edge_dir(i, -1), edge_dir(i + 1, 1)
            if np.hypot(*(Q[k] - P[i])) < tol:                 # at vertex i
                return edge_dir(i - 1, -1), edge_dir(i, 1)
            d = E[i] / max(El[i], 1e-300)
            return d, d

        path = []
        for j, (pa, ka) in enumerate(on):
            pb, kb = on[(j + 1) % len(on)]
            a, b = Q[ka], Q[kb]
            ta, tb = dirs(pa, ka)[1], dirs(pb, kb)[0]
            piece = rope_piece(a, ta, b, tb, C, R, flat, tol)
            if piece is not None and piece[0] == "line":
                path += add_polyline([("q", ka), ("q", kb)])
            elif piece is not None:
                X_ = piece[1]
                path += [a, X_, b]
                lam, mu = tangent_corner(a, ta, b, tb)[1:]
                L += lam + mu
                # angles: rotate q with its tangent around its circle's center
                for k, which in ((ka, 0), (kb, 1)):
                    h = 1e-7
                    vals = []
                    for sg in (1, -1):
                        aa, tta, bb, ttb = a, ta, b, tb
                        if which == 0:
                            aa, tta = rot(a, qc[k], sg * h), rot(ta, np.zeros(2), sg * h)
                        else:
                            bb, ttb = rot(b, qc[k], sg * h), rot(tb, np.zeros(2), sg * h)
                        sol = tangent_corner(aa, tta, bb, ttb)
                        vals.append(sol[1] + sol[2] if sol is not None else lam + mu)
                    gth[k] += (vals[0] - vals[1]) / (2 * h)
            else:
                ia, ib = int(np.floor(pa)) % nP, int(np.floor(pb)) % nP
                if ib < ia or (ib == ia and pb < pa):
                    ib += nP
                nodes = [("q", ka)] + [("v", seq[i % nP]) for i in range(ia + 1, ib + 1)] + [("q", kb)]
                pts = np.array([Q[ka]] + [Vc[seq[i % nP]] for i in range(ia + 1, ib + 1)] + [Q[kb]])
                keep = shortest_clear_path(pts, C, R)
                path += add_polyline([nodes[i] for i in keep])
        paths.append(path)

    # contour vertices -> their power vertices -> the sites
    gV = np.zeros_like(V)
    gV[ts] = gVc[c] / size[c][:, None]
    gX = power_vertex_backprop(X, T, V, A, ok, gV)
    gq = gQ + gX[qs]
    return (L, gq, gth, paths) if with_paths else (L, gq, gth)


# ----------------------------------------------------------------------------- problem
class Problem:
    """Circles (C, R, signs S) with one weight-0 projection point per circle.

    theta_ref (N,) picks each circle's arc: the exposed arc containing it, else
    the nearest one; NaN -> the longest arc. self.theta_ref is then clamped into
    the chosen [lo, hi]. Circles without an exposed set get no q (has_q False).
    Corner circles (exposed set = a single point) keep q frozen there.
    """

    def __init__(self, C, R, S, theta_ref, box=20.0, lw=1.0, tw=0.0, h=0.0, rope=False, flat_deg=10.0):
        self.C, self.R, self.S = C, R, S
        self.lw, self.tw, self.h, self.rope, self.flat_deg = lw, tw, h, rope, flat_deg
        n = len(C)
        self.arcs = [free_arcs(i, C, R) if R[i] > 1e-12 else [] for i in range(n)]
        self.has_q = np.array([len(a) > 0 for a in self.arcs])
        self.lo = np.zeros(n); self.hi = np.zeros(n)
        self.theta_ref = np.zeros(n)
        for i in np.where(self.has_q)[0]:
            if np.isfinite(theta_ref[i]):
                _, t, (lo, hi) = min((*angle_to_arc(theta_ref[i], lo, hi), (lo, hi)) for lo, hi in self.arcs[i])
            else:
                lo, hi = max(self.arcs[i], key=lambda a: a[1] - a[0])
                t = 0.5 * (lo + hi)
            self.lo[i], self.hi[i], self.theta_ref[i] = lo, hi, t
        self.is_corner = self.has_q & ((self.hi - self.lo) < 1e-6)
        self.free = self.has_q & ~self.is_corner
        self.qidx = np.where(self.has_q)[0]

        # fixed sites: circles + box
        self.box = box * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)

    def q_of(self, theta):
        return self.C + self.R[:, None] * np.stack([np.cos(theta), np.sin(theta)], 1)

    def sites(self, theta):
        n, m = len(self.C), len(self.qidx)
        X = np.vstack([self.C, self.q_of(theta)[self.qidx], self.box])
        W = np.concatenate([self.R ** 2, np.zeros(m), np.zeros(4)])
        lab = np.concatenate([self.S, -self.S[self.qidx], np.ones(4, int)])
        return X, W, lab

    def eval(self, theta):
        """Objective lw * L + tw * E and its gradient; L is the contour length
        (see contour_length_and_grad) or, with rope, the rope length (see
        rope_length_and_grad)."""
        X, W, lab = self.sites(theta)
        n, m = len(self.C), len(self.qidx)
        gq = np.zeros((n, 2))
        gth = np.zeros(n)
        f = 0.0
        if not self.rope or self.tw:
            lw = 0.0 if self.rope else self.lw
            L, E, gX = contour_length_and_grad(X, W, lab, lw, self.tw, self.h)
            gq[self.qidx] = gX[n:n + m]
            f = lw * L + self.tw * E
        if self.rope and self.lw:
            Lr, gr, gtr = rope_length_and_grad(X, W, lab, n, np.arange(n, n + m), self.C[self.qidx], self.flat_deg)
            gq[self.qidx] += self.lw * gr
            gth[self.qidx] += self.lw * gtr
            f += self.lw * Lr
        dq = self.R[:, None] * np.stack([-np.sin(theta), np.cos(theta)], 1)
        gth += (gq * dq).sum(1)
        gth[~self.free] = 0.0
        return f, gth

    def length_and_turning(self, theta):
        X, W, lab = self.sites(theta)
        L, E, _ = contour_length_and_grad(X, W, lab, 0.0, 1.0, self.h)
        return L, E

    def rope_length(self, theta):
        X, W, lab = self.sites(theta)
        n, m = len(self.C), len(self.qidx)
        return rope_length_and_grad(X, W, lab, n, np.arange(n, n + m), self.C[self.qidx], self.flat_deg)[0]

    def project(self, theta):
        return np.clip(theta, self.lo, self.hi)


def optimize(prob, theta0, iters=400, tol=1e-12):
    """Projected gradient descent with Armijo backtracking."""
    th = prob.project(theta0.copy())
    L, g = prob.eval(th)
    hist = [L]
    step = 0.2 / max(np.abs(g).max(), 1e-12)
    for it in range(iters):
        accepted = False
        for _ in range(40):
            th_new = prob.project(th - step * g)
            L_new, g_new = prob.eval(th_new)
            if L_new < L - 1e-4 * (g * (th - th_new)).sum() and L_new < L:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        dL = L - L_new
        th, L, g = th_new, L_new, g_new
        hist.append(L)
        step *= 1.5
        if dL < tol:
            break
    return th, np.array(hist)


def sample_spacing(xy):
    """Median distance from each sample to its nearest other sample (the grid
    step 2 / (n - 1) for the page's grid)."""
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def refine_projections(xy, sdf, proj0, turning=0.0, turning_only=False, turning_h=0.1, rope=False,
                       rope_flat=10.0):
    """Move each projection point along the exposed arcs of its circle to
    minimize LENGTH_WEIGHT * the power-contour length (or, with rope, the
    rope length) + turning * its
    turning-angle energy (scale h = turning_h * sample_spacing(xy)), or with
    turning_only the turning-angle energy alone, by projected gradient
    descent starting from proj0 (N, 2).
    NaN rows (dropped estimates) start from their circle center's projection
    onto the power contour of all circles plus the kept points; each start
    angle also picks the exposed arc (or, for a corner, the exposed point).
    Corner circles stay fixed. Returns (N, 2); NaN for circles without an
    exposed arc."""
    R = np.abs(sdf)
    S = np.where(sdf < 0, -1, 1)
    kept = np.isfinite(proj0[:, 0])
    missing = ~kept & (R > 1e-12)
    if missing.any():
        box = 20.0 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
        X = np.vstack([xy, proj0[kept], box])
        W = np.concatenate([R ** 2, np.zeros(kept.sum() + 4)])
        lab = np.concatenate([S, -S[kept], np.ones(4, int)])
        segs = contour_segments(X, W, lab)
        if len(segs):
            proj0 = proj0.copy()
            proj0[missing] = closest_on_segments(xy[missing], segs)
    d = proj0 - xy
    theta_ref = np.arctan2(d[:, 1], d[:, 0])          # NaN where proj0 is NaN
    lw, tw = (0.0, 1.0) if turning_only else (LENGTH_WEIGHT, turning)
    p = Problem(xy, R, S, theta_ref, lw=lw, tw=tw, h=turning_h * sample_spacing(xy), rope=rope,
                flat_deg=rope_flat)
    th = p.theta_ref.copy()
    L0, E0 = p.length_and_turning(th)
    R0 = p.rope_length(th)
    th, hist = optimize(p, th, iters=REFINE_ITERS)
    q = p.q_of(th)
    q[~p.has_q] = np.nan
    L1, E1 = p.length_and_turning(th)
    R1 = p.rope_length(th)
    print(f"  refine{' (rope)' if rope else ''}: contour length {L0:.4f} -> {L1:.4f}, rope length {R0:.4f} -> {R1:.4f}, "
          f"turning {E0:.4f} -> {E1:.4f} "
          f"(weights: length {lw:g}, turning {tw:g}, h = {turning_h:g} x spacing; {len(hist) - 1} iters); "
          f"{p.free.sum()} free, {p.is_corner.sum()} fixed corners, {(~p.has_q).sum()} without exposed arc")
    return q
