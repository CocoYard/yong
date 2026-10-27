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
                         turning_h=0.1):
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
    if refine:
        proj = refine_projections(xy, sdf, proj, turning, turning_only, turning_h)
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


def turning_and_grad(V, edges, at_q, h=0.0):
    """Turning-angle energy of the contour polyline and dE/dV.

    theta_i = the angle between the edges a -> v_i and v_i -> b at each vertex
    v_i with exactly two contour neighbors a, b (0 = straight); only the
    directions of the edges count. Power vertices joined by a zero-length
    contour edge (several triangles sharing one vertex) are merged first into
    one vertex at their mean. Junctions, ends and vertices at a projection
    point (at_q, per triangle) get no term: the contour is normally flat
    through q (the tangent line of q's circle) and turns at the ends of that
    segment, so a corner at q itself is a corner of the shape.

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
    turn; at l = h it is the plain sum. Runs through a vertex without a term
    cost 0, as that vertex would absorb them.

    With e1 = v - a, e2 = b - v, theta = atan2(e1 x e2, e1 . e2), and
    perp(u) = (-u_y, u_x):
        dtheta/da = perp(e1) / |e1|^2
        dtheta/db = perp(e2) / |e2|^2
        dtheta/dv = -dtheta/da - dtheta/db
    and for a short edge p - q, ds/dq = -ds/dp = (q - p) / (l h).
    """
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
    c = np.array([cid[find(t)] for t in ts])          # cluster of each vertex in ts
    k = len(roots)
    size = np.bincount(c, minlength=k).astype(float)
    Vc = np.zeros((k, 2))
    np.add.at(Vc, c, V[ts])
    Vc /= size[:, None]

    # neighbors of each merged vertex along the contour
    pos = {t: c[n] for n, t in enumerate(ts)}
    nbrs = [set() for _ in range(k)]
    for _, _, t1, t2 in edges:
        a, b = pos[t1], pos[t2]
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    corner = np.zeros(k, bool)
    corner[c[at_q[ts]]] = True

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
        null[two] = corner[two]
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
                    S = th[i:j + 1].sum()
                    E += Pr * abs(S)
                    gth[path[i:j + 1]] += Pr * np.sign(S) * sg[i:j + 1]
                    for e in keep:
                        gs[e] += abs(S) * Pr / s[e]
                    for e in cut:
                        gs[e] -= abs(S) * Pr / (1 - s[e])

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
    gV = np.zeros_like(V)
    gV[ts] = gVc[c] / size[c][:, None]
    return E, gV


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
    if tw and edges:
        # power vertices on a projection point (a weight-0 site of their triangle)
        q_site = W[T] == 0
        at_q = (q_site & (np.linalg.norm(X[T] - V[:, None], axis=2) < 1e-6)).any(1)
        E, gE = turning_and_grad(V, edges, at_q, h)
        gV += tw * gE
    gX = np.zeros_like(X)
    # singular triangles (flat) contribute no gradient
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
    return L, E, gX


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


# ----------------------------------------------------------------------------- problem
class Problem:
    """Circles (C, R, signs S) with one weight-0 projection point per circle.

    theta_ref (N,) picks each circle's arc: the exposed arc containing it, else
    the nearest one; NaN -> the longest arc. self.theta_ref is then clamped into
    the chosen [lo, hi]. Circles without an exposed set get no q (has_q False).
    Corner circles (exposed set = a single point) keep q frozen there.
    """

    def __init__(self, C, R, S, theta_ref, box=20.0, lw=1.0, tw=0.0, h=0.0):
        self.C, self.R, self.S = C, R, S
        self.lw, self.tw, self.h = lw, tw, h
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
        """Objective lw * L + tw * E (see contour_length_and_grad) and its gradient."""
        X, W, lab = self.sites(theta)
        L, E, gX = contour_length_and_grad(X, W, lab, self.lw, self.tw, self.h)
        n, m = len(self.C), len(self.qidx)
        gq = np.zeros((n, 2))
        gq[self.qidx] = gX[n:n + m]
        dq = self.R[:, None] * np.stack([-np.sin(theta), np.cos(theta)], 1)
        gth = (gq * dq).sum(1)
        gth[~self.free] = 0.0
        return self.lw * L + self.tw * E, gth

    def length_and_turning(self, theta):
        X, W, lab = self.sites(theta)
        L, E, _ = contour_length_and_grad(X, W, lab, 0.0, 1.0, self.h)
        return L, E

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


def refine_projections(xy, sdf, proj0, turning=0.0, turning_only=False, turning_h=0.1):
    """Move each projection point along the exposed arcs of its circle to
    minimize LENGTH_WEIGHT * the power-contour length + turning * its
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
    p = Problem(xy, R, S, theta_ref, lw=lw, tw=tw, h=turning_h * sample_spacing(xy))
    th = p.theta_ref.copy()
    L0, E0 = p.length_and_turning(th)
    th, hist = optimize(p, th, iters=REFINE_ITERS)
    q = p.q_of(th)
    q[~p.has_q] = np.nan
    L1, E1 = p.length_and_turning(th)
    print(f"  refine: contour length {L0:.4f} -> {L1:.4f}, turning {E0:.4f} -> {E1:.4f} "
          f"(weights: length {lw:g}, turning {tw:g}, h = {turning_h:g} x spacing; {len(hist) - 1} iters); "
          f"{p.free.sum()} free, {p.is_corner.sum()} fixed corners, {(~p.has_q).sum()} without exposed arc")
    return q
