"""Contour simplification ("the rope"): a simplified power contour through
the projection points.

Public interface:
    simplify_contour(Vc, nbrs, Q, qc, C, R, ...) -> Rope
    Rope.paths()             polylines to draw or project onto
    Rope.points[i]           (k, 2) vertices of loop i, Rope.closed[i]
    Rope.backprop(grads)     gradient w.r.t. the vertices -> (gq, gth, gVc)

Walking each contour loop gives the order of the projection points q on it,
and the contour's direction leaving / arriving at each q (the tangent line of
its circle, or at a corner the contour edges on either side). Between two
consecutive q's a, b:
  - the straight segment ab when the tangents turn by less than the flat
    angle, or one tangent points at the other point (within that angle);
  - else the corner X where the two tangent lines meet, a -> X -> b;
  - if that path cuts into a circle, or the tangents meet behind a or beyond
    b, the shortest polyline from a to b through the contour's vertices
    between them that cuts into no circle (the contour itself is one, so
    there always is one).
A loop through fewer than two distinct q's, or not closed, keeps its contour
polyline.
"""

import numpy as np


def simplify_contour(Vc, nbrs, Q, qc, C, R, flat_deg=10.0, tol=1e-6, eps=1e-3):
    """The rope of a power contour.

    Vc (K, 2), nbrs: the contour as a graph (merged power vertices and their
        neighbor sets, see merge_contour in experiments.py).
    Q (M, 2): projection points; qc (M, 2): their circles' centers.
    C (N, 2), R (N,): the SDF circles the rope must not cut into (by more
        than eps).
    flat_deg: below this turn (degrees) two q's are joined straight.
    tol: distance under which a q counts as on the contour / on a vertex.
    """
    flat = np.radians(flat_deg)
    loops = contour_loops(Vc, nbrs)
    rope = Rope(Vc, Q, qc)
    for (closed, seq), on in zip(loops, place_on_loops(Q, Vc, loops, tol)):
        if len(seq) < 2:
            continue
        distinct = len({tuple(np.round(Q[k] / tol)) for _, k in on})
        if not closed or distinct < 2:
            rope.add(closed, [("v", v) for v in seq])
        else:
            rope.add(True, simplify_loop(Vc, seq, on, Q, C, R, flat, tol, eps))
    return rope


class Rope:
    """The rope as loops of vertices. Each vertex is a q ("q", k), a contour
    vertex ("v", merged vertex) or a corner ("x", a, ta, b, tb, ka, kb) where
    the tangent lines of q_ka and q_kb meet."""

    def __init__(self, Vc, Q, qc):
        self.Vc, self.Q, self.qc = Vc, Q, qc
        self.nodes, self.closed, self.points, self.jac = [], [], [], []

    def add(self, closed, nodes):
        P, J = [], []
        for nd in nodes:
            if nd[0] == "q":
                P.append(self.Q[nd[1]]); J.append(None)
            elif nd[0] == "v":
                P.append(self.Vc[nd[1]]); J.append(None)
            else:
                P.append(corner_point(*nd[1:5]))
                J.append(corner_jacobian(nd, self.qc))
        self.nodes.append(nodes)
        self.closed.append(closed)
        self.points.append(np.array(P).reshape(-1, 2))
        self.jac.append(J)

    def paths(self):
        """Each loop as a polyline (closed ones repeat their first point)."""
        return [np.vstack([P, P[:1]]) if cl else P for cl, P in zip(self.closed, self.points)]

    def backprop(self, grads):
        """Gradients w.r.t. each loop's vertices (list of (k, 2)) -> (gq, gth,
        gVc): w.r.t. the q positions, the q angles around their centers
        (through the corners) and the contour vertices."""
        gq = np.zeros_like(self.Q)
        gth = np.zeros(len(self.Q))
        gVc = np.zeros_like(self.Vc)
        for nodes, J_, G in zip(self.nodes, self.jac, grads):
            for nd, J, g in zip(nodes, J_, G):
                if nd[0] == "q":
                    gq[nd[1]] += g
                elif nd[0] == "v":
                    gVc[nd[1]] += g
                else:
                    ka, dA, kb, dB = J
                    gth[ka] += g @ dA
                    gth[kb] += g @ dB
        return gq, gth, gVc


# ----------------------------------------------------------------------------- one loop
def simplify_loop(Vc, seq, on, Q, C, R, flat, tol, eps):
    """Rope nodes of a closed loop (merged vertex ids seq) through its q's on
    = [(parameter, q index), ...] (see place_on_loops)."""
    nP = len(seq)
    P = Vc[seq]
    Ed = np.roll(P, -1, 0) - P
    El = np.linalg.norm(Ed, axis=1)
    U = Ed / np.maximum(El, 1e-300)[:, None]
    # the first edge longer than tol from each edge on, forward / backward (-1: none)
    fwd, bwd = np.full(nP, -1), np.full(nP, -1)
    j = -1
    for i in range(2 * nP - 1, -1, -1):
        if El[i % nP] > tol:
            j = i % nP
        if i < nP:
            fwd[i] = j
    j = -1
    for i in range(2 * nP):
        if El[i % nP] > tol:
            j = i % nP
        if i >= nP:
            bwd[i - nP] = j
    Pl, Ql = P.tolist(), Q.tolist()

    def edge_dir(i, step):
        i %= nP
        j = (fwd if step > 0 else bwd)[i]
        return U[j if j >= 0 else i]

    def dirs(p, k):
        """Contour directions arriving at / leaving q."""
        i = int(np.floor(p)) % nP
        (qx, qy), (ax, ay), (bx, by) = Ql[k], Pl[i], Pl[(i + 1) % nP]
        if np.hypot(qx - bx, qy - by) < tol:               # at vertex i + 1
            return edge_dir(i, -1), edge_dir(i + 1, 1)
        if np.hypot(qx - ax, qy - ay) < tol:               # at vertex i
            return edge_dir(i - 1, -1), edge_dir(i, 1)
        return U[i], U[i]

    nodes = []
    for j, (pa, ka) in enumerate(on):
        pb, kb = on[(j + 1) % len(on)]
        a, b = Q[ka], Q[kb]
        ta, tb = dirs(pa, ka)[1], dirs(pb, kb)[0]
        nodes.append(("q", ka))
        piece = rope_piece(a, ta, b, tb, C, R, flat, tol, eps)
        if piece is not None and piece[0] == "corner":
            nodes.append(("x", a, ta, b, tb, ka, kb))
        elif piece is None:
            ia, ib = int(np.floor(pa)) % nP, int(np.floor(pb)) % nP
            if ib < ia or (ib == ia and pb < pa):
                ib += nP
            mid = [seq[i % nP] for i in range(ia + 1, ib + 1)]
            pts = np.array([a] + [Vc[v] for v in mid] + [b])
            keep = shortest_clear_path(pts, C, R, eps)
            nodes += [("v", mid[i - 1]) for i in keep[1:-1]]
    return nodes


def rope_piece(a, ta, b, tb, C, R, flat, tol, eps):
    """("line",), ("corner", X), or None when the contour fallback is needed."""
    d = b - a
    l = np.hypot(*d)
    if l < tol:
        return ("line",)
    # flat turn, or one tangent (nearly) pointing at the other point
    cf = np.cos(flat)
    if ta @ tb >= cf or ta @ d >= cf * l or tb @ d >= cf * l:
        return None if cuts_circle(a, b, C, R, eps) else ("line",)
    sol = tangent_corner(a, ta, b, tb)
    if sol is None or sol[1] < 0 or sol[2] < 0:
        return None
    X = sol[0]
    return None if cuts_circle(a, X, C, R, eps) or cuts_circle(X, b, C, R, eps) else ("corner", X)


def shortest_clear_path(pts, C, R, eps):
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
            if dj < dist[j] and (abs(i - j) == 1 or not cuts_circle(pts[i], pts[j], C, R, eps)):
                dist[j], prev[j] = dj, i
    path = [k - 1]
    while path[-1] > 0:
        path.append(prev[path[-1]])
    return path[::-1]


# ----------------------------------------------------------------------------- loops and q's
def contour_loops(Vc, nbrs):
    """Loops of the contour graph as (closed, merged vertex ids in walking
    order). A walk stuck at a junction keeps the cycle it closed there."""
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


# ----------------------------------------------------------------------------- geometry
def cuts_circle(p0, p1, C, R, eps):
    """Whether the segment p0 p1 cuts into any circle (C, R) by more than eps."""
    d = p1 - p0
    L2 = d @ d
    A = C - p0
    if L2 > 1e-24:
        A = A - np.clip((A @ d) / L2, 0, 1)[:, None] * d
    return bool(((A * A).sum(1) < (R - eps) ** 2)[R > eps].any())


def tangent_corner(a, ta, b, tb):
    """(X, lam, mu) with X = a + lam ta = b - mu tb, or None if parallel."""
    det = ta[0] * tb[1] - ta[1] * tb[0]
    if abs(det) < 1e-12:
        return None
    d = b - a
    lam = (d[0] * tb[1] - d[1] * tb[0]) / det
    mu = (ta[0] * d[1] - ta[1] * d[0]) / det
    return a + lam * ta, lam, mu


def corner_point(a, ta, b, tb):
    sol = tangent_corner(a, ta, b, tb)
    return sol[0] if sol is not None else (a + b) / 2


def corner_jacobian(nd, qc, h=1e-7):
    """(ka, dX/dangle_a, kb, dX/dangle_b) of a corner node by central
    differences, rotating each q together with its tangent around its
    circle's center."""
    _, a, ta, b, tb, ka, kb = nd
    z = np.zeros(2)

    def rot(p, ctr, t):
        cs, sn = np.cos(t), np.sin(t)
        d = p - ctr
        return ctr + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])

    dA = (corner_point(rot(a, qc[ka], h), rot(ta, z, h), b, tb) -
          corner_point(rot(a, qc[ka], -h), rot(ta, z, -h), b, tb)) / (2 * h)
    dB = (corner_point(a, ta, rot(b, qc[kb], h), rot(tb, z, h)) -
          corner_point(a, ta, rot(b, qc[kb], -h), rot(tb, z, -h))) / (2 * h)
    return ka, dA, kb, dB
