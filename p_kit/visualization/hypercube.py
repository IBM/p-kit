"""Plot the {-1,+1}^n p-bit state space as a hypercube graph"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .utils import m_to_string

# Above this many p-bits, {-1,+1}^n_pbits can no longer be enumerated (2^n
# vertices) - e.g. a 128-p-bit reservoir has more states than atoms in the
# observable universe. hypercube_plot then falls back to a PCA-projected
# view of the sampled trajectory instead of the exact vertex/edge graph.
MAX_EXACT_DIMS = 12

# Above this many vertices, per-vertex bitstring labels are skipped to
# avoid an unreadable plot.
MAX_LABELED_VERTS = 64


def _nested_cube_coords(n, shrink=0.55):
    """Recursively embed the n-cube's vertices in 3D: the first 3 bits form
    a real cube; each further bit halves the current point set into two
    shrunk copies offset along a fixed direction, connected vertex-to-vertex
    (the standard 'tesseract' construction, extended past n=4)."""
    base_bits = list(itertools.product([-1, 1], repeat=min(n, 3)))
    coords = {b: np.array(b, dtype=float) for b in base_bits}
    directions = [np.array(d, dtype=float) for d in
                  [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
    for k in range(3, n):
        d = directions[(k - 3) % len(directions)]
        d = d / np.linalg.norm(d) * 1.5 * (1 - shrink)
        new_coords = {}
        for bits, pos in coords.items():
            new_coords[bits + (1,)] = pos * shrink + d
            new_coords[bits + (-1,)] = pos * shrink - d
        coords = new_coords
    return coords


def _hypercube_plot_exact(output, n_pbits):
    """Enumerate all 2^n_pbits vertices exactly and draw the true
    Hamming-distance-1 edge graph, colored/sized by sampled visit counts."""
    verts = np.array(list(itertools.product([-1, 1], repeat=n_pbits)))
    n_verts = len(verts)
    edges = [(i, j) for i in range(n_verts) for j in range(i + 1, n_verts)
             if np.sum(verts[i] != verts[j]) == 1]

    pos = _nested_cube_coords(n_pbits)
    pos3d = np.array([pos[tuple(v)] for v in verts])

    counts = np.zeros(n_verts, dtype=int)
    for m in output:
        idx = np.where((verts == m).all(axis=1))[0][0]
        counts[idx] += 1

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    lines = [(pos3d[i], pos3d[j]) for i, j in edges]
    edge_collection = Line3DCollection(
        lines, colors="gray", linewidths=0.5, alpha=0.3
    )
    ax.add_collection3d(edge_collection)

    sizes = 60 if counts.max() == 0 else 40 + 400 * counts / counts.max()
    sc = ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2],
                    c=counts, s=sizes, cmap="viridis",
                    edgecolors="black", linewidths=0.3, depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="visits")

    if n_verts <= MAX_LABELED_VERTS:
        fontsize = max(5, 9 - 0.5 * max(0, n_pbits - 3))
        for v in verts:
            p = pos[tuple(v)]
            ax.text(p[0] * 1.18, p[1] * 1.18, p[2] * 1.18,
                    m_to_string(v), fontsize=fontsize, ha="center")

    if n_pbits <= 3:
        ax.set_xlabel("p-bit 0")
        ax.set_ylabel("p-bit 1")
        ax.set_zlabel("p-bit 2")
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_zticks([-1, 1])
    else:
        ax.set_axis_off()

    ax.set_title(f"{{-1,+1}}^{n_pbits} p-bit state space "
                 f"({n_verts} vertices, exact)")
    plt.show()


def _hypercube_plot_projected(output, n_pbits):
    """Too many p-bits to enumerate exactly: PCA-project the *sampled*
    trajectory (each an n_pbits-dim +/-1 vector) into 3D and draw it as a
    path colored by step, since the full vertex/edge structure can't be
    built or even counted at this scale."""
    X = np.asarray(output, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_components = min(3, Vt.shape[0])
    proj = Xc @ Vt[:n_components].T
    if n_components < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - n_components)))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    t = np.arange(len(proj))
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
            color="gray", alpha=0.4, linewidth=1)
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=t, cmap="viridis", s=25, depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="step")
    ax.set_axis_off()
    ax.set_title(f"{{-1,+1}}^{n_pbits} has {2 ** n_pbits:.3g} vertices "
                 f"(too many to enumerate)\n"
                 f"PCA-projected sampled trajectory instead")
    plt.show()


def hypercube_plot(output, n_pbits=None):
    """Plot a p-circuit's {-1,+1}^n_pbits state space.

    For n_pbits <= MAX_EXACT_DIMS, this enumerates every vertex exactly:
    for n_pbits <= 3 it's a literal cube, otherwise a nested/recursive-cube
    3D projection (vertices still connected exactly by Hamming distance 1).
    Each vertex is sized/colored by how often ``output`` visited it.

    Above MAX_EXACT_DIMS, 2^n_pbits vertices can no longer be enumerated
    (e.g. a 128-p-bit reservoir has 2^128 states) - this instead PCA-
    projects the sampled trajectory into 3D and draws it as a path.

    Parameters
    ----------
    output : array-like, shape (n_samples, n_pbits)
        A +/-1 state trajectory, e.g. ``all_m`` from ``Solver.solve()`` or
        a sequence of ``return_final=True`` states.
    n_pbits : int, optional
        Number of p-bits. Inferred from ``output``'s second dimension when
        omitted.
    """
    output = np.asarray(output)
    if n_pbits is None:
        n_pbits = output.shape[1]

    if n_pbits <= MAX_EXACT_DIMS:
        _hypercube_plot_exact(output, n_pbits)
    else:
        _hypercube_plot_projected(output, n_pbits)
