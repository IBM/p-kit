"""Plot the {-1,+1}^n p-bit state space as a hypercube graph"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .utils import m_to_string

# Above this many p-bits, {-1,+1}^n_pbits can no longer be enumerated (2^n
# vertices) - e.g. a 128-p-bit reservoir has more states than atoms in the
# observable universe. hypercube_plot then falls back to a statistical view
# of the sampled trajectory instead of the exact vertex/edge graph.
MAX_EXACT_DIMS = 12

# Above this many vertices, per-vertex bitstring labels are skipped to
# avoid an unreadable plot.
MAX_LABELED_VERTS = 64

# How many leading principal components to compare against the noise null.
N_PCA_COMPONENTS = 15


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


def _pca_explained_variance(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    return var / var.sum()


def _column_shuffle_null(X, rng, n_components, n_trials=20):
    """Null PCA spectrum: independently permute each column (p-bit) across
    samples. This preserves every p-bit's own marginal (its bias/frequency)
    exactly while destroying all cross-p-bit and temporal correlation - the
    fair 'no structure' baseline to compare the real spectrum against."""
    spectra = np.zeros((n_trials, n_components))
    noise = np.empty_like(X)
    for t in range(n_trials):
        for j in range(X.shape[1]):
            noise[:, j] = rng.permutation(X[:, j])
        spectra[t] = _pca_explained_variance(noise)[:n_components]
    return spectra


def _correlation_function(X, max_lag, n_pbits):
    """C(dt) = <m(t).m(t+dt)> / n_pbits, the standard two-point overlap
    used to characterize relaxation/mixing in stochastic spin dynamics.
    Related to the mean Hamming distance at lag dt by
    d_H(dt) = n_pbits * (1 - C(dt)) / 2."""
    lags = np.arange(1, max_lag + 1)
    C = np.empty(max_lag)
    for i, dt in enumerate(lags):
        overlap = (X[:-dt] * X[dt:]).sum(axis=1) / n_pbits
        C[i] = overlap.mean()
    return lags, C


def _hypercube_plot_projected(output, n_pbits, labels=None):
    """Too many p-bits to enumerate exactly (2^n_pbits vertices - e.g. a
    128-p-bit reservoir has ~3.4e38 of them). A spatial scatter of a lossy
    3D projection turns out to be close to meaningless on its own: a
    trajectory that looks like a structureless tangle can still carry a
    strong, statistically unambiguous signal, and "6% of variance in 3
    components" can be either nothing or a huge excess depending entirely
    on what the noise floor is. So instead of guessing from a picture:

    - the relaxation/correlation function C(dt) = <m(t).m(t+dt)>/n_pbits
      shows how fast the system decorrelates/mixes - the hypercube
      analogue of a diffusion curve, since the mean Hamming distance at
      lag dt is n_pbits*(1-C(dt))/2.
    - the PCA spectrum is shown against a column-shuffled noise null (each
      p-bit's own time series independently permuted, preserving its
      marginal but destroying all correlation), so "is there structure at
      all" has an actual answer instead of an eyeballed guess.
    - if `labels` (one per sample, e.g. a per-step token/class id) is
      given, a third panel projects onto the top-3 PCA directions colored
      by label, with a between/within separability ratio - spatial
      structure is usually organized by an external driver like this, not
      by time or raw position alone.
    """
    X = np.asarray(output, dtype=float)
    n_samples = len(X)
    rng = np.random.default_rng(0)

    n_panels = 3 if labels is not None else 2
    fig = plt.figure(figsize=(6.5 * n_panels, 5.5))

    # --- panel 1: relaxation / correlation function ---
    max_lag = max(1, min(300, n_samples // 3))
    lags, C = _correlation_function(X, max_lag, n_pbits)
    shuffled = X[rng.permutation(n_samples)]
    baseline = (X * shuffled).sum(axis=1).mean() / n_pbits

    ax1 = fig.add_subplot(1, n_panels, 1)
    ax1.plot(lags, C, label="C(dt) = <m(t).m(t+dt)>/n")
    ax1.axhline(baseline, color="gray", linestyle="--",
                label=f"fully-mixed baseline ({baseline:.3f})")
    ax1.set_xlabel("lag dt (samples)")
    ax1.set_ylabel("overlap / correlation")
    ax1.set_title("Relaxation: correlation vs. time lag")
    ax1.legend()

    # --- panel 2: PCA spectrum vs. noise null ---
    n_components = min(N_PCA_COMPONENTS, n_pbits)
    real_var = _pca_explained_variance(X)[:n_components]
    null_spectra = _column_shuffle_null(X, rng, n_components)
    null_mean = null_spectra.mean(axis=0)
    null_std = null_spectra.std(axis=0)

    ax2 = fig.add_subplot(1, n_panels, 2)
    idx = np.arange(n_components)
    ax2.bar(idx - 0.2, real_var, width=0.4, label="real trajectory")
    ax2.bar(idx + 0.2, null_mean, width=0.4, yerr=null_std,
            label="shuffled-column noise")
    ax2.set_xlabel("principal component")
    ax2.set_ylabel("explained variance ratio")
    top3_excess = real_var[:3].sum() - null_mean[:3].sum()
    top3_z = top3_excess / max(null_std[:3].sum(), 1e-12)
    ax2.set_title(f"PCA spectrum vs. noise null\n"
                  f"(top-3 excess: {top3_z:.0f} std devs above noise)")
    ax2.legend()

    # --- panel 3 (optional): projection colored by external label ---
    if labels is not None:
        labels = np.asarray(labels)
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:3].T

        grand_mean = proj.mean(axis=0)
        between = within = 0.0
        for g in np.unique(labels):
            pts = proj[labels == g]
            if len(pts) < 2:
                continue
            gm = pts.mean(axis=0)
            between += len(pts) * np.sum((gm - grand_mean) ** 2)
            within += np.sum((pts - gm) ** 2)
        sep_ratio = between / within if within > 0 else float("nan")

        _, label_codes = np.unique(labels, return_inverse=True)
        ax3 = fig.add_subplot(1, n_panels, 3, projection="3d")
        ax3.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=label_codes, cmap="tab20", s=25, depthshade=False)
        ax3.set_axis_off()
        ax3.set_title("PCA-3 projection colored by label\n"
                      f"separability ratio: {sep_ratio:.3f}")

    fig.suptitle(f"{{-1,+1}}^{n_pbits} has {2 ** n_pbits:.3g} vertices "
                 f"(too many to enumerate)\n"
                 f"statistical view of {n_samples} sampled states")
    fig.tight_layout()
    plt.show()


def hypercube_plot(output, n_pbits=None, labels=None):
    """Plot a p-circuit's {-1,+1}^n_pbits state space.

    For n_pbits <= MAX_EXACT_DIMS, this enumerates every vertex exactly:
    for n_pbits <= 3 it's a literal cube, otherwise a nested/recursive-cube
    3D projection (vertices still connected exactly by Hamming distance 1).
    Each vertex is sized/colored by how often ``output`` visited it.

    Above MAX_EXACT_DIMS, 2^n_pbits vertices can no longer be enumerated
    (e.g. a 128-p-bit reservoir has 2^128 states). A spatial scatter of a
    lossy 3D projection is close to meaningless there on its own, so this
    instead shows a relaxation/correlation curve and the trajectory's PCA
    spectrum against a noise null - see ``_hypercube_plot_projected`` for
    why. Pass ``labels`` (one per sample) to add a labeled 3D projection
    when there's a natural external driver (e.g. a per-step token id) to
    color by.

    Parameters
    ----------
    output : array-like, shape (n_samples, n_pbits)
        A +/-1 state trajectory, e.g. ``all_m`` from ``Solver.solve()`` or
        a sequence of ``return_final=True`` states.
    n_pbits : int, optional
        Number of p-bits. Inferred from ``output``'s second dimension when
        omitted.
    labels : array-like, shape (n_samples,), optional
        Only used above MAX_EXACT_DIMS: per-sample group labels (e.g. a
        driving token/class id) to color a labeled PCA projection by.
    """
    output = np.asarray(output)
    if n_pbits is None:
        n_pbits = output.shape[1]

    if n_pbits <= MAX_EXACT_DIMS:
        _hypercube_plot_exact(output, n_pbits)
    else:
        _hypercube_plot_projected(output, n_pbits, labels=labels)
