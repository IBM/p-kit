"""Utils method for visualization"""
from collections import namedtuple
import numpy as np

# How many leading principal components to compare against the noise null,
# in hypercube_metrics/hypercube_plot's statistical view.
N_PCA_COMPONENTS = 15


def m_to_string(outputs):
    ret = ""
    for output in outputs:
        ret += "1" if output == 1 else "0"
    return ret

def extract_tsp_path(sample_matrix):
    """ Extracts the TSP path from a binary matrix. """
    path = []
    for order in range(sample_matrix.shape[1]): 
        step = (sample_matrix[:, order])
        maxes = np.argwhere(step == np.max(step)).flatten()
        city = np.random.choice(maxes)
        path.append(city)
    return path

def tsp_hist(samples, city_graph):
    hist = {}
    for i in range(len(samples)):
        s = samples[i, :].reshape((len(city_graph), len(city_graph[0])))
        path = extract_tsp_path(s)
        key = "".join([str(p) for p in path])
        if key in hist:
            hist[key] = hist[key] + 1
        else:
            hist[key] = 1

    return hist


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


def _relaxation(X, n_pbits, rng, max_lag=None):
    """Correlation curve C(dt), its fully-mixed baseline, and a scalar
    relaxation time: the first lag at which C(dt)'s excess over baseline
    decays to 1/e of its initial value (the usual exponential-decay-time
    convention, read off a curve that need not itself be exponential)."""
    n_samples = len(X)
    if max_lag is None:
        max_lag = max(1, min(300, n_samples // 3))
    lags, C = _correlation_function(X, max_lag, n_pbits)
    shuffled = X[rng.permutation(n_samples)]
    baseline = (X * shuffled).sum(axis=1).mean() / n_pbits

    excess = C - baseline
    if excess[0] <= 0:
        tau = float(lags[0])
    else:
        below = np.where(excess <= excess[0] / np.e)[0]
        tau = float(lags[below[0]]) if len(below) else float(lags[-1])

    return lags, C, baseline, tau


def _pca_vs_noise(X, n_pbits, rng, n_components):
    """Real PCA spectrum, its column-shuffled noise null, and a scalar
    signal-to-noise readout: how many noise standard deviations the top-3
    real components' combined explained variance sits above the top-3
    noise mean - "is there structure at all" as a number instead of an
    eyeballed bar chart."""
    n_components = min(n_components, n_pbits)
    real_var = _pca_explained_variance(X)[:n_components]
    null_spectra = _column_shuffle_null(X, rng, n_components)
    null_mean = null_spectra.mean(axis=0)
    null_std = null_spectra.std(axis=0)
    top3_excess = real_var[:3].sum() - null_mean[:3].sum()
    snr = top3_excess / max(null_std[:3].sum(), 1e-12)
    return real_var, null_mean, null_std, snr


def _separability(X, labels):
    """Top-3 PCA projection plus its between/within group separability
    ratio for the given per-sample labels."""
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

    return proj, sep_ratio


HypercubeMetrics = namedtuple(
    "HypercubeMetrics", ["relaxation_time", "pca_snr", "separability_ratio"]
)


def hypercube_metrics(output, n_pbits=None, labels=None,
                      n_components=N_PCA_COMPONENTS, seed=0):
    """Compute hypercube_plot's statistical-view diagnostics without
    plotting - handy for benchmarking a reservoir/model's dynamics
    directly (e.g. to compare configurations or models numerically).

    Parameters
    ----------
    output : array-like, shape (n_samples, n_pbits)
        A +/-1 state trajectory, e.g. from repeated ``model.step(...)``.
    n_pbits : int, optional
        Inferred from ``output``'s second dimension when omitted.
    labels : array-like, shape (n_samples,), optional
        Per-sample group labels (e.g. a driving token id). When omitted,
        ``separability_ratio`` is ``None``.
    n_components : int, optional
        Leading PCA components to compare against the noise null.
    seed : int, optional
        Seed for the noise-null RNG, for reproducible metrics.

    Returns
    -------
    HypercubeMetrics
        ``relaxation_time``: lag (in samples) at which the correlation
        curve's excess over its fully-mixed baseline decays to 1/e.
        ``pca_snr``: top-3 PCA explained-variance excess over the
        column-shuffled noise null, in noise standard deviations.
        ``separability_ratio``: between/within-group variance ratio of
        the top-3 PCA projection, or ``None`` if ``labels`` is omitted.
    """
    X = np.asarray(output, dtype=float)
    if n_pbits is None:
        n_pbits = X.shape[1]
    rng = np.random.default_rng(seed)

    _, _, _, tau = _relaxation(X, n_pbits, rng)
    _, _, _, snr = _pca_vs_noise(X, n_pbits, rng, n_components)
    sep_ratio = _separability(X, labels)[1] if labels is not None else None

    return HypercubeMetrics(tau, snr, sep_ratio)
