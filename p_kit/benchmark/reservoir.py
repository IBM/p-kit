"""Benchmark a p-bit reservoir model's state-space dynamics."""

import numpy as np

from p_kit.visualization import hypercube_metrics


def drive_reservoir(model, text, n_chars=2000):
    """Reset `model` and step its reservoir through a slice of `text`,
    returning the {-1,+1}^n_pbits state trajectory and the driving
    character ids (one per step)."""
    model.reset()
    sample = text[:n_chars]
    char_ids = [model.char_to_id[c] for c in sample if c in model.char_to_id]
    history = np.array([model.step(i) for i in char_ids])
    return history, char_ids


def benchmark_reservoir(model, text, n_chars=2000):
    """Drive `model`'s reservoir on a slice of `text` and report its
    dynamics as three numbers: relaxation time (samples), PCA
    signal-over-noise (std devs above the shuffled-column null), and
    label-separability ratio (how cleanly the driving characters cluster
    in the reservoir's top-3 PCA directions)."""
    history, char_ids = drive_reservoir(model, text, n_chars)
    return hypercube_metrics(history, labels=char_ids)
