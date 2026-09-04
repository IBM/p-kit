from .port import *

import numpy as np
from typing import Dict, Any


class PCircuit:
    """Create and holds J and h parameters.

    Parameters
    ----------
    n_pbits: string
        Identifier of the pipeline (for log purposes).

    Attributes
    ----------
    h : np.array((n_pbits, 1))
        biases
    J : np.array((n_pbits, n_pbits))
        weights
    ports : Dict[str, Any] | None
        circuit ports

    """

    def __init__(self, n_pbits: int, ports: Dict[str, Any] = None):
        self.n_pbits = n_pbits
        self.ports = ports
        
        self.h = np.zeros((n_pbits,))
        self.J = np.zeros((n_pbits, n_pbits))
        self._connections = {}
        # Bumped on every in-place J mutation so callers that cache J by
        # identity (e.g. CaSuDaSolver's cache_J) can detect staleness
        # even though id(self.J) doesn't change across set_weight() calls.
        self._j_version = 0
        self._correlation_components = {}
        
        if ports:
            self._initialize_ports(ports)

    def _initialize_ports(self, port_attrs: Dict[str, Any]) -> None:
        """Initialize ports from attributes dictionary."""
        # Build cumulative index mapping respecting port widths
        port_indices = {}
        idx = 0
        for name, port in port_attrs.items():
            port_indices[name] = idx
            idx += port.width
            
        # Set up each port
        for name, port in port_attrs.items():
            new_port = Port(name=port.name, width=port.width)
            new_port.circuit = self
            new_port.index = port_indices[name]
            # Check ports name doesn't conflict with reserved attributes.
            assert name not in ("ports", "h", "J")
            setattr(self, name, new_port)

    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        self.J[from_pbit, to_pbit] = weight
        if sym:
            self.J[to_pbit, from_pbit] = weight
        self._j_version += 1

    def add_correlation(self, i, j, weight=1.0, sym=True):
        """Add positive coupling. i,j: p-bit indices; weight: strength; sym: symmetric coupling."""
        w = abs(weight)
        self.J[i, j] += w
        if sym:
            self.J[j, i] += w
        self._j_version += 1

    def add_anticorrelation(self, i, j, weight=1.0, sym=True):
        """Add negative coupling. i,j: p-bit indices; weight: strength; sym: symmetric coupling."""
        w = -abs(weight)
        self.J[i, j] += w
        if sym:
            self.J[j, i] += w
        self._j_version += 1

    def add_group_coupling(self, group_a, group_b, weights=1.0, sym=True):
        """Couple two groups. group_a/group_b: indices; weights: scalar or matrix; sym: symmetric coupling."""
        a = np.asarray(group_a, dtype=int)
        b = np.asarray(group_b, dtype=int)
        W = np.full((len(a), len(b)), float(weights)) if np.isscalar(weights) else np.asarray(weights, dtype=float)
        if W.shape != (len(a), len(b)):
            raise ValueError(f"weights shape {W.shape}, expected {(len(a), len(b))}")
        self.J[np.ix_(a, b)] += W
        if sym:
            self.J[np.ix_(b, a)] += W.T
        self._j_version += 1

    def add_group_competition(self, group, strength=1.0):
        """Add pairwise anticorrelation within a group. group: indices; strength: inhibition strength."""
        g = np.asarray(group, dtype=int)
        w = abs(float(strength))
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                self.J[g[i], g[j]] -= w
                self.J[g[j], g[i]] -= w
        self._j_version += 1

    def set_correlation_component(self, name, J=None, h=None):
        """Set named dynamic component. name: identifier; J: coupling matrix; h: bias vector."""
        J = np.zeros_like(self.J) if J is None else np.asarray(J, dtype=float)
        h = np.zeros_like(self.h) if h is None else np.asarray(h, dtype=float).reshape(-1)
        if J.shape != self.J.shape:
            raise ValueError(f"J shape {J.shape}, expected {self.J.shape}")
        if h.shape != self.h.shape:
            raise ValueError(f"h shape {h.shape}, expected {self.h.shape}")
        self._correlation_components[name] = {"J": J.copy(), "h": h.copy()}

    def get_correlation_component(self, name):
        """Return named component. name: component identifier."""
        return self._correlation_components[name]

    def remove_correlation_component(self, name):
        """Remove named component. name: component identifier."""
        del self._correlation_components[name]

    def clear_correlation_components(self):
        self._correlation_components.clear()

    def correlation_components(self):
        return tuple(self._correlation_components)

    def effective_parameters(self, component_scales=None):
        """Return effective J,h. component_scales: {name: scale} for active components."""
        J = self.J.copy()
        h = self.h.copy()
        if component_scales:
            for name, scale in component_scales.items():
                c = self._correlation_components[name]
                J += scale * c["J"]
                h += scale * c["h"]
        return J, h

    def copy(self):
        new_circuit = PCircuit(self.n_pbits, self.ports)
        new_circuit.J = self.J
        new_circuit.h = self.h
        new_circuit._j_version = self._j_version
        new_circuit._correlation_components = {
            name: {"J": c["J"].copy(), "h": c["h"].copy()}
            for name, c in self._correlation_components.items()
        }
        return new_circuit