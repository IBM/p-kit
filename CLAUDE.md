# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**p-kit** is a Python library (IBM, BSD-3-Clause) for simulating probabilistic circuits using p-bits that fluctuate between -1 and +1. It implements two frameworks:
- **PSL (Probabilistic Spin Logic)**: Invertible classical logic via Gibbs sampling — logic gates map to probabilistic spin logic gates.
- **PAOA (Probabilistic Approximate Optimization Algorithms)**: Probabilistic analog of QAOA for combinatorial optimization.

## Commands

```bash
pip install .           # Install base package
pip install .[tests]    # Include test deps (pytest, seaborn, flake8)

pytest                  # Run all tests
pytest tests/test_core.py::test_canary  # Run a single test
flake8 p_kit tests      # Lint
```

CI runs tests on Python 3.10/3.11 across Ubuntu, macOS, Windows (`python-app.yml`), and validates all examples execute without error (`run_examples.yml`).

## Architecture

### Decorator-based circuit API (`p_kit.psl`)

The entry point is a **decorator API** in `p_kit/psl/decorators.py`:

- `@pcircuit(n_pbits)`: Transforms a class into a circuit by injecting `J` (n×n coupling matrix) and `h` (n×1 bias vector) management, plus port tracking. The J/h matrices are typically defined as class-level `np.ndarray` attributes.
- `@module`: Transforms a class into a container of circuit instances with `synthesize()` support. Records all child circuit instances and their port connections, then merges their local J/h matrices into a global representation.

**`Port`** (`p_kit/psl/port.py`): Represents a circuit's I/O terminal. Ports are connected with `port.connect(other_port, ConnectionType)`:
- `NoCopyConnection`: Shared global index — no extra coupling terms
- `VanillaCopyConnection`: Adds a weight-1 coupling between the two p-bits
- `WeightedCopyConnection`: Same with custom weight

**`ModuleContext`** (`p_kit/psl/context.py`): Manages global index assignment during synthesis. `synthesize(format='sparse'|'dense')` returns the merged J and h across all registered instances.

### Solvers (`p_kit/solver/`)

**`Solver`** (base): Parameters are `Nt` (timesteps), `dt`, `i0` (correlation strength 0–1), optional `expected_mean` and `seed`. Subclasses implement `solve(circuit)`.

**`CaSuDaSolver`**: Primary solver. At each step, computes input currents `I = i0 * (m @ J + h)`, then updates magnetization `m` stochastically. Returns `(all_I, all_m, E)` — full `Nt × n_pbits` trajectories for currents and magnetizations, plus energy vector.

**`GibbsSolver`**: Asynchronous single-p-bit-per-step Gibbs sampler (contrast with CaSuDaSolver's synchronous, vectorized update). Resamples exactly one randomly chosen p-bit per timestep via a sigmoid acceptance probability, conditioned on the current state of all others. Same constructor and return contract as `CaSuDaSolver` (drop-in swap), but converges slower — useful as a textbook-correct reference to validate CaSuDa's synchronous approximation against.

**Annealing** (`p_kit/solver/annealing.py`): `constant` and `linear` schedules that modify `i0` over time. `execute(solver, circuit, annealing, n_shots, n_last_samples, n_jobs)` runs the solver multiple times in parallel (via `joblib`) and collects final samples.

### Gate library (`p_kit/psl/gates/`)

Pre-built `@pcircuit` classes with hardcoded J/h matrices: `ANDGate`, `ORGate` (3 p-bits each), `FullAdder` (5 p-bits). These are the primary reference for how to encode logic as probabilistic couplings.

### Visualization (`p_kit/visualization/`)

`histplot`, `energyplot`, `vin_vout`, `heatmap`, `plot3d`, `visualize_tsp_route`. Most functions take the solver output arrays directly.

### Library (`p_kit/library/`)

Higher-level algorithms built on top of the framework — currently `p_kit.library.tsp` for the Traveling Salesman Problem.

## Test structure

Tests in `tests/` use `conftest.py` fixtures (e.g., `rndstate` for seeded random). Visualization tests use `@requires_matplotlib` / `@requires_seaborn` decorators (defined in conftest) to skip when optional deps are absent. The test suite is intentionally lightweight — `test_canary` is `assert True`.

## Key dependency versions

`numpy<2.3`, `scipy==1.15.3`, `cython==3.2.4`, `cvxpy==1.7.5`, `matplotlib==3.10.9`. Python >=3.10 required.
