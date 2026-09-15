"""
Demo 2: calibrate and execute a quantum circuit with a p-bit/Riemannian runtime backend.

## Purpose

Demo 1 establishes the basic single-qubit representation:

Demo 2 follows a more familiar quantum-computing workflow and adds an explicit
noise model and calibration loop:

## Workflow

1. Define an intended quantum circuit.
2. Define backend imperfections using the NoiseModel class.
3. Execute the circuit with PBitRuntimeBackend.
4. Realize the output state statistically with p-bits.
5. Estimate the resulting quantum state on CP^1 using a Riemannian mean.
6. Compare the produced state with the ideal target using Fubini-Study distance.
7. Calibrate the programmable gate parameters.
8. Execute the corrected circuit again.
9. Compare state fidelity and Born measurement probabilities before/after calibration.

## Main components

QuantumCircuit
Describes the intended quantum computation and programmable gate parameters.

NoiseModel
Explicitly describes backend imperfections. In this demo it contains
reproducible coherent gate-angle errors. The class is separate from the
quantum circuit so that the intended computation remains ideal while the
runtime backend models imperfect execution.

PBitRuntimeBackend
Executes the circuit using the geometric single-qubit model and realizes
the resulting state statistically with p-bits.

Quantum-state manifold
A pure single-qubit state is represented as a point on CP^1, equivalently
the Bloch sphere S^2. The manifold constrains the stochastic representation
to valid pure quantum states.

Riemannian state estimator
Combines repeated noisy p-bit state realizations using an intrinsic
Frechet/Karcher mean on the quantum-state manifold.

Fubini-Study / geometric loss
Measures the distance between the estimated runtime state and the desired
ideal target state.

Calibration
Adjusts the programmable gate commands so that execution on the imperfect
stochastic runtime backend produces a state closer to the desired target.

Corrected QuantumCircuit
The calibrated circuit is specialized to the imperfect runtime backend.
Its command parameters therefore do not necessarily equal the ideal
parameters.

## Interpretation

The p-bits provide the stochastic realization of the quantum state.

CP^1 geometry provides the state constraint, the Riemannian estimator, and the
Fubini-Study calibration objective.

The NoiseModel class makes runtime imperfections explicit rather than hiding
them inside the circuit or the backend implementation.

The main question explored by this demo is:

Can a noisy stochastic p-bit quantum runtime be calibrated using
quantum-state Riemannian geometry so that it still produces the
expected quantum-state result?

## IMPORTANT

The gate imperfections used in this file are deliberately simulated so that
the calibration step has a reproducible error to correct. They are not meant
to model a particular physical quantum processor or p-bit device.

The same NoiseModel/runtime interface could later be replaced or populated
with experimentally measured distortions from a physical Probana/p-kit
backend.

This is a stochastic research demo, not a physical quantum computer.
"""

from dataclasses import dataclass
import copy
import numpy as np

from p_kit.psl import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver


# ============================================================================
# Demo configuration
# ============================================================================

TARGET_THETA = 0.95
TARGET_PHI = 0.70

# p-bit state realization
STATE_NT = 2_000
STATE_BURN_IN = 400
STATE_SHOTS = 4
N_REALIZATIONS = 7
DT = 0.02
BIAS_CLIP = 0.995

# Measurement
MEASUREMENT_SHOTS = 2048
MEASUREMENT_NT = 250

# Reproducibility
BASE_REALIZATION_SEED = 1000
MEASUREMENT_SEED = 9000

# Calibration search
INITIAL_CALIBRATION_STEP = 0.08
MIN_CALIBRATION_STEP = 0.00125
MAX_CALIBRATION_ROUNDS = 18

# --------------------------------------------------------------------------
# Explicit demo noise parameters.
#
# These are deliberately injected coherent gate-angle errors.  They make the
# noisy-backend calibration reproducible and visible:
#
#     actual Rz = gain * commanded Rz + offset
#     actual Ry = gain * commanded Ry + offset
#
# They are not claimed to model a particular physical device.
# --------------------------------------------------------------------------

RZ_NOISE_OFFSET = 0.045
RY_NOISE_GAIN = 0.960
RY_NOISE_OFFSET = -0.025


# ============================================================================
# CP^1 / Bloch-sphere geometry
# ============================================================================

class CP1BlochManifold:
    """Pure single-qubit state manifold CP^1 represented by Bloch S^2."""

    X_AXIS = np.array([1.0, 0.0, 0.0])
    Y_AXIS = np.array([0.0, 1.0, 0.0])
    Z_AXIS = np.array([0.0, 0.0, 1.0])

    @staticmethod
    def point(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)

        if norm == 0.0:
            raise ValueError("The zero vector cannot define a pure qubit state.")

        return vector / norm

    @classmethod
    def tangent(cls, point, vector):
        point = cls.point(point)
        vector = np.asarray(vector, dtype=float)
        return vector - np.dot(vector, point) * point

    @classmethod
    def exp_map(cls, point, tangent_vector):
        """
        Sphere exponential map.

        The sphere geodesic angle is twice the Fubini-Study distance, but the
        same intrinsic mean point is obtained because that constant metric
        scaling does not change the minimizer.
        """
        point = cls.point(point)
        tangent_vector = cls.tangent(point, tangent_vector)
        angle = np.linalg.norm(tangent_vector)

        if angle < 1e-15:
            return point.copy()

        return cls.point(
            np.cos(angle) * point
            + np.sin(angle) * tangent_vector / angle
        )

    @classmethod
    def log_map(cls, point_a, point_b):
        """Sphere logarithmic map used by the intrinsic mean."""
        point_a = cls.point(point_a)
        point_b = cls.point(point_b)

        dot = float(np.clip(np.dot(point_a, point_b), -1.0, 1.0))
        angle = float(np.arccos(dot))

        if angle < 1e-15:
            return np.zeros(3)

        if np.pi - angle < 1e-12:
            raise ValueError("Log map is not unique for antipodal points.")

        direction = point_b - dot * point_a
        direction /= np.linalg.norm(direction)

        return angle * direction

    @classmethod
    def fs_distance(cls, point_a, point_b):
        """Fubini-Study distance for pure single-qubit states."""
        point_a = cls.point(point_a)
        point_b = cls.point(point_b)

        dot = float(np.clip(np.dot(point_a, point_b), -1.0, 1.0))

        return 0.5 * float(np.arccos(dot))

    @classmethod
    def fidelity(cls, point_a, point_b):
        """Pure-state fidelity."""
        point_a = cls.point(point_a)
        point_b = cls.point(point_b)

        return float(
            np.clip(
                0.5 * (1.0 + np.dot(point_a, point_b)),
                0.0,
                1.0,
            )
        )

    @classmethod
    def rotate(cls, point, axis, angle):
        """
        Single-qubit unitary action represented as a Bloch-sphere rotation.
        """
        point = cls.point(point)
        axis = cls.point(axis)

        c = np.cos(angle)
        s = np.sin(angle)

        return cls.point(
            c * point
            + s * np.cross(axis, point)
            + (1.0 - c) * np.dot(axis, point) * axis
        )

    @classmethod
    def born_expectation(cls, point, axis):
        point = cls.point(point)
        axis = cls.point(axis)

        return float(np.clip(np.dot(point, axis), -1.0, 1.0))

    @classmethod
    def born_probability(cls, point, axis):
        return 0.5 * (1.0 + cls.born_expectation(point, axis))


M = CP1BlochManifold()
H_AXIS = M.point([1.0, 0.0, 1.0])


# ============================================================================
# Geometry-aware averaging
# ============================================================================

def projected_euclidean_mean(points):
    """Mean in R^3 followed by projection back to CP^1."""
    points = np.asarray(points, dtype=float)
    return M.point(np.mean(points, axis=0))


def riemannian_mean(points, max_iter=64, tol=1e-10):
    """
    Intrinsic Frechet/Karcher mean on CP^1.

    This is the geometry-aware state estimate used by the calibration loss.
    """
    points = np.asarray(points, dtype=float)
    mean = projected_euclidean_mean(points)

    for _ in range(max_iter):
        logs = np.asarray([M.log_map(mean, point) for point in points])
        update = np.mean(logs, axis=0)

        if np.linalg.norm(update) < tol:
            break

        mean = M.exp_map(mean, update)

    return M.point(mean)


# ============================================================================
# Minimal single-qubit circuit interface
# ============================================================================

@dataclass
class Gate:
    name: str
    angle: float | None = None
    label: str | None = None


class QuantumCircuit:
    """
    Minimal one-qubit circuit used only by this research demo.

    The interface intentionally resembles a normal quantum-circuit workflow.
    """

    def __init__(self):
        self.gates = []

    def copy(self):
        return copy.deepcopy(self)

    def h(self):
        self.gates.append(Gate("h"))
        return self

    def x(self):
        self.gates.append(Gate("x"))
        return self

    def y(self):
        self.gates.append(Gate("y"))
        return self

    def z(self):
        self.gates.append(Gate("z"))
        return self

    def t(self):
        self.gates.append(Gate("t"))
        return self

    def rx(self, angle, label=None):
        self.gates.append(Gate("rx", float(angle), label))
        return self

    def ry(self, angle, label=None):
        self.gates.append(Gate("ry", float(angle), label))
        return self

    def rz(self, angle, label=None):
        self.gates.append(Gate("rz", float(angle), label))
        return self

    def set_parameter(self, label, value):
        found = False

        for gate in self.gates:
            if gate.label == label:
                gate.angle = float(value)
                found = True

        if not found:
            raise KeyError(f"No circuit parameter labelled {label!r}.")

        return self

    def get_parameter(self, label):
        for gate in self.gates:
            if gate.label == label:
                return float(gate.angle)

        raise KeyError(f"No circuit parameter labelled {label!r}.")

    def __str__(self):
        parts = ["|0>"]

        for gate in self.gates:
            if gate.angle is None:
                parts.append(f"-- {gate.name.upper()}")
            elif gate.label is None:
                parts.append(f"-- {gate.name.upper()}({gate.angle:.4f})")
            else:
                parts.append(
                    f"-- {gate.name.upper()}({gate.label}={gate.angle:.4f})"
                )

        return " ".join(parts)


# ============================================================================
# Exact geometric circuit execution
# ============================================================================

def apply_gate_ideal(state, gate):
    """Apply one ideal gate to a Bloch point."""
    if gate.name == "h":
        return M.rotate(state, H_AXIS, np.pi)

    if gate.name == "x":
        return M.rotate(state, M.X_AXIS, np.pi)

    if gate.name == "y":
        return M.rotate(state, M.Y_AXIS, np.pi)

    if gate.name == "z":
        return M.rotate(state, M.Z_AXIS, np.pi)

    if gate.name == "t":
        return M.rotate(state, M.Z_AXIS, np.pi / 4.0)

    if gate.name == "rx":
        return M.rotate(state, M.X_AXIS, gate.angle)

    if gate.name == "ry":
        return M.rotate(state, M.Y_AXIS, gate.angle)

    if gate.name == "rz":
        return M.rotate(state, M.Z_AXIS, gate.angle)

    raise ValueError(f"Unsupported gate: {gate.name!r}")


def execute_ideal(circuit):
    """Execute a circuit exactly on CP^1."""
    state = M.Z_AXIS.copy()

    for gate in circuit.gates:
        state = apply_gate_ideal(state, gate)

    return M.point(state)


# ============================================================================
# Explicit noise model
# ============================================================================

@dataclass(frozen=True)
class GateAngleError:
    """
    Coherent single-gate angle error.

    effective_angle = gain * commanded_angle + offset
    """
    gain: float = 1.0
    offset: float = 0.0

    def apply(self, angle):
        return self.gain * float(angle) + self.offset


class NoiseModel:
    """
    Minimal Qiskit-Aer like noise container for this research demo.

    The noise model belongs to the runtime backend, not to the ideal QuantumCircuit.
    That keeps the intended circuit separate from imperfections of the device
    or simulator used to execute it.
    """

    def __init__(self):
        self.gate_angle_errors = {}

    def add_gate_angle_error(self, gate_name, gain=1.0, offset=0.0):
        gate_name = str(gate_name).lower()

        if gate_name not in {"rx", "ry", "rz"}:
            raise ValueError(
                "Gate-angle noise is supported only for rx, ry and rz."
            )

        self.gate_angle_errors[gate_name] = GateAngleError(
            gain=float(gain),
            offset=float(offset),
        )
        return self

    def apply(self, gate):
        distorted = copy.deepcopy(gate)
        error = self.gate_angle_errors.get(gate.name)

        if error is not None:
            distorted.angle = error.apply(gate.angle)

        return distorted

    def describe(self):
        if not self.gate_angle_errors:
            return ["ideal noise model"]

        lines = []

        for gate_name in sorted(self.gate_angle_errors):
            error = self.gate_angle_errors[gate_name]
            lines.append(
                f"{gate_name.upper()}: actual angle = "
                f"{error.gain:.6f} * commanded angle "
                f"{error.offset:+.6f} rad"
            )

        return lines


# ============================================================================
# p-bit / Riemannian runtime backend
# ============================================================================

class PBitRuntimeBackend:
    """
    Stochastic p-bit single-qubit runtime backend.

    Gate evolution is represented geometrically on CP^1. The resulting state
    is realized statistically by three p-bits whose means represent the Bloch
    coordinates.

    An explicit NoiseModel is attached to the backend so noisy execution and
    calibration are visible and reproducible.
    """

    def __init__(
        self,
        noise_model=None,
        state_nt=STATE_NT,
        state_burn_in=STATE_BURN_IN,
        state_shots=STATE_SHOTS,
        dt=DT,
        bias_clip=BIAS_CLIP,
    ):
        self.noise_model = noise_model if noise_model is not None else NoiseModel()

        self.state_nt = int(state_nt)
        self.state_burn_in = int(state_burn_in)
        self.state_shots = int(state_shots)
        self.dt = float(dt)
        self.bias_clip = float(bias_clip)

        if self.state_shots <= 1:
            raise ValueError("state_shots must be > 1.")

    def _distort_gate(self, gate):
        """Apply the explicit runtime-backend noise model to one gate."""
        return self.noise_model.apply(gate)

    def effective_geometric_state(self, circuit):
        """
        Deterministic geometric state after applying the backend distortion.

        This is not used as the calibration observation.  Calibration uses
        stochastic p-bit realizations below.
        """
        state = M.Z_AXIS.copy()

        for gate in circuit.gates:
            effective_gate = self._distort_gate(gate)
            state = apply_gate_ideal(state, effective_gate)

        return M.point(state)

    def _realize_state_once(self, target_state, seed):
        """
        Realize one Bloch point statistically using three independent p-bits.
        """
        target_state = M.point(target_state)

        circuit = PCircuit(3)
        circuit.J = np.zeros((3, 3))

        clipped = np.clip(
            target_state,
            -self.bias_clip,
            self.bias_clip,
        )
        circuit.h = np.arctanh(clipped)

        solver = CaSuDaSolver(
            Nt=self.state_nt,
            dt=self.dt,
            i0=1.0,
            expected_mean=0.0,
            seed=int(seed),
        )

        # CaSuDaSolver multi-shot path:
        # samples.shape == (Nt, n_shots, n_pbits)
        samples = solver.solve(
            circuit,
            n_shots=self.state_shots,
        )

        assert samples.ndim == 3, (
            f"expected (Nt, n_shots, n_pbits), got {samples.shape}"
        )

        raw_mean = np.mean(
            samples[self.state_burn_in:, :, :],
            axis=(0, 1),
        )

        return raw_mean, M.point(raw_mean)

    def run(
        self,
        circuit,
        n_realizations=N_REALIZATIONS,
        seed_base=BASE_REALIZATION_SEED,
    ):
        """
        Execute a circuit and estimate its output state from p-bit realizations.
        """
        effective_state = self.effective_geometric_state(circuit)

        raw_vectors = []
        points = []

        for index in range(int(n_realizations)):
            raw, point = self._realize_state_once(
                effective_state,
                seed=int(seed_base) + index,
            )

            raw_vectors.append(raw)
            points.append(point)

        raw_vectors = np.asarray(raw_vectors)
        points = np.asarray(points)

        intrinsic_mean = riemannian_mean(points)

        return {
            "effective_state": effective_state,
            "raw_vectors": raw_vectors,
            "points": points,
            "state": intrinsic_mean,
            "raw_radius_mean": float(
                np.mean(np.linalg.norm(raw_vectors, axis=1))
            ),
            "dispersion_fs": float(
                np.mean(
                    [
                        M.fs_distance(intrinsic_mean, point)
                        for point in points
                    ]
                )
            ),
        }

    def sample_measurement(
        self,
        state,
        axis,
        shots=MEASUREMENT_SHOTS,
        seed=MEASUREMENT_SEED,
    ):
        """
        Sample a projective measurement with one p-bit.

        The p-bit expectation is set to the Born expectation r dot n.
        """
        shots = int(shots)

        if shots <= 0:
            raise ValueError("shots must be positive.")

        expectation = M.born_expectation(state, axis)

        circuit = PCircuit(1)
        circuit.J = np.zeros((1, 1))

        clipped = float(
            np.clip(
                expectation,
                -self.bias_clip,
                self.bias_clip,
            )
        )
        circuit.h = np.array([np.arctanh(clipped)])

        solver = CaSuDaSolver(
            Nt=MEASUREMENT_NT,
            dt=self.dt,
            i0=1.0,
            expected_mean=0.0,
            seed=int(seed),
        )

        solver_shots = max(shots, 2)
        samples = solver.solve(
            circuit,
            n_shots=solver_shots,
        )

        assert samples.ndim == 3, (
            f"expected (Nt, n_shots, n_pbits), got {samples.shape}"
        )

        terminal = np.asarray(samples[-1, :shots, 0])
        plus = int(np.sum(terminal > 0))
        minus = shots - plus

        return {
            "+": plus,
            "-": minus,
            "shots": shots,
            "empirical_p_plus": plus / shots,
            "target_p_plus": M.born_probability(state, axis),
        }


# ============================================================================
# Calibration
# ============================================================================

class CalibrationEvaluator:
    """
    Evaluate candidate commanded parameters using the stochastic backend.

    Candidate evaluations are cached.  Common seeds are used for all
    candidates so the optimization is not comparing unrelated random draws.
    """

    def __init__(
        self,
        backend,
        template_circuit,
        target_state,
        theta_label="theta",
        phi_label="phi",
    ):
        self.backend = backend
        self.template_circuit = template_circuit.copy()
        self.target_state = M.point(target_state)

        self.theta_label = theta_label
        self.phi_label = phi_label

        self.cache = {}
        self.execution_count = 0

    @staticmethod
    def _key(theta, phi):
        return (
            round(float(theta), 12),
            round(float(phi), 12),
        )

    def evaluate(self, theta, phi):
        key = self._key(theta, phi)

        if key in self.cache:
            return self.cache[key]

        candidate = self.template_circuit.copy()
        candidate.set_parameter(self.theta_label, theta)
        candidate.set_parameter(self.phi_label, phi)

        execution = self.backend.run(
            candidate,
            n_realizations=N_REALIZATIONS,
            seed_base=BASE_REALIZATION_SEED,
        )

        state = execution["state"]

        result = {
            "theta": float(theta),
            "phi": float(phi),
            "circuit": candidate,
            "execution": execution,
            "distance": M.fs_distance(state, self.target_state),
            "fidelity": M.fidelity(state, self.target_state),
        }

        self.cache[key] = result
        self.execution_count += 1

        return result


def calibrate_two_parameters(
    evaluator,
    theta_initial,
    phi_initial,
):
    """
    Simple derivative-free coordinate search.

    The objective is the Fubini-Study distance between the target ideal state
    and the Riemannian mean of repeated p-bit executions.
    """
    theta = float(theta_initial)
    phi = float(phi_initial)

    step_theta = INITIAL_CALIBRATION_STEP
    step_phi = INITIAL_CALIBRATION_STEP

    history = []

    current = evaluator.evaluate(theta, phi)

    for round_index in range(MAX_CALIBRATION_ROUNDS):
        candidates = [
            (theta, phi),
            (theta + step_theta, phi),
            (theta - step_theta, phi),
            (theta, phi + step_phi),
            (theta, phi - step_phi),
        ]

        results = [
            evaluator.evaluate(candidate_theta, candidate_phi)
            for candidate_theta, candidate_phi in candidates
        ]

        best = min(results, key=lambda result: result["distance"])

        if best["distance"] + 1e-15 < current["distance"]:
            theta = best["theta"]
            phi = best["phi"]
            current = best
        else:
            step_theta *= 0.5
            step_phi *= 0.5

        history.append(
            {
                "round": round_index,
                "theta": theta,
                "phi": phi,
                "distance": current["distance"],
                "fidelity": current["fidelity"],
                "step_theta": step_theta,
                "step_phi": step_phi,
            }
        )

        if max(step_theta, step_phi) < MIN_CALIBRATION_STEP:
            break

    return {
        "theta": theta,
        "phi": phi,
        "result": current,
        "history": history,
    }


# ============================================================================
# Reporting helpers
# ============================================================================

def print_state_comparison(
    label,
    state,
    target_state,
):
    print(label)
    print("-" * len(label))
    print(f"state      = {np.array2string(state, precision=7)}")
    print(
        f"d_FS       = {M.fs_distance(state, target_state):.8f}"
    )
    print(
        f"fidelity   = {M.fidelity(state, target_state):.9f}"
    )
    print()


def print_calibration_history(history):
    print("CALIBRATION")
    print("===========")
    print(
        f"{'round':>5s} "
        f"{'theta_cmd':>11s} "
        f"{'phi_cmd':>11s} "
        f"{'d_FS':>11s} "
        f"{'fidelity':>12s} "
        f"{'step':>9s}"
    )
    print("-" * 68)

    for row in history:
        print(
            f"{row['round']:5d} "
            f"{row['theta']:11.6f} "
            f"{row['phi']:11.6f} "
            f"{row['distance']:11.7f} "
            f"{row['fidelity']:12.9f} "
            f"{max(row['step_theta'], row['step_phi']):9.6f}"
        )

    print()


def measurement_table(
    backend,
    target_state,
    before_state,
    after_state,
):
    print("MEASUREMENT CHECK")
    print("=================")
    print(
        f"{'axis':>4s} "
        f"{'ideal P(+)':>12s} "
        f"{'before P(+)':>13s} "
        f"{'after P(+)':>12s} "
        f"{'after sampled':>14s}"
    )
    print("-" * 63)

    axes = (
        ("X", M.X_AXIS),
        ("Y", M.Y_AXIS),
        ("Z", M.Z_AXIS),
    )

    for axis_index, (name, axis) in enumerate(axes):
        ideal_p = M.born_probability(target_state, axis)
        before_p = M.born_probability(before_state, axis)
        after_p = M.born_probability(after_state, axis)

        sampled = backend.sample_measurement(
            after_state,
            axis,
            shots=MEASUREMENT_SHOTS,
            seed=MEASUREMENT_SEED + axis_index,
        )

        print(
            f"{name:>4s} "
            f"{ideal_p:12.6f} "
            f"{before_p:13.6f} "
            f"{after_p:12.6f} "
            f"{sampled['empirical_p_plus']:14.6f}"
        )

    print()


# ============================================================================
# Main demo
# ============================================================================

def main():
    print("Demo 2: p-bit quantum-circuit calibration + execution")
    print("======================================================")
    print()

    # ----------------------------------------------------------------------
    # 1. Define the intended quantum circuit in the normal forward direction.
    # ----------------------------------------------------------------------

    intended_circuit = QuantumCircuit()
    intended_circuit.h()
    intended_circuit.rz(TARGET_THETA, label="theta")
    intended_circuit.ry(TARGET_PHI, label="phi")
    intended_circuit.t()

    print("INTENDED CIRCUIT")
    print("================")
    print(intended_circuit)
    print()

    # ----------------------------------------------------------------------
    # 2. Compute the ideal target state.
    # ----------------------------------------------------------------------

    target_state = execute_ideal(intended_circuit)

    print("Ideal target state:")
    print(np.array2string(target_state, precision=7))
    print()

    # ----------------------------------------------------------------------
    # 3. Build an explicit noise model and attach it to the runtime backend.
    #
    # This mirrors the usual simulator pattern:
    #
    #     circuit -> backend(noise_model=...) -> result
    #
    # The ideal circuit itself remains unchanged.
    # ----------------------------------------------------------------------

    noise_model = NoiseModel()
    noise_model.add_gate_angle_error(
        "rz",
        offset=RZ_NOISE_OFFSET,
    )
    noise_model.add_gate_angle_error(
        "ry",
        gain=RY_NOISE_GAIN,
        offset=RY_NOISE_OFFSET,
    )

    backend = PBitRuntimeBackend(
        noise_model=noise_model,
    )

    print("NOISE MODEL")
    print("===========")
    for line in noise_model.describe():
        print(line)
    print(
        "(These coherent gate errors are deliberately injected for the demo.)"
    )
    print()

    # ----------------------------------------------------------------------
    # 4. Execute the intended circuit before calibration.
    # ----------------------------------------------------------------------

    before = backend.run(
        intended_circuit,
        n_realizations=N_REALIZATIONS,
        seed_base=BASE_REALIZATION_SEED,
    )

    before_state = before["state"]

    print_state_comparison(
        "BEFORE CALIBRATION",
        before_state,
        target_state,
    )

    print(
        f"mean p-bit realization dispersion d_FS = "
        f"{before['dispersion_fs']:.8f}"
    )
    print(
        f"mean raw Bloch radius                  = "
        f"{before['raw_radius_mean']:.8f}"
    )
    print()

    # ----------------------------------------------------------------------
    # 5. Calibrate the programmable Rz/Ry parameters.
    # ----------------------------------------------------------------------

    evaluator = CalibrationEvaluator(
        backend=backend,
        template_circuit=intended_circuit,
        target_state=target_state,
        theta_label="theta",
        phi_label="phi",
    )

    calibration = calibrate_two_parameters(
        evaluator,
        theta_initial=TARGET_THETA,
        phi_initial=TARGET_PHI,
    )

    print_calibration_history(calibration["history"])

    calibrated_theta = calibration["theta"]
    calibrated_phi = calibration["phi"]

    calibrated_circuit = intended_circuit.copy()
    calibrated_circuit.set_parameter("theta", calibrated_theta)
    calibrated_circuit.set_parameter("phi", calibrated_phi)

    # ----------------------------------------------------------------------
    # 6. Re-execute the calibrated circuit.
    # ----------------------------------------------------------------------

    after = backend.run(
        calibrated_circuit,
        n_realizations=N_REALIZATIONS,
        seed_base=BASE_REALIZATION_SEED,
    )

    after_state = after["state"]

    print("CALIBRATED CIRCUIT")
    print("==================")
    print(calibrated_circuit)
    print()

    print_state_comparison(
        "AFTER CALIBRATION",
        after_state,
        target_state,
    )

    # ----------------------------------------------------------------------
    # 7. Summary.
    # ----------------------------------------------------------------------

    before_distance = M.fs_distance(before_state, target_state)
    after_distance = M.fs_distance(after_state, target_state)

    before_fidelity = M.fidelity(before_state, target_state)
    after_fidelity = M.fidelity(after_state, target_state)

    print("FINAL COMPARISON")
    print("================")
    print(
        f"intended theta             = {TARGET_THETA:.6f}"
    )
    print(
        f"calibrated commanded theta = {calibrated_theta:.6f}"
    )
    print(
        f"intended phi               = {TARGET_PHI:.6f}"
    )
    print(
        f"calibrated commanded phi   = {calibrated_phi:.6f}"
    )
    print()

    print(
        f"before d_FS                = {before_distance:.8f}"
    )
    print(
        f"after  d_FS                = {after_distance:.8f}"
    )
    print(
        f"before fidelity            = {before_fidelity:.9f}"
    )
    print(
        f"after  fidelity            = {after_fidelity:.9f}"
    )

    if after_distance > 0.0:
        print(
            f"FS-error reduction factor  = "
            f"{before_distance / after_distance:.3f}x"
        )

    print(
        f"candidate runtime-backend executions = "
        f"{evaluator.execution_count}"
    )
    print()

    # The ideal programmed circuit with the calibrated parameters is not
    # expected to equal the target.  The calibration is compensating the
    # imperfect backend, not recovering the original ideal parameters.
    ideal_calibrated_state = execute_ideal(calibrated_circuit)

    print(
        "Ideal simulator d_FS at the calibrated commanded parameters = "
        f"{M.fs_distance(ideal_calibrated_state, target_state):.8f}"
    )
    print(
        "This can remain non-zero because the calibrated commands are chosen "
        "for the imperfect p-bit backend."
    )
    print()

    # ----------------------------------------------------------------------
    # 8. Measurement verification.
    # ----------------------------------------------------------------------

    measurement_table(
        backend,
        target_state,
        before_state,
        after_state,
    )

    print("INTERPRETATION")
    print("==============")
    print(
        "The circuit is first executed normally on a stochastic p-bit backend. "
        "The resulting CP^1 state is estimated with a Riemannian mean. "
        "Fubini-Study distance to the ideal target is then used as the "
        "calibration objective. The corrected circuit is finally executed "
        "again on the same backend."
    )
    print()
    print(
        "The key result is not that the calibrated command angles reproduce "
        "the ideal command angles. The goal is that the calibrated p-bit "
        "backend output moves closer to the desired quantum state."
    )


if __name__ == "__main__":
    main()
