"""

This is a single-qubit research demo. The objective of the demo is to make
a link between p-bit probabilistic computing, Riemannian geometry, and the
quantum Bloch sphere. It explores this combination for a potential future 
quantum simulator.

A pure single-qubit state is treated first as a point on the Riemannian
manifold CP^1, represented by the Bloch sphere S^2 with the Fubini-Study
metric. Three independent p-bits statistically realize the Bloch
coordinates, gates act as CP^1 isometries, and one additional p-bit samples
projective measurements.

So the the qubit lives on CP^1, p-bits realize that state statistically, 
gates move it geometrically, and measurements are sampled probabilistically.

GATE_SEQUENCE represents a small quantum circuit test. 

ManifoldPBitQubit      # single-qubit simulator object
├── CP1BlochManifold   # geometry
└── PBitCP1Backend     # realization using p-bits

The demo reports:
  * the raw p-bit Bloch-vector norm before projection back to CP^1;
  * an effective pre-projection purity indicator (1 + |raw_means|^2) / 2;
  * Fubini-Study reconstruction error and target-state fidelity;
  * tangent-space error decomposition into local polar/azimuthal components;
  * mean and standard deviation over multiple independent p-bit trials;
  * Born-rule measurement counts and projective-collapse behavior.

The raw norm/purity values are diagnostics of the stochastic state estimate;
finite sampling and finite p-bit bias can move the raw estimate away from the
unit sphere, so they should not by themselves be interpreted as physical
mixed-state tomography.

Results:
    The quantum manifold  p-bit representation is numerically stable and reproduces
    single-qubit circuit behavior well.
    
The question is how far we can go from here.

Could coupled p-bits provide a stochastic representation of joint probability
distributions and correlations associated with points on higher-dimensional
quantum-state manifolds?

"""

import numpy as np

from p_kit.psl import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver


# ---------------------------------------------------------------------------
# p-bit numerical parameters
# ---------------------------------------------------------------------------

BIAS_CLIP = 0.995

STATE_NT = 12_000
STATE_BURN_IN = 2_000
STATE_SHOTS = 32

MEASURE_NT = 1_000

DT = 0.02

N_TRIALS = 8          # multi-seed statistics; each trial re-runs the whole
                      # 4-gate circuit, so cost scales linearly with this.
BASE_SEED = 42

GATE_SEQUENCE = [ # quantum circuit
    ("h", ()),
    ("t", ()),
    ("ry", (np.pi / 3.0,)),
    ("rz", (-np.pi / 5.0,)),
]


# ---------------------------------------------------------------------------
# Riemannian manifold: CP^1 represented by the unit Bloch sphere
# ---------------------------------------------------------------------------

class CP1BlochManifold:
    """
    Pure single-qubit state manifold CP^1 represented as S^2.
    Riemannian geometry is implemented cleanly with NumPy here.
    """

    X_AXIS = np.array([1.0, 0.0, 0.0])
    Y_AXIS = np.array([0.0, 1.0, 0.0])
    Z_AXIS = np.array([0.0, 0.0, 1.0])

    @staticmethod
    def point(r):
        r = np.asarray(r, dtype=float)
        norm = np.linalg.norm(r)
        if norm == 0.0:
            raise ValueError("The zero vector cannot define a pure qubit state.")
        return r / norm

    @classmethod
    def from_angles(cls, theta, phi):
        return cls.point(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

    @classmethod
    def angles(cls, point):
        r = cls.point(point)
        theta = float(np.arccos(np.clip(r[2], -1.0, 1.0)))
        phi = float(np.arctan2(r[1], r[0]))
        return theta, phi

    @classmethod
    def tangent(cls, point, ambient_vector):
        p = cls.point(point)
        v = np.asarray(ambient_vector, dtype=float)
        return v - np.dot(v, p) * p

    @classmethod
    def exp_map(cls, point, tangent_vector):
        p = cls.point(point)
        v = cls.tangent(p, tangent_vector)
        theta = np.linalg.norm(v)
        if theta < 1e-15:
            return p.copy()
        return cls.point(np.cos(theta) * p + np.sin(theta) * (v / theta))

    @classmethod
    def log_map(cls, point_a, point_b):
        p = cls.point(point_a)
        q = cls.point(point_b)
        dot = float(np.clip(np.dot(p, q), -1.0, 1.0))
        angle = float(np.arccos(dot))
        if angle < 1e-15:
            return np.zeros(3)
        if np.pi - angle < 1e-12:
            raise ValueError("Log map is not unique for antipodal states.")
        direction = q - dot * p
        direction /= np.linalg.norm(direction)
        return angle * direction

    @classmethod
    def geodesic(cls, point_a, point_b, t):
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must lie in [0, 1].")
        p = cls.point(point_a)
        return cls.exp_map(p, t * cls.log_map(p, point_b))

    @classmethod
    def distance(cls, point_a, point_b):
        p = cls.point(point_a)
        q = cls.point(point_b)
        dot = float(np.clip(np.dot(p, q), -1.0, 1.0))
        return 0.5 * float(np.arccos(dot))

    @classmethod
    def fidelity(cls, point_a, point_b):
        p = cls.point(point_a)
        q = cls.point(point_b)
        return float(np.clip(0.5 * (1.0 + np.dot(p, q)), 0.0, 1.0))

    @classmethod
    def rotate(cls, point, axis, angle):
        r = cls.point(point)
        n = cls.point(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        rotated = c * r + s * np.cross(n, r) + (1.0 - c) * np.dot(n, r) * n
        return cls.point(rotated)

    @classmethod
    def born_expectation(cls, point, axis):
        r = cls.point(point)
        n = cls.point(axis)
        return float(np.clip(np.dot(r, n), -1.0, 1.0))

    @classmethod
    def born_probability(cls, point, axis):
        expectation = cls.born_expectation(point, axis)
        return 0.5 * (1.0 + expectation)

def tangent_frame(theta, phi):
    """Orthonormal tangent basis (e_theta, e_phi) at Bloch angles (theta, phi)."""
    e_theta = np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return e_theta, e_phi

def tangent_decompose(manifold, base_point, other_point):
    """
    Decompose the Riemannian reconstruction error into the local
    (e_theta, e_phi) tangent basis.

    Returns (d_theta, d_phi, total_angle), where total_angle is the
    Bloch-sphere tangent norm and equals 2 * d_FS.

    At the Bloch poles, spherical tangent components are undefined,
    so d_theta and d_phi are returned as NaN.
    """
    tangent = manifold.log_map(base_point, other_point)
    norm = np.linalg.norm(tangent)

    if norm < 1e-12:
        return 0.0, 0.0, 0.0

    theta, phi = manifold.angles(base_point)

    # Spherical-coordinate tangent directions are undefined at the poles.
    if abs(np.sin(theta)) < 1e-10:
        return np.nan, np.nan, float(norm)

    e_theta, e_phi = tangent_frame(theta, phi)

    return (
        float(np.dot(tangent, e_theta)),
        float(np.dot(tangent, e_phi)),
        float(norm),
    )

# ---------------------------------------------------------------------------
# p-kit realization and measurement backend
# ---------------------------------------------------------------------------

class PBitCP1Backend:
    """Realize CP^1 points using p-bit expectation values and sample measurements."""

    def __init__(
        self,
        bias_clip=BIAS_CLIP,
        state_nt=STATE_NT,
        state_burn_in=STATE_BURN_IN,
        state_shots=STATE_SHOTS,
        measure_nt=MEASURE_NT,
        dt=DT,
        seed=1234,
    ):
        self.bias_clip = float(bias_clip)
        self.state_nt = int(state_nt)
        self.state_burn_in = int(state_burn_in)
        self.state_shots = int(state_shots)
        self.measure_nt = int(measure_nt)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

    def _next_seed(self):
        return int(self.rng.integers(0, 2**31 - 1))

    def _build_independent_circuit(self, target_means):
        target_means = np.asarray(target_means, dtype=float).reshape(-1)
        circuit = PCircuit(len(target_means))
        circuit.J = np.zeros((len(target_means), len(target_means)))
        clipped = np.clip(target_means, -self.bias_clip, self.bias_clip)
        circuit.h = np.arctanh(clipped)
        return circuit

    def realize(self, manifold, target_point):
        """
        Realize a pure-state manifold point with three p-bits.

        Returns (raw_means, reconstructed): raw_means is the direct p-bit
        expectation estimate (may have norm < 1 -- see module docstring),
        reconstructed is its projection onto CP^1.
        """
        target = manifold.point(target_point)
        circuit = self._build_independent_circuit(target)

        solver = CaSuDaSolver(
            Nt=self.state_nt, dt=self.dt, i0=1.0, expected_mean=0.0,
            seed=self._next_seed(),
        )
        samples = solver.solve(circuit, n_shots=self.state_shots)

        assert samples.ndim == 3, (
            f"expected (Nt, n_shots, n_pbits) samples, got shape {samples.shape}"
        )
        raw_means = np.mean(samples[self.state_burn_in:, :, :], axis=(0, 1))
        reconstructed = manifold.point(raw_means)
        return raw_means, reconstructed

    def sample_measurement(self, manifold, point, axis, shots=1024):
        """Sample a projective measurement using p-bit solver shots."""
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive.")
    
        expectation = manifold.born_expectation(point, axis)
        circuit = self._build_independent_circuit([expectation])
    
        solver = CaSuDaSolver(
            Nt=self.measure_nt,
            dt=self.dt,
            i0=1.0,
            expected_mean=0.0,
            seed=self._next_seed(),
        )
    
        # CaSuDaSolver returns a 3-D trajectory for n_shots > 1:
        # (Nt, n_shots, n_pbits).
        # Use at least 2 solver shots to avoid the special n_shots == 1 return.
        solver_shots = max(shots, 2)
        samples = solver.solve(circuit, n_shots=solver_shots)
    
        assert samples.ndim == 3, (
            f"expected (Nt, n_shots, n_pbits), got {samples.shape}"
        )
    
        # One terminal p-bit value per logical measurement shot.
        terminal = np.asarray(samples[-1, :shots, 0])
    
        plus = int(np.sum(terminal > 0))
    
        return {
            "+": plus,
            "-": shots - plus,
            "shots": shots,
            "empirical_p_plus": plus / shots,
            "target_p_plus": 0.5 * (1.0 + expectation),
            "expectation": expectation,
        }

    def sample_one(self, manifold, point, axis):
        result = self.sample_measurement(manifold, point, axis, shots=1)
        return +1 if result["+"] == 1 else -1


# ---------------------------------------------------------------------------
# Single-qubit manifold simulator
# ---------------------------------------------------------------------------

class ManifoldPBitQubit:
    H_AXIS = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)

    def __init__(self, backend=None):
        self.M = CP1BlochManifold()
        self.backend = backend if backend is not None else PBitCP1Backend()
        self.history = []
        self.reset()

    def reset(self):
        north = self.M.Z_AXIS.copy()
        self.ideal_state = north.copy()
        self.raw_means, self.state = self.backend.realize(self.M, north)
        self.history = []
        self._record("reset |0>")
        return self

    def prepare(self, theta, phi):
        target = self.M.from_angles(theta, phi)
        self.ideal_state = target.copy()
        self.raw_means, self.state = self.backend.realize(self.M, target)
        self.history = []
        self._record(f"prepare(theta={theta:.4f}, phi={phi:.4f})")
        return self

    def _record(self, gate):
        raw_norm = float(np.linalg.norm(self.raw_means))
        d_theta, d_phi, _tangent_norm = tangent_decompose(
            self.M, self.ideal_state, self.state
        )
        self.history.append(
            {
                "gate": gate,
                "ideal": self.ideal_state.copy(),
                "state": self.state.copy(),
                "raw": self.raw_means.copy(),
                "raw_norm": raw_norm,
                "purity": 0.5 * (1.0 + raw_norm ** 2),
                "fidelity": self.M.fidelity(self.ideal_state, self.state),
                "fs_error": self.M.distance(self.ideal_state, self.state),
                "tangent_theta": d_theta,
                "tangent_phi": d_phi,
            }
        )

    def _rotation(self, axis, angle, name):
        self.ideal_state = self.M.rotate(self.ideal_state, axis, angle)
        target_from_pbits = self.M.rotate(self.state, axis, angle)
        self.raw_means, self.state = self.backend.realize(self.M, target_from_pbits)
        self._record(name)
        return self

    def rx(self, angle):
        return self._rotation(self.M.X_AXIS, angle, f"Rx({angle:.4f})")

    def ry(self, angle):
        return self._rotation(self.M.Y_AXIS, angle, f"Ry({angle:.4f})")

    def rz(self, angle):
        return self._rotation(self.M.Z_AXIS, angle, f"Rz({angle:.4f})")

    def x(self):
        return self._rotation(self.M.X_AXIS, np.pi, "X")

    def y(self):
        return self._rotation(self.M.Y_AXIS, np.pi, "Y")

    def z(self):
        return self._rotation(self.M.Z_AXIS, np.pi, "Z")

    def h(self):
        return self._rotation(self.H_AXIS, np.pi, "H")

    def s(self):
        return self._rotation(self.M.Z_AXIS, np.pi / 2.0, "S")

    def t(self):
        return self._rotation(self.M.Z_AXIS, np.pi / 4.0, "T")

    def bloch_vector(self):
        return self.state.copy()

    def ideal_bloch_vector(self):
        return self.ideal_state.copy()

    def fidelity_to_ideal(self):
        return self.M.fidelity(self.ideal_state, self.state)

    def fs_error(self):
        return self.M.distance(self.ideal_state, self.state)

    def sample_counts(self, axis="Z", shots=1024):
        axis_vector = self._axis_vector(axis)
        return self.backend.sample_measurement(self.M, self.state, axis_vector, shots=shots)

    def measure_once(self, axis="Z", collapse=True):
        axis_vector = self._axis_vector(axis)
        outcome = self.backend.sample_one(self.M, self.state, axis_vector)
        if collapse:
            collapsed = outcome * self.M.point(axis_vector)
            self.ideal_state = collapsed.copy()
            self.raw_means, self.state = self.backend.realize(self.M, collapsed)
            self._record(f"measure {axis} -> {outcome:+d}")
        return outcome

    def _axis_vector(self, axis):
        if isinstance(axis, str):
            name = axis.upper()
            if name == "X":
                return self.M.X_AXIS
            if name == "Y":
                return self.M.Y_AXIS
            if name == "Z":
                return self.M.Z_AXIS
            raise ValueError("axis string must be 'X', 'Y', or 'Z'.")
        return self.M.point(axis)

    def print_history(self):
        print("gate/state history (single trial)")
        print("==================================")
        header = (
            f"{'operation':18s} {'fidelity':>10s} {'d_FS':>10s} "
            f"{'purity':>8s} {'|raw|':>8s} {'d_theta':>9s} {'d_phi':>9s}"
        )
        print(header)
        print("-" * len(header))
        for row in self.history:
            print(
                f"{row['gate']:18s} "
                f"{row['fidelity']:10.6f} "
                f"{row['fs_error']:10.6f} "
                f"{row['purity']:8.5f} "
                f"{row['raw_norm']:8.5f} "
                f"{row['tangent_theta']:9.5f} "
                f"{row['tangent_phi']:9.5f}"
            )
        print(
            "  (purity = effective pre-projection indicator "
            "(1+|raw_means|^2)/2; d_theta/d_phi = tangent components of the "
            "ideal -> reconstructed error, in the local polar/azimuthal frame)"
        )

def run_single_qubit_trial(seed):
    q = ManifoldPBitQubit(backend=PBitCP1Backend(seed=seed))
    for name, args in GATE_SEQUENCE:
        getattr(q, name)(*args)
    return q


def summarize_trials(qubits):
    n_steps = len(qubits[0].history)
    print(f"aggregate statistics over {len(qubits)} independent p-bit trials")
    print("=" * 78)
    header = (
        f"{'step':18s} {'fidelity':>12s} {'d_FS':>12s} "
        f"{'purity':>12s} {'|d_theta|':>12s} {'|d_phi|':>12s}"
    )
    print(header)
    print("-" * len(header))
    for i in range(n_steps):
        gate_name = qubits[0].history[i]["gate"]
        fid = np.array([q.history[i]["fidelity"] for q in qubits])
        dfs = np.array([q.history[i]["fs_error"] for q in qubits])
        pur = np.array([q.history[i]["purity"] for q in qubits])
        dth = np.array([abs(q.history[i]["tangent_theta"]) for q in qubits])
        dph = np.array([abs(q.history[i]["tangent_phi"]) for q in qubits])

        print(
            f"{gate_name:18s} {fid.mean():12.6f} {dfs.mean():12.6f} "
            f"{pur.mean():12.6f} {dth.mean():12.6f} {dph.mean():12.6f}"
        )
        print(
            f"{'  (std)':18s} {fid.std():12.6f} {dfs.std():12.6f} "
            f"{pur.std():12.6f} {dth.std():12.6f} {dph.std():12.6f}"
        )
    print()


def main(n_trials=N_TRIALS):
    print("Single-qubit manifold-first p-bit simulator")
    print("=====================================================")
    print(f"Running {n_trials} independent trials of H, T, Ry(pi/3), Rz(-pi/5)...")
    print()

    trials = [run_single_qubit_trial(seed=BASE_SEED + i) for i in range(n_trials)]

    trials[0].print_history()
    print()
    summarize_trials(trials)

    q = trials[0]
    print("final state (trial 0)")
    print("======================")
    print("ideal Bloch vector :", q.ideal_bloch_vector())
    print("p-bit Bloch vector :", q.bloch_vector())
    print(f"final fidelity     : {q.fidelity_to_ideal():.6f}")
    print(f"final d_FS error   : {q.fs_error():.6f}")
    print()

    shots = 2048
    print(f"measurement counts ({shots} shots, trial 0 final state)")
    print("=========================================================")
    print("axis     ideal P(+)   manifold P(+)   p-bit P(+)      counts")
    print("-" * 66)
    for axis_name, axis_vector in [
        ("X", q.M.X_AXIS), ("Y", q.M.Y_AXIS), ("Z", q.M.Z_AXIS),
        ("XY", np.array([1.0, 1.0, 0.0])),
    ]:
        ideal_p = q.M.born_probability(q.ideal_bloch_vector(), axis_vector)
        manifold_p = q.M.born_probability(q.bloch_vector(), axis_vector)
        result = q.sample_counts(axis_vector, shots=shots)
        print(
            f"{axis_name:4s}     {ideal_p:10.4f}   {manifold_p:13.4f}   "
            f"{result['empirical_p_plus']:10.4f}     "
            f"+:{result['+']:4d}  -:{result['-']:4d}"
        )

    print()
    print("projective-collapse demonstration")
    print("==================================")
    before = q.bloch_vector()
    outcome = q.measure_once("Z", collapse=True)
    after = q.bloch_vector()
    print("state before Z measurement :", before)
    print(f"observed outcome           : {outcome:+d}")
    print("post-measurement p-bit state:", after)
    print(f"fidelity to collapsed pole : {q.fidelity_to_ideal():.6f}")
    repeat = q.sample_counts("Z", shots=shots)
    print(
        f"repeat Z measurement       : +:{repeat['+']}  -:{repeat['-']} "
        f"(P(+)= {repeat['empirical_p_plus']:.4f})"
    )
    print()

if __name__ == "__main__":
    main()
