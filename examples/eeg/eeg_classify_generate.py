"""
This is an EEG P300 classification and generation demo. It is a first proof
of concept towards heavier use of probabilistic circuits. The EEG generation is
where probabilistic sampling is used.

I have noticed that the following algorithm is both simple and provides very
good classification results. In this demo it is implemented as FlatLR:
it applies 1-18 Hz band-pass filtering, resamples at 64 Hz and applies Logistic
Regression. It is simple, fast, yet the results are surprisingly good. So it
was a good candidate to encode on a p-kit PCircuit.

The demo runs in 3 modes:
- classify - compares classical FlatLR with JointPBit, a hybrid Fourier + PCircuit version
- generate - uses the same PCircuit architecture for P300 generation, independently for each subject
- both - demonstrates classification and generation

Training:
classical parameter estimation
-> compile the learned parameters into J, h of a probabilistic PCircuit

Classification:
observed EEG p-bits
-> exact conditional inference P(class | EEG)
-> no stochastic solver needed

Generation:
class p-bit clamped
-> stochastic PCircuit sampling
-> BitPlane + BlockGibbs

Results:
- Classification results are quite good. The demo tests 5 P300 datasets from MOABB.
- Generation is meaningful, but less impressive. This might partly be because there is
  limited data per subject and the generation method is not fully optimized.
- Processing time is reasonable.

Notes:
- JointPBit first transforms EEG into 1-18 Hz Fourier coefficients. This part
  is deterministic and runs outside the PCircuit.
- The Fourier coefficients are quantized into fixed-point p-bit variables and
  represented inside the PCircuit.
- The PCircuit is not only a final classifier. It contains a probabilistic
  model of the EEG spectral coefficients, their dependencies and their
  coupling to the class p-bit.
- Classification fixes the EEG p-bits and computes the conditional probability
  of the class p-bit.
- Generation does the reverse: the P300 class p-bit is fixed and the EEG
  coefficient p-bits are sampled from the same PCircuit.
- FixedPointQuadratic represents the continuous spectral distribution as a
  p-bit quadratic energy model.
- BitPlaneCaSuDaSolver is used to obtain an initial p-bit state for generation.
- BlockGibbsSolver then updates all bits representing one Fourier coefficient
  jointly. With 6 bits per coefficient, it evaluates the 64 possible states
  and samples from their exact conditional distribution.
- The generated spectral coefficients are finally transformed back to the
  time domain to obtain synthetic P300 EEG.
- Therefore the demo is hybrid: the Fourier transform/reconstruction is
  classical, while the probabilistic representation, class coupling and
  conditional generation are implemented with p-kit PCircuits.
- The P300 EEG generation is currently not fully optimized.

Result1:
    BNCI2014-008, 8 subjects:

    Classification: JointPBit 0.8603 ± 0.0461 AUC vs FlatLR 0.8553 ± 0.0480.
    P300 generation: mean ERP r=0.7102 vs real-real 0.8199.
    Global spectral structure: r=0.4712 vs real-real 0.7729.
    Precision-edge structure: r=0.1673 vs real-real 0.8711.
    2 minutes
    
Result 2:

    5 datasets, subjects 135:

    CLASSIFICATION — MOABB WithinSession
      JointPBit : AUC=0.8551 ± 0.0770
      FlatLR    : AUC=0.8632 ± 0.0713
    P300 GENERATION — held-out real EEG
      Mean ERP          : r=0.3564   real-real=0.6434
      Global structure  : r=0.1311   real-real=0.5141
      Precision edges   : r=0.2789   real-real=0.7996

"""

import copy, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from moabb.datasets import BNCI2014_008, BNCI2014_009, BNCI2015_003, BI2014a, BI2015a
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from p_kit.backends import NumpyBackend
from p_kit.psl.p_circuit import PCircuit
from p_kit.psl.fixed_point_quadratic import FixedPointQuadratic
from p_kit.solver.bitplane_csd_solver import BitPlaneCaSuDaSolver
from p_kit.solver.block_gibbs_solver import BlockGibbsSolver

warnings.filterwarnings("ignore")

MODE = "both"
#DATASETS = [BNCI2014_008(), BNCI2014_009(), BNCI2015_003(), BI2014a(), BI2015a()]
DATASETS = [BNCI2014_008()]
SUBJECTS = None
SEED, TEST_SIZE = 42, .25
N_JOBS, N_SPLITS = 1, 5
FS_OUT, BITS, CLIP = 64, 6, 4.
PRECISION_EDGES_PER_COEFF, PRECISION_SCALE = .70, 4.
GAMMA = 8.
NT, BLOCK_SWEEPS = 16, 4
N_GENERATED = 1000
PLOT = False

class NonFilterP300(P300):
    def __init__(self):
        super().__init__(fmin=1, fmax=24, resample=None)
    def _get_raw_pipelines(self):
        return [None]

def flatten(X):
    return X.reshape(len(X), -1)

def get_fs(ds):
    try:
        return float(ds.metadata.acquisition.sampling_rate)
    except Exception:
        s = ds.subject_list[0]
        d = ds.get_data(subjects=[s])[s]
        return float(next(iter(next(iter(d.values())).values())).info["sfreq"])

def fourier_basis(n, fs):
    t, Q, f = np.arange(n), [np.ones(n)/np.sqrt(n)], [0.]
    for k in range(1, (n+1)//2):
        a, fk = np.sqrt(2/n), k*fs/n
        Q += [a*np.cos(2*np.pi*k*t/n), a*np.sin(2*np.pi*k*t/n)]
        f += [fk, fk]
    if n % 2 == 0:
        Q.append((-1.)**t/np.sqrt(n)); f.append(fs/2)
    return np.column_stack(Q), np.asarray(f)

def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def cmatrix(X):
    return np.nan_to_num(np.corrcoef(X, rowvar=False))

def structure(A, B):
    A, B = cmatrix(A), cmatrix(B)
    i, j = np.triu_indices(A.shape[0], 1)
    return corr(A[i, j], B[i, j])

def edge_structure(A, B, edges):
    A, B = cmatrix(A), cmatrix(B)
    return corr([A[i, j] for i, j in edges], [B[i, j] for i, j in edges])

def fisher_mean(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(x, -.999999, .999999)))))

def p300_label(classes):
    for c in classes:
        s = str(c).lower().replace("_", "").replace("-", "").replace(" ", "")
        if ("target" in s and "nontarget" not in s) or s == "p300":
            return c
    return classes[-1]

def included_subjects(ds):
    available = list(ds.subject_list)
    if SUBJECTS is None:
        return available, available
    wanted = SUBJECTS.get(ds.code, []) if isinstance(SUBJECTS, dict) else list(SUBJECTS)
    missing = [s for s in wanted if s not in available]
    if missing:
        raise ValueError(f"{ds.code}: unknown subjects {missing}")
    return available, wanted

class PBitEEGJoint(ClassifierMixin, BaseEstimator):
    def __init__(self, fs, duration, gamma=GAMMA, edges_per_coeff=PRECISION_EDGES_PER_COEFF,
                 precision_scale=PRECISION_SCALE, seed=SEED):
        self.fs, self.duration, self.gamma = fs, duration, gamma
        self.edges_per_coeff, self.precision_scale, self.seed = edges_per_coeff, precision_scale, seed

    def _prepare(self, X):
        n = int(round(self.fs*self.duration))
        X = np.asarray(X)
        if X.shape[-1] < n:
            raise ValueError(f"Need {n} samples, got {X.shape[-1]}")
        return X[..., :n]

    def _spectral(self, X):
        X = self._prepare(X)
        return np.einsum("nct,tk->nck", X, self.Q_).reshape(len(X), -1)

    def transform_u(self, X):
        return self.scaler_.transform(self._spectral(X))

    def _bitplane_solver(self):
        return BitPlaneCaSuDaSolver(
            Nt=NT, dt=.1, i0=1., bit_width=BITS, seed=self.seed,
            backend=NumpyBackend(dtype=np.float32, compile=True),
            cache_J=True, use_sparse=True, cache_static=True,
            reuse_buffers=True, shuffle_planes=True
        )

    def _block_solver(self):
        return BlockGibbsSolver(
            Nt=BLOCK_SWEEPS, dt=1., i0=1., seed=self.seed+123,
            backend=NumpyBackend(dtype=np.float32), shuffle_blocks=True
        )

    def fit(self, X, y):
        X = self._prepare(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Binary P300 problem expected")
        self.n_channels_, self.n_times_ = X.shape[1:]
        Q, f = fourier_basis(self.n_times_, self.fs)
        self.Q_ = Q[:, (f >= 1) & (f <= 18)]
        self.n_modes_ = self.Q_.shape[1]
        ratio = self.fs/FS_OUT
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("fs/FS_OUT must be integer")
        self.Q64_ = self.Q_[::int(round(ratio))]

        Z = self._spectral(X)
        self.scaler_ = StandardScaler().fit(Z)
        U = self.scaler_.transform(Z)
        self.n_coeff_ = U.shape[1]
        self.n_edges_ = max(1, int(round(self.edges_per_coeff*self.n_coeff_)))

        c0, c1 = self.classes_
        mu0, mu1 = U[y == c0].mean(0), U[y == c1].mean(0)
        m = (mu0+mu1)/2
        R = np.empty_like(U)
        R[y == c0] = U[y == c0]-mu0
        R[y == c1] = U[y == c1]-mu1

        K0 = LedoitWolf().fit(R).precision_
        d = np.diag(K0).copy()
        ii, jj = np.triu_indices(len(d), 1)
        score = np.abs(K0[ii, jj])/np.sqrt(d[ii]*d[jj])
        k = min(self.n_edges_, len(score))
        q = np.argsort(score)[-k:]
        K = np.diag(d)
        K[ii[q], jj[q]] = K0[ii[q], jj[q]]
        K[jj[q], ii[q]] = K0[jj[q], ii[q]]
        e = np.linalg.eigvalsh(K)[0]
        if e < 1e-4:
            K += np.eye(len(K))*(1e-4-e)
        K *= self.precision_scale
        self.edges_ = [(int(ii[z]), int(jj[z])) for z in q]

        X64 = np.einsum("nck,tk->nct", Z.reshape(len(X), self.n_channels_, self.n_modes_), self.Q64_)
        F = X64.reshape(len(X64), -1)
        ts = StandardScaler().fit(F)
        lr = LogisticRegression(C=.1, max_iter=3000, class_weight="balanced",
                                random_state=self.seed).fit(ts.transform(F), y)
        wr = lr.coef_[0]/ts.scale_
        br = lr.intercept_[0]-wr@ts.mean_
        ws = (wr.reshape(self.n_channels_, X64.shape[-1])@self.Q64_).reshape(-1)
        self.w_ = ws*self.scaler_.scale_
        self.b_ = br+ws@self.scaler_.mean_

        # Here starts the PCircuit part of PBitEEGJoint
        # It does encode the learned spectral distribution and class coupling
        # into a joint p-bit PCircuit, which is then used for both classification
        # and conditional P300 generation.
        self.quadratic_ = FixedPointQuadratic(K, m, bit_width=BITS, clip=CLIP)
        self.class_bit_ = self.quadratic_.n_pbits
        self.circuit_ = PCircuit(self.class_bit_+1)
        self.quadratic_.apply(self.circuit_)
        jy = (self.gamma*.5*self.w_[:, None]*self.quadratic_.beta[None, :]).reshape(-1)
        self.circuit_.J[:-1, -1] = self.circuit_.J[-1, :-1] = jy
        self.circuit_.h[-1] = self.gamma*.5*self.b_
        self.base_h_ = self.circuit_.h.copy()
        self.bitplane_solver_ = self._bitplane_solver()
        self.block_solver_ = self._block_solver()
        return self

    def _encode(self, U):
        return self.quadratic_.encode(U)

    def decision_function(self, X):
        S = self._encode(self.transform_u(X))
        return 2*(self.base_h_[-1]+S@self.circuit_.J[:-1, -1])

    def predict_proba(self, X):
        z = np.clip(self.decision_function(X), -40, 40)
        p = 1/(1+np.exp(-z))
        return np.c_[1-p, p]

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= .5).astype(int)]

    def _condition(self, sign):
        c = PCircuit(self.circuit_.n_pbits)
        c.J = self.circuit_.J.copy()
        c.h = self.base_h_.copy()
        c._pkit_block_groups = [g.copy() for g in self.quadratic_.groups()]
        c.h[:-1] += sign*c.J[:-1, -1]
        c.J[:-1, -1] = c.J[-1, :-1] = 0
        c.h[-1] = 50*sign
        return c

    def generate(self, label, n=N_GENERATED, return_u=False):
        sign = 1. if label == self.classes_[1] else -1.
        c = self._condition(sign)
        M0 = self.bitplane_solver_.solve(c, n_shots=n, return_final=True).astype(np.float32)
        M0[:, self.class_bit_] = sign
        M = self.block_solver_.solve(
            c, n_shots=n, initial_state=M0,
            clamped={self.class_bit_: sign}, return_final=True
        ).astype(np.float32)
        U = self.quadratic_.decode(M[:, :self.class_bit_])
        Z = U*self.scaler_.scale_+self.scaler_.mean_
        X = np.einsum("nck,tk->nct", Z.reshape(n, self.n_channels_, self.n_modes_), self.Q64_)
        return (X, U) if return_u else X

def classification_dataset(ds, subjects, fs, dur):
    print(f"\n[{ds.code}] CLASSIFICATION: MOABB WithinSessionEvaluation, {len(subjects)} subject(s), {N_SPLITS} folds/session")
    flat = {"FlatLR": make_pipeline(
        FunctionTransformer(flatten, validate=False), StandardScaler(),
        LogisticRegression(C=.1, max_iter=3000, class_weight="balanced", random_state=SEED)
    )}
    pbit = {"JointPBit": PBitEEGJoint(fs, dur)}
    print(f"[{ds.code}] CLASSIFICATION: train/evaluate FlatLR inside each session...")
    rf = WithinSessionEvaluation(
        paradigm=P300(fmin=1, fmax=18, resample=FS_OUT), datasets=[ds],
        n_splits=N_SPLITS, random_state=SEED, n_jobs=N_JOBS,
        overwrite=True, suffix=f"eegpcg011_{re.sub('[^A-Za-z0-9]','_',ds.code)}_flat"
    ).process(flat)
    print(f"[{ds.code}] CLASSIFICATION: train/evaluate JointPBit inside each session...")
    rp = WithinSessionEvaluation(
        paradigm=NonFilterP300(), datasets=[ds],
        n_splits=N_SPLITS, random_state=SEED, n_jobs=N_JOBS,
        overwrite=True, suffix=f"eegpcg011_{re.sub('[^A-Za-z0-9]','_',ds.code)}_pbit"
    ).process(pbit)
    r = pd.concat([rf, rp], ignore_index=True)
    session = r.groupby(["dataset", "subject", "session", "pipeline"], as_index=False).score.mean()
    subject = session.groupby(["dataset", "subject", "pipeline"], as_index=False).score.mean()
    pivot = subject.pivot(index="subject", columns="pipeline", values="score")
    print(f"[{ds.code}] CLASSIFICATION: per-subject mean AUC over sessions")
    for s, row in pivot.iterrows():
        print(f"  S{s}: JointPBit={row.get('JointPBit', np.nan):.3f} FlatLR={row.get('FlatLR', np.nan):.3f}")
    for pipe in ["JointPBit", "FlatLR"]:
        x = subject.loc[subject.pipeline == pipe, "score"].to_numpy()
        print(f"  {pipe:9s}: {x.mean():.4f} ± {x.std(ddof=1) if len(x)>1 else 0:.4f} (n={len(x)})")
    return subject

def plot_subject(subject, RP, GP, p300_r):
    v = max(np.abs(RP).max(), np.abs(GP).max())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].imshow(RP, aspect="auto", origin="lower", vmin=-v, vmax=v)
    ax[1].imshow(GP, aspect="auto", origin="lower", vmin=-v, vmax=v)
    ax[0].set_title("Held-out real P300")
    ax[1].set_title("Generated P300")
    ax[0].set_ylabel("Channel")
    fig.suptitle(f"Subject {subject}: P300 r={p300_r:.3f}")
    plt.show()

def generation_subject(ds, subject, fs, dur, pn, ps, seed, index, total):
    tag = f"[{ds.code}] GEN {index}/{total} S{subject}"
    print(f"{tag}: load and split...")
    X, y, meta = pn.get_data(dataset=ds, subjects=[subject])
    Xs, ys, metas = ps.get_data(dataset=ds, subjects=[subject])
    cols = ["subject", "session", "run"]
    if not np.array_equal(y, ys) or not np.array_equal(meta[cols].to_numpy(), metas[cols].to_numpy()):
        raise RuntimeError(f"{ds.code} subject {subject}: paradigm alignment failed")
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(X, y))
    print(f"{tag}: train joint PCircuit on {len(tr)} epochs...")
    model = PBitEEGJoint(fs, dur, seed=seed).fit(X[tr], y[tr])
    print(f"{tag}: trained | coeff={model.n_coeff_}, edges={len(model.edges_)}, pbits={model.class_bit_+1}")
    p300 = p300_label(model.classes_)
    print(f"{tag}: generate {N_GENERATED} P300 epochs with block Gibbs...")
    gp, ugp = model.generate(p300, N_GENERATED, True)
    print(f"{tag}: evaluate held-out P300...")
    rp = Xs[te][y[te] == p300]
    rp0 = Xs[tr][y[tr] == p300]
    up = model.transform_u(X[te][y[te] == p300])
    up0 = model.transform_u(X[tr][y[tr] == p300])
    RP, GP, RP0 = rp.mean(0), gp.mean(0), rp0.mean(0)
    row = {
        "dataset": ds.code, "subject": subject, "train": len(tr), "test": len(te),
        "coeff": model.n_coeff_, "pbits": model.class_bit_+1,
        "p300": corr(GP, RP), "global": structure(ugp, up),
        "edge": edge_structure(ugp, up, model.edges_),
        "ref_p300": corr(RP0, RP), "ref_global": structure(up0, up),
        "ref_edge": edge_structure(up0, up, model.edges_)
    }
    print(f"{tag}: done | P={row['p300']:.3f} G={row['global']:.3f} E={row['edge']:.3f}")
    if PLOT:
        plot_subject(subject, RP, GP, row["p300"])
    return row

def generation_dataset(ds, subjects, fs, dur):
    print(f"\n[{ds.code}] GENERATION: {len(subjects)} subject(s), each trained/generated/evaluated separately")
    pn, ps = NonFilterP300(), P300(fmin=1, fmax=18, resample=FS_OUT)
    rows = [generation_subject(ds, s, fs, dur, pn, ps, SEED+i, i+1, len(subjects))
            for i, s in enumerate(subjects)]
    print(f"[{ds.code}] GENERATION SUMMARY (Fisher-z mean correlations, n={len(rows)})")
    for key, name in [
        ("p300", "P300 mean"), ("global", "Global structure"), ("edge", "Precision edges"),
        ("ref_p300", "Real-real P300 ref"), ("ref_global", "Real-real global ref"),
        ("ref_edge", "Real-real edge ref")
    ]:
        print(f"  {name:22s}: r={fisher_mean([r[key] for r in rows]):.4f}")
    return rows

def run():
    if MODE not in ("classify", "generate", "both"):
        raise ValueError("MODE must be classify, generate or both")
    class_all, gen_all = [], []
    counts = [included_subjects(ds) for ds in DATASETS]
    n_available = sum(len(a) for a, _ in counts)
    n_included = sum(len(s) for _, s in counts)
    print("EEG p-bit joint PCircuit: classification + P300 generation")
    print(f"Datasets={len(DATASETS)}, included subjects={n_included}/{n_available}")
    print(f"Mode={MODE}, gamma={GAMMA}, edges/coeff={PRECISION_EDGES_PER_COEFF:g}, block_sweeps={BLOCK_SWEEPS}, generated/class={N_GENERATED}")
    for dsi, original in enumerate(DATASETS, 1):
        available, subjects = included_subjects(original)
        if not subjects:
            continue
        ds = copy.copy(original)
        ds.subject_list = list(subjects)
        fs = get_fs(ds)
        dur = float(ds.interval[1]-ds.interval[0])
        print("\n" + "="*72)
        print(f"Dataset {dsi}/{len(DATASETS)}: {ds.code}")
        print(f"Included subjects: {len(subjects)}/{len(available)} -> {subjects}")
        print(f"Sampling rate={fs:g} Hz, epoch duration={dur:g} s")
        if MODE in ("classify", "both"):
            class_all.append(classification_dataset(ds, subjects, fs, dur))
        if MODE in ("generate", "both"):
            gen_all += generation_dataset(ds, subjects, fs, dur)
    print("\n" + "="*72)
    print("FINAL DEMO SUMMARY")

    nsubjects = 0

    if class_all:
        c = pd.concat(class_all, ignore_index=True)
        p = c.pivot(index=["dataset", "subject"], columns="pipeline", values="score").dropna()
        j = p["JointPBit"].to_numpy()
        f = p["FlatLR"].to_numpy()
        nsubjects = len(p)

        print(f"Subjects: {nsubjects}")
        print("\nCLASSIFICATION — MOABB WithinSession")
        print(f"  JointPBit : AUC={j.mean():.4f} ± {j.std(ddof=1):.4f}")
        print(f"  FlatLR    : AUC={f.mean():.4f} ± {f.std(ddof=1):.4f}")
        print(f"  Delta     : {np.mean(j-f):+.4f}")

    if gen_all:
        if not class_all:
            print(f"Subjects: {len(gen_all)}")

        print("\nP300 GENERATION — held-out real EEG")
        for key, ref, name in [
            ("p300", "ref_p300", "Mean ERP"),
            ("global", "ref_global", "Global structure"),
            ("edge", "ref_edge", "Precision edges"),
        ]:
            g = fisher_mean([r[key] for r in gen_all])
            rr = fisher_mean([r[ref] for r in gen_all])
            print(f"  {name:18s}: r={g:.4f}   real-real={rr:.4f}")

    print("="*72)    

if __name__ == "__main__":
    run()
