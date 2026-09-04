"""
Fashion-MNIST controlled-correlation p-bit demo.

The image-classification task is expressed largely as a probabilistic circuit,
which stores the model through base h/J terms and named correlation components.
The solver controls how these components are activated over time.

Image evidence is applied through h, while each class is represented by a learned
J/h correlation component that is gradually activated by CorrelationAnnealingSolver.

The current model uses 720 p-bits. The implementation is intentionally simple.
Calculation speed has been optimized, but it can be further improved.

This is POC demo and classification accuracy can also be improved.
Inference is slower/heavier than training in general.
"""
from pathlib import Path
from urllib.request import urlretrieve
import gzip, struct
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from p_kit.psl import PCircuit
from p_kit.solver.corr_ann_solver import CorrelationAnnealingSolver, staged_ramp

from joblib import Parallel, delayed
import os
import time

N_WORKERS = min(4, os.cpu_count() or 1)
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SIZE = 28
N_FEATURES = 10
PATCH_H, PATCH_W = 6, 7
STRIDE = 3
FEATURE_PATCHES = 40000
FEATURE_BETA = 2.0

TRAIN_PER_CLASS = 2000
OFFSETS = [(0,1),(0,2),(0,3),(1,0),(2,0),(3,0),
           (1,1),(2,2),(1,-1),(2,-2),(1,2),(2,1),(1,-2),(2,-1)]

I0 = 2.0
SAMPLES = 10
NT = 60
TEST_LIMIT = 200 # max test images
SEED = 1234

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# Fashion-MNIST
# ----------------------------------------------------------------------

DATA = Path("data/fashion")
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

DATA.mkdir(parents=True, exist_ok=True)
for name in FILES:
    p = DATA / name
    if not p.exists():
        print("Downloading", name)
        urlretrieve(BASE + name, p)

def load_images(path):
    with gzip.open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, r, c) / 255.0

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8)

x_train = load_images(DATA / FILES[0])
y_train = load_labels(DATA / FILES[1])
x_test = load_images(DATA / FILES[2])
y_test = load_labels(DATA / FILES[3])

# ----------------------------------------------------------------------
# Shared local categorical features
# ----------------------------------------------------------------------

def positions(size, patch):
    p = list(range(0, size - patch + 1, STRIDE))
    if p[-1] != size - patch:
        p.append(size - patch)
    return p

YS, XS = positions(SIZE, PATCH_H), positions(SIZE, PATCH_W)
GH, GW = len(YS), len(XS)
GRID_N = GH * GW
N_PBITS = GRID_N * N_FEATURES

patches = []
while len(patches) < FEATURE_PATCHES:
    im = x_train[rng.integers(len(x_train))]
    y, x = rng.choice(YS), rng.choice(XS)
    p = im[y:y+PATCH_H, x:x+PATCH_W]
    if p.mean() > .04:
        patches.append(p.ravel())

km = MiniBatchKMeans(N_FEATURES - 1, batch_size=1024, n_init=5,
                     random_state=SEED).fit(patches)
templates = np.vstack([np.zeros(PATCH_H * PATCH_W), km.cluster_centers_])

def feature_prob(images):
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None]
    P = np.stack([images[:, y:y+PATCH_H, x:x+PATCH_W].reshape(len(images), -1)
                  for y in YS for x in XS], axis=1)
    d = ((P[:,:,None] - templates[None,None]) ** 2).mean(-1)
    z = (d - d.min(2, keepdims=True)) / (d.std(2, keepdims=True) + 1e-6)
    z = -FEATURE_BETA * z
    z -= z.max(2, keepdims=True)
    e = np.exp(z)
    return e / e.sum(2, keepdims=True)

def image_state(image):
    p = feature_prob(image)[0]
    u = np.log(np.clip(p, 1e-6, 1))
    u -= u.mean(1, keepdims=True)
    q = np.zeros_like(p)
    q[np.arange(GRID_N), p.argmax(1)] = 1
    return (u / 2).ravel(), (2 * q - 1).ravel()

# ----------------------------------------------------------------------
# Learn class-dependent feature correlations
# ----------------------------------------------------------------------

def shifted(F, dy, dx):
    ya, yb = (slice(0,-dy), slice(dy,None)) if dy else (slice(None), slice(None))
    if dx > 0:
        xa, xb = slice(0,-dx), slice(dx,None)
    elif dx < 0:
        xa, xb = slice(-dx,None), slice(0,dx)
    else:
        xa = xb = slice(None)
    return F[:,ya,xa], F[:,yb,xb]

priors = np.zeros((10, N_FEATURES))
joints = np.zeros((10, len(OFFSETS), N_FEATURES, N_FEATURES))
acount = np.zeros((10, len(OFFSETS), N_FEATURES))

print("Learning class correlations...")

for c in range(10):
    ids = rng.choice(np.flatnonzero(y_train == c), TRAIN_PER_CLASS, replace=False)
    F = feature_prob(x_train[ids]).reshape(-1, GH, GW, N_FEATURES)
    priors[c] = F.sum((0,1,2))

    for o, (dy, dx) in enumerate(OFFSETS):
        a, b = shifted(F, dy, dx)
        a, b = a.reshape(-1, N_FEATURES), b.reshape(-1, N_FEATURES)
        acount[c,o] = a.sum(0)
        joints[c,o] = a.T @ b

alpha = 1.0
prior_p = (priors + alpha/N_FEATURES) / (priors.sum(1, keepdims=True) + alpha)
global_prior = priors.sum(0) / priors.sum()

cond = (joints + alpha/N_FEATURES) / (acount[:,:,:,None] + alpha)
global_cond = (joints.sum(0) + alpha/N_FEATURES) / (acount.sum(0)[:,:,None] + alpha)

prior_w = np.clip(np.log((prior_p + 1e-12) / (global_prior + 1e-12)), -2.5, 2.5)
rel_w = np.clip(np.log((cond + 1e-12) / (global_cond[None] + 1e-12)), -2.5, 2.5)

# ----------------------------------------------------------------------
# Convert categorical correlations to Ising J/h
# ----------------------------------------------------------------------

def block(y, x):
    s = (y * GW + x) * N_FEATURES
    return np.arange(s, s + N_FEATURES)

def pairs(dy, dx):
    for y in range(GH):
        for x in range(GW):
            yy, xx = y + dy, x + dx
            if 0 <= yy < GH and 0 <= xx < GW:
                yield y, x, yy, xx

def build_model(c):
    J = np.zeros((N_PBITS, N_PBITS))
    h = np.zeros(N_PBITS)
    offset = 0.0
    U = np.tile(prior_w[c], (GRID_N, 1))

    for o, (dy, dx) in enumerate(OFFSETS):
        W = rel_w[c,o]
        row, col, mean = W.mean(1), W.mean(0), W.mean()
        W0 = W - row[:,None] - col[None,:] + mean

        for y, x, yy, xx in pairs(dy, dx):
            a, b = block(y,x), block(yy,xx)
            J[np.ix_(a,b)] += W0 / 4
            J[np.ix_(b,a)] += W0.T / 4
            U[y*GW+x] += row
            U[yy*GW+xx] += col
            offset += mean

    for s in range(GRID_N):
        u = U[s]
        mean = u.mean()
        h[s*N_FEATURES:(s+1)*N_FEATURES] = (u - mean) / 2
        offset -= mean

    return J, h, offset

models = [build_model(c) for c in range(10)]
radius = max(np.max(np.abs(np.linalg.eigvalsh(J))) for J, _, _ in models)
scale = 1.0 / radius

circuits, solvers, offsets = [], [], []

for c, (J, h, offset) in enumerate(models):
    circuit = PCircuit(N_PBITS)
    circuit.set_correlation_component("class", J=scale*J, h=scale*h)

    solver = CorrelationAnnealingSolver(
        Nt=NT, dt=.1667, i0=I0, seed=SEED+c,
        block_size=N_FEATURES,
        component_schedules={"class": staged_ramp(.2, .8)}
    )

    circuits.append(circuit)
    solvers.append(solver)
    offsets.append(scale * offset)

# ----------------------------------------------------------------------
# Controlled-correlation classification
# ----------------------------------------------------------------------

def classify(image):
    h_image, initial = image_state(image)
    E = np.zeros(10)

    for c in range(10):
        circuits[c].h = h_image
        _, best_E = solvers[c].solve(
            circuits[c], n_shots=SAMPLES, initial_state=initial,
            return_best=True, target_scales={"class": 1.0}
        )
        E[c] = best_E.mean() + offsets[c]

    return E.argmin()

def classify_chunk(indices):
    return [(i, classify(x_test[i])) for i in indices]


# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()

    print(f"Running controlled-correlation demo ({N_PBITS} p-bits, {N_WORKERS} workers)...")

    n = min(TEST_LIMIT, len(x_test))
    chunks = np.array_split(np.arange(n), N_WORKERS)

    results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(classify_chunk)(chunk) for chunk in chunks
    )

    pred = np.empty(n, dtype=int)
    for chunk in results:
        for i, p in chunk:
            pred[i] = p

    elapsed = time.perf_counter() - t0
    correct = np.sum(pred == y_test[:n])

    print(f"Final accuracy: {100*correct/n:.2f}%")
    print(f"Elapsed: {elapsed:.1f} s ({elapsed/60:.1f} min)")