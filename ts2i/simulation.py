"""Data-generating mechanism of the simulation study.

A latent chain z_t in {0, 1, 2, 3}, t = 1, ..., 401, starts from a uniformly distributed state.  At each
step it stays with probability 1 - rho, moves by the cyclic map A (0 -> 2 -> 3 -> 1 -> 0) with probability
rho * alpha and by its inverse B with probability rho * (1 - alpha), where alpha = (1 + Y * gamma) / 2 and
Y in {-1, +1} is the class label.  Each series has its own move rate rho ~ U(0.5, 0.9), gain
beta ~ U(0.8, 1.2) and offset beta0 ~ U(-0.2, 0.2) and is observed as

    x_t = beta * (z_t - 1.5) + beta0 + Y * delta + eta * e_t,     e_t independent U(-1, 1).

``generate`` reproduces the series of the study: every series has its own random stream, determined by
the replicate, the setting, the set (training or test) and the index of the series.
"""
from __future__ import annotations

import numpy as np

T = 401
A_MAP = (2, 0, 3, 1)          # 0 -> 2 -> 3 -> 1 -> 0
B_MAP = (1, 3, 0, 2)          # inverse cycle
SETTINGS = (                  # fixed order; k is the position of the setting in the random-stream key
    {"id": "A35", "k": 1, "gamma": 0.35, "delta": 0.0, "eta": 0.10, "role": "primary"},
    {"id": "A70", "k": 2, "gamma": 0.70, "delta": 0.0, "eta": 0.10, "role": "primary"},
    {"id": "NOISE", "k": 3, "gamma": 0.70, "delta": 0.0, "eta": 0.75, "role": "challenge"},
    {"id": "TIES", "k": 4, "gamma": 0.70, "delta": 0.0, "eta": 0.0, "role": "challenge"},
    {"id": "LEVEL", "k": 5, "gamma": 0.0, "delta": 0.35, "eta": 0.10, "role": "control"},
    {"id": "NULL", "k": 6, "gamma": 0.0, "delta": 0.0, "eta": 0.10, "role": "control"},
)
SIZES = {"train": 128, "test": 512}
N_REPLICATES = 20
_MASTER = (20261116, 61)
_SET = {"train": 0, "test": 1}
_PURPOSE = {"series": 0, "order": 1}


def setting(name: str) -> dict:
    return next(s for s in SETTINGS if s["id"] == name)


def _rng(replicate: int, k: int, which: str, purpose: str, index: int) -> np.random.Generator:
    key = [_MASTER[0], _MASTER[1], int(replicate), int(k), _SET[which], _PURPOSE[purpose], int(index)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))


def draw_series_constants(rng: np.random.Generator) -> dict:
    """Draws of one series, in this order: rho, beta, beta0, z_1, 400 uniforms for the moves, 401 noise terms."""
    return {"rho": float(rng.uniform(0.5, 0.9)), "beta": float(rng.uniform(0.8, 1.2)), "beta0": float(rng.uniform(-0.2, 0.2)),
            "z1": int(rng.integers(0, 4)), "V": rng.random(T - 1), "E": rng.uniform(-1.0, 1.0, T)}


def simulate_series(Y: int, s: dict, d: dict):
    """Latent path z (T,) and observations x (T,) for label Y in {-1, +1}, setting ``s`` and draws ``d``."""
    if Y not in (-1, 1):
        raise ValueError("Y must be -1 or +1")
    gamma, delta, eta = float(s["gamma"]), float(s["delta"]), float(s["eta"])
    alpha = (1.0 + Y * gamma) / 2.0
    rho = float(d["rho"])
    stay = 1.0 - rho
    to_a = 1.0 - rho + rho * alpha
    Z = np.empty(T, dtype=np.int64)
    Z[0] = int(d["z1"])
    V = np.asarray(d["V"], dtype=np.float64)
    for t in range(1, T):
        v, prev = V[t - 1], Z[t - 1]
        if v < stay:
            Z[t] = prev
        elif v < to_a:
            Z[t] = A_MAP[prev]
        else:
            Z[t] = B_MAP[prev]
    X = float(d["beta"]) * (Z.astype(np.float64) - 1.5) + float(d["beta0"]) + Y * delta + eta * np.asarray(d["E"], dtype=np.float64)
    return Z, X


def generate(replicate: int, name: str, which: str):
    """Series and labels of one replicate (1, ..., 20), setting and set ("train" or "test").

    Returns X of shape (n, 401) and labels in {0, 1} (label 0 is Y = -1, label 1 is Y = +1), n = 128 for the
    training set and 512 for the test set, with equal class sizes.
    """
    if not 1 <= replicate <= N_REPLICATES:
        raise ValueError("replicate must be in 1, ..., 20")
    s, n = setting(name), SIZES[which]
    labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    X = np.empty((n, T), dtype=np.float64)
    for i in range(n):
        d = draw_series_constants(_rng(replicate, s["k"], which, "series", i))
        _, X[i] = simulate_series(-1 if labels[i] == 0 else 1, s, d)
    order = _rng(replicate, s["k"], which, "order", 0).permutation(n)
    return X[order], labels[order]
