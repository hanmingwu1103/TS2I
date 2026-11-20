"""Image representations of a univariate time series: MQI, MQI-LCF and MQI-UDI.

Every image is a stack of planes of 32 x 32 pixels.

* MQI (multiscale quantile image), six planes: local quantiles of the level, the lag sum and the
  lag difference of the standardized series, in overlapping windows at two scales.
  Rows are probability levels, columns are window positions.
* MQI-LCF, ten planes: MQI and four level-conditioned directional increment planes.  Within each
  window the positive and the negative increments are summarized by kernel-weighted root mean
  squares according to the ranks of their arrival levels.  Rows are rank-grid points.
* MQI-UDI, ten planes: MQI and four unconditional directional increment planes, in which each
  column repeats the root mean square of the positive or of the negative increments of the window.

All constants (mean and standard deviation of the series, ranges used for clipping, mean and
standard deviation of every plane) are estimated by ``fit`` from the series passed to it and are
then held fixed by ``transform``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

SIZE = 32                                   # rows and columns of every plane
FRACTIONS = (0.10, 0.30)                    # window fractions: short and long scale
BANDWIDTH = 1.0 / 16.0                      # bandwidth h of the Gaussian rank kernel
REPRESENTATIONS = ("MQI", "MQI-LCF", "MQI-UDI")
PROJECTIONS = ("level", "lag_sum", "lag_difference")

_ROWS = np.arange(SIZE, dtype=np.float64) / (SIZE - 1)      # grid u_r = r / 31
_TWO_H2 = 2.0 * BANDWIDTH * BANDWIDTH
_LEVEL, _DIFF = 0, 2


def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None]
    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError("X must be an array of shape (n, T) with T >= 3.")
    if not np.isfinite(X).all():
        raise ValueError("X contains nonfinite observations.")
    return X


def _finite(a, what):
    if not np.isfinite(a).all():
        raise FloatingPointError(f"nonfinite {what}")
    return a


def projections(Y: np.ndarray) -> np.ndarray:
    """Level, lag sum and lag difference of standardized series: array (n, 3, T - 1).

    Position j = 1, ..., T - 1 carries y_j, (y_j + y_{j-1}) / sqrt(2) and (y_j - y_{j-1}) / sqrt(2).
    """
    Y = _as_matrix(Y)
    root2 = np.sqrt(2.0)
    return np.stack((Y[:, 1:], (Y[:, 1:] + Y[:, :-1]) / root2, (Y[:, 1:] - Y[:, :-1]) / root2), axis=1)


def window_layout(N: int, fraction: float, size: int = SIZE):
    """Starts and common length of ``size`` equally spaced windows over N positions.

    The length is L = min(N, max(2, round(N * fraction))).  When N - L < size - 1 some
    neighboring windows coincide; they are retained.
    """
    length = min(N, max(2, int(round(N * fraction))))
    starts = np.rint(np.linspace(0, N - length, size)).astype(np.int64)
    return starts, int(length)


def _fit_range(values: np.ndarray):
    """Minimum and maximum; a range narrower than 1e-12 is widened by 1/2 at each end."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        low, high = low - 0.5, high + 0.5
    return low, high


def _plane_scale_fit(Z: np.ndarray):
    """Mean and standard deviation of every plane over series and pixels (SD not below 1e-6)."""
    mean = Z.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(Z.std(axis=(0, 2, 3), keepdims=True), 1e-6)
    return mean, std


def _plane_scale_apply(Z: np.ndarray, mean, std) -> np.ndarray:
    return ((Z - mean) / std).astype(np.float32)


def _chunk(L: int) -> int:
    return max(1, int(2.5e8 // (SIZE * SIZE * L * 8)))


class TimeSeriesImageEncoder:
    """Encoder for one of the representations ``"MQI"``, ``"MQI-LCF"`` and ``"MQI-UDI"``.

    >>> enc = TimeSeriesImageEncoder("MQI-LCF").fit(X_train)       # X_train: (n, T)
    >>> images = enc.transform(X_test)                            # (m, 10, 32, 32), float32
    """

    def __init__(self, representation: str = "MQI-LCF"):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        self.representation = representation
        self.n_planes = 6 if representation == "MQI" else 10
        self._fitted = False

    # ------------------------------------------------------------------ fitting
    def fit(self, X_train) -> "TimeSeriesImageEncoder":
        X = _as_matrix(X_train)
        self.mean_ = float(X.mean())
        scale = float(X.std())
        self.scale_ = scale if scale > 1e-12 else 1.0
        self.length_ = int(X.shape[1])
        P = projections((X - self.mean_) / self.scale_)
        self.ranges_ = [_fit_range(P[:, c]) for c in range(3)]
        self._fitted = True
        Q = self._quantile_planes(X)
        self.q_mean_, self.q_std_ = _plane_scale_fit(Q)
        if self.representation != "MQI":
            raw = _finite(self._added_planes(X), "added planes")
            self.added_ranges_ = (_fit_range(raw[:, (0, 1)]), _fit_range(raw[:, (2, 3)]))
            z = self._added_to_unit(raw)
            mean, std = _plane_scale_fit(z)
            self.a_mean_ = np.asarray(mean, dtype=np.float32).reshape(1, 4, 1, 1)
            self.a_std_ = np.asarray(std, dtype=np.float32).reshape(1, 4, 1, 1)
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, X) -> np.ndarray:
        """Standardized planes, array (n, 6 or 10, 32, 32) of dtype float32."""
        if not self._fitted:
            raise RuntimeError("Call fit(X_train) before transform(X).")
        X = _as_matrix(X)
        if X.shape[1] != self.length_:
            raise ValueError(f"series length {X.shape[1]} differs from the fitted length {self.length_}")
        out = _plane_scale_apply(self._quantile_planes(X), self.q_mean_, self.q_std_)
        if self.representation == "MQI":
            return out
        z = self._added_to_unit(_finite(self._added_planes(X), "added planes"))
        added = _plane_scale_apply(z, self.a_mean_, self.a_std_)
        return np.concatenate([out, added], axis=1)

    def fit_transform(self, X_train) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    # ------------------------------------------------------------- MQI planes
    def _quantile_planes(self, X: np.ndarray) -> np.ndarray:
        """Six planes of local quantiles scaled to [0, 1] by the fitted ranges (float32).

        Plane order: level, lag sum, lag difference; within each, short then long windows.
        """
        P = projections((X - self.mean_) / self.scale_)
        out = np.empty((len(X), 6, SIZE, SIZE), dtype=np.float32)
        levels = np.linspace(0, 1, SIZE)
        plane = 0
        for c in range(3):
            low, high = self.ranges_[c]
            width = high - low
            values = np.clip(P[:, c], low, high)
            for fraction in FRACTIONS:
                starts, length = window_layout(values.shape[1], fraction)
                for row, z in enumerate(values):
                    windows = np.stack([z[s:s + length] for s in starts])
                    out[row, plane] = (np.quantile(windows, levels, axis=1, method="linear") - low) / width
                plane += 1
        return out

    # ----------------------------------------------------------- added planes
    def _added_planes(self, X: np.ndarray) -> np.ndarray:
        """Four added planes before normalization, float64 (n, 4, 32, 32).

        Plane order: positive short, positive long, negative short, negative long.
        """
        P = projections((X - self.mean_) / self.scale_)
        _finite(P, "standardized projections")
        a = np.clip(P[:, _LEVEL], *self.ranges_[_LEVEL])       # arrival levels
        d = np.clip(P[:, _DIFF], *self.ranges_[_DIFF])         # increments
        n, N = a.shape
        out = np.empty((n, 4, SIZE, SIZE))
        for si, f in enumerate(FRACTIONS):
            starts, L = window_layout(N, f)
            idx = starts[:, None] + np.arange(L)[None, :]       # positions of each window
            A = a[:, idx]                                       # (n, 32 windows, L)
            D = d[:, idx]
            vp2 = _finite(np.square(np.maximum(D, 0.0)), "squared positive parts")
            vn2 = _finite(np.square(np.maximum(-D, 0.0)), "squared negative parts")
            if self.representation == "MQI-UDI":
                up = np.sqrt(_finite(vp2.mean(axis=-1), "positive mean"))
                un = np.sqrt(_finite(vn2.mean(axis=-1), "negative mean"))
                out[:, si] = np.broadcast_to(up[:, None, :], (n, SIZE, SIZE))
                out[:, 2 + si] = np.broadcast_to(un[:, None, :], (n, SIZE, SIZE))
                continue
            cs = _chunk(L)
            for i0 in range(0, n, cs):
                sl = slice(i0, min(n, i0 + cs))
                # normalized midranks of the arrival levels within each window
                u = (rankdata(A[sl], method="average", axis=-1) - 1.0) / (L - 1.0)
                K = np.exp(-np.square(u[..., None, :] - _ROWS[:, None]) / _TWO_H2)
                w = K / K.sum(axis=-1, keepdims=True)           # (c, windows, rows, L)
                sp = np.einsum("nkrj,nkj->nrk", w, vp2[sl])     # (c, rows, windows)
                sn = np.einsum("nkrj,nkj->nrk", w, vn2[sl])
                out[sl, si] = np.sqrt(sp)
                out[sl, 2 + si] = np.sqrt(sn)
        return out

    def _added_to_unit(self, raw: np.ndarray) -> np.ndarray:
        """Clip the added planes to their fitted ranges and map them to [0, 1] (float32)."""
        z = np.empty_like(raw)
        for planes, (low, high) in (((0, 1), self.added_ranges_[0]), ((2, 3), self.added_ranges_[1])):
            z[:, planes] = (np.clip(raw[:, planes], low, high) - low) / (high - low)
        return _finite(z.astype(np.float32), "added planes scaled to the unit interval")
