"""Existing image encodings used for comparison: GAF, MTF, RP and a CWT image.

Each encoding gives one plane of 32 x 32 pixels, standardized with the mean and the standard
deviation of that plane over the series passed to ``fit``.

* GAF: Gramian angular summation field of ``pyts`` with each series scaled to [-1, 1].
* MTF: Markov transition field of ``pyts`` with eight quantile bins determined within each series.
* RP: unthresholded recurrence plot (Euclidean distances, embedding dimension 3, lag 4) of the
  series standardized with the pooled mean and standard deviation of the fitting series, averaged
  to 32 x 32 pixels and divided by sqrt(3) times the range of the standardized fitting series.
* CWT: log(1 + |W|^2) of a complex Morlet wavelet transform (``cmor1.5-1.0`` of PyWavelets) at
  32 geometrically spaced scales from 2 to T/4, with the time axis reduced to 32 positions by
  averaging over intervals.
"""
from __future__ import annotations

import numpy as np

from .representations import SIZE, _plane_scale_apply, _plane_scale_fit

METHODS = ("GAF", "MTF", "RP", "CWT")
WAVELET = "cmor1.5-1.0"


def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] < 2:
        raise ValueError("X must be a nonempty array of shape (n, T).")
    if not np.isfinite(X).all():
        raise ValueError("X contains nonfinite observations.")
    return X


def cwt_scales(T: int) -> np.ndarray:
    return np.geomspace(2.0, T / 4.0, SIZE)


def interval_weights(T: int, n_cols: int = SIZE) -> np.ndarray:
    """Weights that average T unit cells over ``n_cols`` equal intervals: array (n_cols, T)."""
    w = np.zeros((n_cols, T), dtype=np.float64)
    t = np.arange(T, dtype=np.float64)
    for j in range(n_cols):
        a, b = j * T / n_cols, (j + 1) * T / n_cols
        w[j] = np.maximum(0.0, np.minimum(t + 1.0, b) - np.maximum(t, a)) / (b - a)
    return w


class ExistingImageEncoder:
    """Encoder for one of ``"GAF"``, ``"MTF"``, ``"RP"`` and ``"CWT"``; output (n, 1, 32, 32), float32."""

    n_planes = 1

    def __init__(self, method: str, batch_size: int = 16):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.batch_size = method, int(batch_size)
        self._fitted = False

    def fit(self, X_train) -> "ExistingImageEncoder":
        X = _as_matrix(X_train)
        if self.method in ("GAF", "MTF") and X.shape[1] < SIZE:
            raise ValueError("GAF and MTF need series of length at least 32.")
        if self.method == "RP" and X.shape[1] <= 8:
            raise ValueError("Embedding dimension 3 and lag 4 need series longer than 8.")
        self.length_ = int(X.shape[1])
        self.mean_ = float(X.mean())
        sd = float(X.std())
        if self.method == "CWT":
            self.scale_ = max(sd, 1e-6)
        else:
            self.scale_ = sd if sd > 1e-12 else 1.0
            span = float((X.max() - self.mean_) / self.scale_) - float((X.min() - self.mean_) / self.scale_)
            self.rp_denominator_ = float(np.sqrt(3.0) * (span if span > 1e-12 else 1.0))
            from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
            if self.method == "GAF":
                self.transformer_ = GramianAngularField(image_size=SIZE, sample_range=(-1, 1), method="summation",
                                                        overlapping=False, flatten=False)
            elif self.method == "MTF":
                self.transformer_ = MarkovTransitionField(image_size=SIZE, n_bins=8, strategy="quantile",
                                                          overlapping=False, flatten=False)
            else:
                self.transformer_ = RecurrencePlot(dimension=3, time_delay=4, threshold=None, flatten=False)
            self.transformer_.fit(X)                        # the pyts transformers have no fitted state
        self._fitted = True
        self.plane_mean_, self.plane_std_ = _plane_scale_fit(self._image(X))
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(X_train) before transform(X).")
        X = _as_matrix(X)
        if X.shape[1] != self.length_:
            raise ValueError(f"series length {X.shape[1]} differs from the fitted length {self.length_}")
        return _plane_scale_apply(self._image(X), self.plane_mean_, self.plane_std_)

    def fit_transform(self, X_train) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    # ------------------------------------------------------------------ images before standardization
    def _image(self, X: np.ndarray) -> np.ndarray:
        out = self._cwt_image(X) if self.method == "CWT" else self._pyts_image(X)
        if not np.isfinite(out).all():
            raise FloatingPointError(f"nonfinite values in the {self.method} image")
        return out

    def _pyts_image(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), 1, SIZE, SIZE), dtype=np.float32)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            block = X[start:stop]
            if self.method == "RP":
                import torch
                from torch.nn import functional as F
                distances = self.transformer_.transform((block - self.mean_) / self.scale_)
                tensor = torch.from_numpy(np.asarray(distances, dtype=np.float32))[:, None]
                with torch.no_grad():
                    images = F.adaptive_avg_pool2d(tensor, (SIZE, SIZE))
                out[start:stop] = images.numpy() / self.rp_denominator_
            else:
                out[start:stop, 0] = self.transformer_.transform(block).astype(np.float32)
        return out

    def _cwt_image(self, X: np.ndarray) -> np.ndarray:
        import pywt
        Z = np.ascontiguousarray((X - self.mean_) / self.scale_, dtype=np.float64)
        T = Z.shape[1]
        coef, _ = pywt.cwt(Z, cwt_scales(T), WAVELET, sampling_period=1.0, method="fft", axis=-1, precision=12)
        power = np.log1p(np.abs(np.moveaxis(coef, 0, 1)) ** 2)        # (n, 32 scales, T), rows = increasing scale
        return np.einsum("nkt,jt->nkj", power, interval_weights(T)).astype(np.float32)[:, None, :, :]
