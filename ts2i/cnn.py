"""Compact residual CNN and the training procedure common to all image representations.

Network: a 3 x 3 convolutional input layer with 16 channels, two residual blocks with 16 and 32 channels
(group normalization, ReLU), each followed by 2 x 2 max pooling, a fully connected layer with 64 units,
dropout with probability 0.3 and the output layer; 150,304 + 144 P + 65 C weights for P planes and C classes.

Training: cross-entropy loss, AdamW (moment coefficients 0.9 and 0.999, weight decay 1e-4), batch size 16,
no data augmentation and no learning-rate schedule.  For each learning rate in (3e-4, 1e-3) the network is
trained on the fitting subset for at most 120 epochs; from the 20th epoch onward the epoch with the smallest
validation loss is recorded, and training stops after 20 eligible epochs without a decrease of more than 1e-8.
The learning rate with the smaller selected validation loss is chosen (ties within 1e-8: the smaller rate, then
the earlier epoch), and the final model is trained on the whole training set with that rate for the selected
number of epochs, from the same initial weights.

Importing this module sets PyTorch to one thread, deterministic algorithms and single precision.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn

LEARNING_RATES = (3e-4, 1e-3)
BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 128
MAX_EPOCHS = 120
MIN_SELECTABLE_EPOCH = 20
PATIENCE = 20
LOSS_IMPROVEMENT = 1e-8
LR_TIE_TOLERANCE = 1e-8
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-4

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)


class _ResBlockA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(4, 16, eps=1e-5, affine=True)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(4, 16, eps=1e-5, affine=True)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(h + x)


class _ResBlockB(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, 32, eps=1e-5, affine=True)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, 32, eps=1e-5, affine=True)
        self.short = nn.Conv2d(16, 32, 1, bias=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(h + self.short(x))


def build_network(n_planes: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(n_planes, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16, eps=1e-5, affine=True), nn.ReLU(inplace=False),
                         _ResBlockA(), nn.MaxPool2d(2), _ResBlockB(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(inplace=False), nn.Dropout(.3), nn.Linear(64, n_classes))


def expected_weights(n_planes: int, n_classes: int) -> int:
    return 150304 + 144 * n_planes + 65 * n_classes


def _n_weights(model) -> int:
    return int(sum(v.numel() for v in model.parameters()))


def _clone_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_initial_state(n_planes: int, n_classes: int, seed: int) -> dict:
    """Initial weights: PyTorch default initialization after seeding with ``seed``."""
    seed_everything(int(seed))
    model = build_network(n_planes, n_classes)
    assert _n_weights(model) == expected_weights(n_planes, n_classes)
    return _clone_state(model)


class _Selection:
    """Selection of the epoch: eligible from MIN_SELECTABLE_EPOCH; an epoch is recorded only if its validation
    loss is below the best one by more than LOSS_IMPROVEMENT; stop after PATIENCE eligible epochs without that."""

    def __init__(self):
        self.best_loss, self.best_epoch, self.stale, self.best_state = float("inf"), None, 0, None

    def update(self, epoch: int, val_loss: float, get_state) -> bool:
        if not math.isfinite(val_loss):
            raise FloatingPointError(f"nonfinite validation loss at epoch {epoch}")
        if epoch >= MIN_SELECTABLE_EPOCH:
            if val_loss < self.best_loss - LOSS_IMPROVEMENT:
                self.best_loss, self.best_epoch, self.stale = float(val_loss), int(epoch), 0
                self.best_state = get_state()
            else:
                self.stale += 1
        return self.stale >= PATIENCE


def _validation_loss(model, V_t, vy_t, loss_fn) -> float:
    b = VALIDATION_BATCH_SIZE
    model.eval()
    with torch.no_grad():
        tot = sum(float(loss_fn(model(V_t[i:i + b]), vy_t[i:i + b])) * len(V_t[i:i + b]) for i in range(0, len(V_t), b))
    return tot / len(V_t)


def predict_proba(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.from_numpy(np.ascontiguousarray(X[i:i + 128]))), 1).numpy() for i in range(0, len(X), 128)])


def train(n_planes, n_classes, seed, X, y, lr, init_state, *, validation=None, epochs=None) -> dict:
    """Train from ``init_state``.

    With ``validation=(X_val, y_val)``: selection run (at most MAX_EPOCHS epochs); the result contains
    ``selected_epoch`` and ``selected_loss``.  Otherwise: training for exactly ``epochs`` epochs.
    Random streams: dropout after seeding with ``seed + 1000``, order of the series from a generator
    seeded with ``seed + 50000``.
    """
    seed_everything(int(seed))
    model = build_network(n_planes, n_classes)
    model.load_state_dict(init_state)
    seed_everything(int(seed) + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY, amsgrad=False)
    X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    y_t = torch.tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(int(seed) + 50000)
    selection = None
    if validation is not None:
        V_t = torch.from_numpy(np.ascontiguousarray(validation[0], dtype=np.float32))
        vy_t = torch.tensor(validation[1], dtype=torch.long)
        selection = _Selection()
        epochs = MAX_EPOCHS
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        perm = torch.randperm(len(X_t), generator=gen)
        for ix in perm.split(BATCH_SIZE):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_t[ix]), y_t[ix])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ix)
        row = {"epoch": epoch, "train_loss": total / len(X_t)}
        if not math.isfinite(row["train_loss"]):
            raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
        stop = False
        if selection is not None:
            row["validation_loss"] = _validation_loss(model, V_t, vy_t, loss_fn)
            stop = selection.update(epoch, row["validation_loss"], lambda: _clone_state(model))
        history.append(row)
        if stop:
            break
    out = {"model": model, "history": history, "epochs_run": len(history)}
    if selection is not None:
        if selection.best_state is None:
            raise RuntimeError("no eligible validation epoch")
        out.update(selected_epoch=selection.best_epoch, selected_loss=selection.best_loss, best_state=selection.best_state)
    return out


def select_learning_rate(records: dict) -> dict:
    """records: {lr: {"selected_loss": ..., "selected_epoch": ...}}.  Smallest loss; ties within the tolerance:
    the smaller learning rate, then the earlier epoch."""
    best = min(r["selected_loss"] for r in records.values())
    tied = sorted([(lr, r["selected_epoch"]) for lr, r in records.items() if r["selected_loss"] <= best + LR_TIE_TOLERANCE], key=lambda t: (t[0], t[1]))
    return {"lr": tied[0][0], "epoch": tied[0][1]}
