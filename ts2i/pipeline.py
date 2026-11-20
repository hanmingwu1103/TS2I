"""Classification of one dataset with one image representation, as described in the paper.

The function below strings together the steps of the classification procedure.  The studies of the paper
were run with additional bookkeeping code (caching of images, logging, checks of the saved files), which is
not part of this package.
"""
from __future__ import annotations

import numpy as np

from . import make_encoder

VALIDATION_FRACTION = 0.25


def scores(y, proba, n_classes) -> dict:
    """Macro-F1 (a class with a zero denominator contributes zero), accuracy and balanced accuracy."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    pred = np.asarray(proba, dtype=np.float32).astype(np.float64).argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(n_classes), zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def evaluate(representation: str, X_train, y_train, X_test, y_test, seed: int, sort_subsets: bool = False) -> dict:
    """Train the common CNN on images of the training series and score it on the test series.

    1. The training set is divided into a fitting subset (75%) and a validation subset (25%), stratified by class.
    2. The constants of the representation are estimated from the fitting subset; for each learning rate the
       network is trained on the fitting subset, and the learning rate and the number of epochs are selected
       on the validation subset.
    3. The constants are estimated again from the whole training set, and the final network is trained on the
       whole training set with the selected learning rate and number of epochs, from the same initial weights.

    ``sort_subsets``: the order of the series within the fitting subset affects the batches.  In the studies on
    real data the indices were used in the order returned by ``train_test_split`` (the default here); in the
    simulation study they were sorted in ascending order (``sort_subsets=True``).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from . import cnn

    X_train, X_test = np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)
    le = LabelEncoder().fit(y_train)
    y, yt = le.transform(y_train).astype(np.int64), le.transform(y_test).astype(np.int64)
    K = len(le.classes_)
    fit_ix, val_ix = train_test_split(np.arange(len(y)), test_size=VALIDATION_FRACTION, random_state=int(seed), stratify=y)
    if sort_subsets:
        fit_ix, val_ix = np.sort(fit_ix), np.sort(val_ix)

    enc = make_encoder(representation).fit(X_train[fit_ix])
    F, V = enc.transform(X_train[fit_ix]), enc.transform(X_train[val_ix])
    P = F.shape[1]
    init = cnn.make_initial_state(P, K, seed)
    runs = {lr: cnn.train(P, K, seed, F, y[fit_ix], lr, init, validation=(V, y[val_ix])) for lr in cnn.LEARNING_RATES}
    choice = cnn.select_learning_rate(runs)

    enc = make_encoder(representation).fit(X_train)
    final = cnn.train(P, K, seed, enc.transform(X_train), y, choice["lr"], init, epochs=choice["epoch"])
    out = scores(yt, cnn.predict_proba(final["model"], enc.transform(X_test)), K)
    out.update(learning_rate=choice["lr"], epochs=choice["epoch"], n_planes=P, n_classes=K)
    return out
