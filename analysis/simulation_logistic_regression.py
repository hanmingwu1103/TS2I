"""Simulation study with a second classifier: logistic regression on the pixels of the images.

Sensitivity analysis of the simulation study of the paper.  The series, the images and the division of the
training set are those used for the common CNN; the CNN is replaced by logistic regression with an L2 penalty
on all pixels of the image.  The inverse penalty strength C is selected on the validation subset by the
cross-entropy (ties: the smaller C), and the classifier is refitted to the whole training set.  The fits are
computed as in analysis/realdata_logistic_regression.py (tolerance 1e-8, coordinates of the row space of the
training images; see "Computation" and "Convergence" there).

    python analysis/simulation_logistic_regression.py                 # summary of the saved scores
    python analysis/simulation_logistic_regression.py --run out.csv   # recompute all 360 classifiers (about 12 minutes)

Saved scores: results/simulation_scores_logistic.csv (scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1).
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPRESENTATIONS = ("MQI", "MQI-LCF", "MQI-UDI")
SETTINGS = ("A35", "A70", "NOISE", "TIES", "LEVEL", "NULL")


def one(task):
    replicate, name = task
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.model_selection import train_test_split
    from ts2i import make_encoder, simulation
    import realdata_logistic_regression as R

    X, y = simulation.generate(replicate, name, "train")
    Xt, yt = simulation.generate(replicate, name, "test")
    fit_ix, val_ix = train_test_split(np.arange(len(y)), test_size=0.25, random_state=61000 + replicate, stratify=y)
    fit_ix, val_ix = np.sort(fit_ix), np.sort(val_ix)          # as in the simulation study
    rows = []
    for rep in REPRESENTATIONS:
        enc = make_encoder(rep).fit(X[fit_ix])
        Zf, (Zv,), p, how = R.reduce(enc.transform(X[fit_ix]), [enc.transform(X[val_ix])])
        losses, its = {}, {}
        for C in R.C_GRID:
            m, its[C], _, _ = R.fit(Zf, y[fit_ix], C)
            losses[C] = float(log_loss(y[val_ix], m.predict_proba(Zv), labels=[0, 1]))
        best = min(losses.values())
        C_sel = min(C for C in R.C_GRID if losses[C] <= best + R.TIE)
        enc = make_encoder(rep).fit(X)
        Za, (Zb,), p, how = R.reduce(enc.transform(X), [enc.transform(Xt)])
        m, it, _, _ = R.fit(Za, y, C_sel)
        pred = m.predict(Zb)
        rows.append({"replicate": replicate, "setting": name, "representation": rep,
                     "macro_f1": float(f1_score(yt, pred, average="macro", labels=[0, 1], zero_division=0)),
                     "accuracy": float(accuracy_score(yt, pred)),
                     "single_class_prediction": int(len(np.unique(pred)) == 1),
                     "selected_C": C_sel, "validation_log_loss": best, "n_features": p,
                     "iterations": it, "max_iterations_in_selection": max(its.values()),
                     "not_converged": int(max(max(its.values()), it) >= R.MAX_ITER), "n_train": len(y), "n_test": len(yt)})
    return rows


def run(path, workers):
    tasks = [(r, s) for r in range(1, 21) for s in SETTINGS]
    t0, out = time.time(), []
    with Pool(workers) as pool:
        for i, rows in enumerate(pool.imap(one, tasks), 1):
            out.extend(rows)
            if i % 12 == 0:
                print(f"{i}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


def summary(path):
    from scipy import stats
    f1, acc, single = defaultdict(dict), defaultdict(dict), defaultdict(int)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["setting"], r["representation"])
        f1[key][int(r["replicate"])] = 100.0 * float(r["macro_f1"])
        acc[key][int(r["replicate"])] = 100.0 * float(r["accuracy"])
        single[key] += int(r["single_class_prediction"])
    print("Logistic regression on the pixels: mean macro-F1 (SD over 20 replicates), mean accuracy, models predicting a single class")
    for s in SETTINGS:
        for rep in REPRESENTATIONS:
            a, b = np.array(list(f1[(s, rep)].values())), np.array(list(acc[(s, rep)].values()))
            print(f"  {s:6s} {rep:8s} macro-F1 {a.mean():6.2f} ({a.std(ddof=1):5.2f})   accuracy {b.mean():6.2f}   single class {single[(s, rep)]}")
    reps = sorted(f1[("A35", "MQI")])
    primary = lambda table, rep: np.array([(table[("A35", rep)][r] + table[("A70", rep)][r]) / 2 for r in reps])
    print("  Contrasts of the paper (A35 and A70 averaged within each replicate; 95% t intervals):")
    for other in ("MQI", "MQI-UDI"):
        e = primary(f1, "MQI-LCF") - primary(f1, other)
        se = e.std(ddof=1) / np.sqrt(len(e)); h = stats.t.ppf(0.975, len(e) - 1) * se
        ea = primary(acc, "MQI-LCF") - primary(acc, other)
        print(f"  MQI-LCF - {other:8s} mean {e.mean():6.2f}  95% interval [{e.mean() - h:.2f}, {e.mean() + h:.2f}]  t = {e.mean() / se:.2f}  "
              f"range of the effects [{e.min():.2f}, {e.max():.2f}]  accuracy gain {ea.mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", metavar="CSV", help="recompute the fits and write the scores to this file")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.run:
        run(a.run, a.workers)
    summary(a.run or ROOT / "results" / "simulation_scores_logistic.csv")
