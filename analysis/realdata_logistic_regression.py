"""Studies on real data with a second classifier: logistic regression on the pixels of the images.

Sensitivity analysis of the two studies on real data of the paper.  The series, the images and the division of each
training set are those of the analyses with the common CNN; the CNN is replaced by logistic regression with an L2
penalty on all pixels of the image.  The inverse penalty strength C is selected on the validation subset by the
cross-entropy (ties: the smaller C), and the classifier is refitted to the whole training set with the constants of
the representation estimated from the whole training set, as for the CNN.

Stages (each resumable; ``encode`` writes one file per dataset, ``select`` one per dataset, representation and
division, ``final`` one per dataset and representation; ``collect`` writes the score file):
    encode   images of the fitting, validation, training and test series, and their reduction (see below).  With
             ``--manifest`` every image is compared with the SHA-256 of the array stored by the authors as input of
             the CNN; the manifest refers to files that are not part of this repository, and the option can be left
             out.  No labels of test series are read.
    select   fits on the fitting subsets and cross-entropy on the validation subsets for the grid of C.
    final    refit on the whole training set with the selected C and scores on the test set.  The only stage that
             reads the labels of the test series.
    collect  score file (results/realdata_scores_logistic.csv in this repository).

Data.  ``--meta`` names a JSON file that lists the datasets and, relative to ``--data``, their files:
analysis/realdata_studies_metadata.json is the file of the authors.  The development datasets are read from one
``.npz`` file each (arrays X_train, y_train, X_test, y_test), the datasets of the independent evaluation from the
``.ts`` files of their sources (training and test file, series of equal length).  The datasets are not distributed
with this repository (see "Data" in the README).

Computation.  With n series and p > n pixels, the fitted probabilities of L2-penalized logistic regression depend
on the pixels only through their coordinates in the row space of the n x p matrix F of the training images: with
the thin QR decomposition F' = QR, the model with covariates F and coefficients w is the model with covariates
R' (n x n) and coefficients b = Q'w, the penalty |w|^2 = |b|^2 is unchanged, and new images B enter as BQ.  The
classifier is fitted to R'; this is a change of coordinates, not of the estimator.  When p <= n the pixels are used
directly.

Convergence.  The classifier is the minimizer of the penalized negative log-likelihood.  With the default tolerance
of scikit-learn (1e-4), L-BFGS can stop before the minimum when the penalty is weak and the pixels are many more than
the series, and the point at which it stops depends on the coordinates.  The fits are therefore computed with
tolerance 1e-8 and at most 100,000 iterations.  scikit-learn does not pass a limit on the number of evaluations of
the objective to SciPy, whose default is 15,000; this limit is removed here (``MAXFUN``).  In most fits L-BFGS stops
when the relative decrease of the objective falls below 64 machine epsilons.  The largest absolute component of the
gradient at termination is recorded for every fit.

    python analysis/realdata_logistic_regression.py <stage> --data <folder of the datasets>
           --meta analysis/realdata_studies_metadata.json --work <working folder> [--manifest <stored_array_hashes.jsonl>]
           [--study development|independent] [--datasets a,b] [--workers 2] [--out scores.csv]

Saved scores: results/realdata_scores_logistic.csv (scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TIE = 1e-12
TOL = 1e-8          # tolerance of L-BFGS on the gradient (scikit-learn default: 1e-4); see "Convergence" in the docstring
MAX_ITER = 100000
MAXFUN = 10_000_000  # evaluations of the objective allowed to SciPy's L-BFGS-B (its default, 15,000, is not set by scikit-learn)
PROPOSED = ("MQI", "MQI-LCF", "MQI-UDI")
EXISTING = ("GAF", "MTF", "RP", "CWT")
SEEDS = {**{r: (101, 202, 303) for r in PROPOSED}, **{r: (101, 202) for r in EXISTING}}
CRC_CWT = ("Beef", "InsectEPGRegularTrain", "NonInvasiveFetalECGThorax2", "SemgHandMovementCh2", "ShapesAll", "SyntheticControl")

ARGS = None
_MAN, _META = {}, {}


def sha_array(a) -> str:
    """Hash of the studies: sha256(str(dtype) + str(shape) + bytes)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def manifest():
    if not _MAN:
        for line in Path(ARGS.manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                _MAN[r["file"]] = r["members"]
    return _MAN


def meta():
    if not _META:
        _META.update(json.loads(Path(ARGS.meta).read_text(encoding="utf-8")))
    return _META


def datasets(study):
    return list(meta()["development"]["order"]) if study == "development" else sorted(meta()["independent"])


def representations(study):
    return PROPOSED + EXISTING if study == "development" else PROPOSED


# ------------------------------------------------------------------------------------------------- raw series
def _ts(path, values=True, labels=True):
    """Univariate .ts file with series of equal length (reader of the independent evaluation)."""
    vals, labs, in_data = [], [], False
    with open(path, "rb") as f:
        for raw in f:
            s = raw.strip()
            if not in_data:
                in_data = s.lower() == b"@data"
                continue
            if not s or s.startswith(b"#"):
                continue
            if s.count(b":") != 1:
                raise ValueError("record does not have exactly one class-label colon")
            ser, lab = s.rsplit(b":", 1)
            if values:
                vals.append(np.array(ser.split(b","), dtype=np.float64))
            if labels:
                labs.append(lab.decode("utf-8").strip())
    return (np.stack(vals) if values else None), (np.array(labs) if labels else None)


def load_train(study, ds):
    from sklearn.preprocessing import LabelEncoder
    if study == "development":
        m = meta()["development"]["datasets"][ds]
        z = np.load(Path(ARGS.data) / m["cached_path"], allow_pickle=False)
        X, y_raw = np.asarray(z["X_train"], dtype=np.float64), np.asarray(z["y_train"])
    else:
        m = meta()["independent"][ds]
        X, y_raw = _ts(Path(ARGS.data) / m["train_path"])
    if X.shape != (m["train_cases"], m["length"]) or not np.isfinite(X).all():
        raise ValueError(f"{ds}: training series do not have the recorded shape")
    le = LabelEncoder().fit(y_raw)
    if len(le.classes_) != m["classes"]:
        raise ValueError(f"{ds}: number of classes differs from the record")
    return X, le.transform(y_raw).astype(np.int64), le


def load_test_series(study, ds):
    if study == "development":
        m = meta()["development"]["datasets"][ds]
        X = np.asarray(np.load(Path(ARGS.data) / m["cached_path"], allow_pickle=False)["X_test"], dtype=np.float64)
    else:
        m = meta()["independent"][ds]
        X, _ = _ts(Path(ARGS.data) / m["test_path"], labels=False)
    if X.shape != (m["test_cases"], m["length"]) or not np.isfinite(X).all():
        raise ValueError(f"{ds}: test series do not have the recorded shape")
    return X


def load_test_labels(study, ds, le):
    """Read in the stage ``final`` only."""
    if study == "development":
        y_raw = np.asarray(np.load(Path(ARGS.data) / meta()["development"]["datasets"][ds]["cached_path"], allow_pickle=False)["y_test"])
    else:
        _, y_raw = _ts(Path(ARGS.data) / meta()["independent"][ds]["test_path"], values=False)
    return le.transform(y_raw).astype(np.int64)


def split(y, seed):
    from sklearn.model_selection import train_test_split
    fit_ix, val_ix = train_test_split(np.arange(len(y)), test_size=0.25, random_state=int(seed), stratify=y)
    return np.asarray(fit_ix), np.asarray(val_ix)


# ------------------------------------------------------------------------- arrays stored by the studies (hashes)
PSP = "Code/PCEI_Staged_Pilot/psp_20261109T1712Z/cache"
MDP = "Code/MDF_RMS_Development_Pilot/mdp_20261114T1238Z/cache"
LCF = "Code/LCF_Development_Pilot/lcf_20261114T1809Z/cache"
CRI = "Code/Common_Reader_Image_Comparison/cri_20261111T0406Z/artifacts"
CRC = "Code/CNN_Reader_Calibration/crc_20261111T0112Z/artifacts"
LVC = "Code/LCF_Independent_Validation_Continuation/lvc_20261115T0736Z/cache"


def stored(study, ds, rep, part):
    """Stored arrays that the recomputed images of one partition must reproduce.

    Returns {"fit": [(planes, file, member), ...], "val" or "test": [...], "index": {name: (file, member)}}.
    ``part`` is "inner_s<seed>" or "full".
    """
    inner = part != "full"
    second = "val" if inner else "test"
    out = {"fit": [], second: [], "index": {}}
    arm = {"MQI-LCF": "C", "MQI-UDI": "U"}.get(rep)
    if study == "development":
        if rep in PROPOSED:
            if part == "inner_s303":
                f = f"{MDP}/{ds}/inner_s303_P.npz"
                out["fit"].append((slice(0, 6), f, "Z_fit")); out["val"].append((slice(0, 6), f, "Z_val"))
                out["index"] = {"fit_ix": (f, "fit_ix"), "val_ix": (f, "val_ix"), "y_fit": (f, "y_fit"), "y_val": (f, "y_val")}
            else:
                f = f"{PSP}/{ds}/{'full_any' if not inner else part}_quant.npz"
                out["fit"].append((slice(0, 6), f, "Z_fit")); out[second].append((slice(0, 6), f, "Z_ev"))
                out["index"] = {"y_fit": (f, "y_fit")}
                if inner:
                    out["index"].update({"fit_ix": (f, "fit_ix"), "val_ix": (f, "val_ix"), "y_val": (f, "y_ev")})
            if arm:
                out["fit"].append((slice(6, 10), f"{LCF}/{ds}/{part}_{arm}.npz", "S_fit"))
                out[second].append((slice(6, 10), f"{LCF}/{ds}/{part}_{arm}.npz", "S_val") if inner
                                   else (slice(6, 10), f"{LCF}/{ds}/test_full_{arm}.npz", "S_test"))
        elif rep == "CWT":
            root = CRC if ds in CRC_CWT else CRI
            f = f"{root}/cwt/{ds}/{part}.npz"
            out["fit"].append((slice(0, 1), f, "Z_fit"))
            out[second].append((slice(0, 1), f, "Z_val") if inner else (slice(0, 1), f"{CRI}/cwt_test/{ds}.npz", "Z_test"))
            if inner:
                out["index"] = {"fit_ix": (f, "fit_ix"), "val_ix": (f, "val_ix")}
        else:
            f = f"{PSP}/{ds}/{'full_any' if not inner else part}_{rep.lower()}.npz"
            out["fit"].append((slice(0, 1), f, "Z_fit")); out[second].append((slice(0, 1), f, "Z_ev"))
            if inner:
                out["index"] = {"fit_ix": (f, "fit_ix"), "val_ix": (f, "val_ix")}
    else:
        f = f"{LVC}/{ds}/{part}_P.npz"
        out["fit"].append((slice(0, 6), f, "Z_fit"))
        out[second].append((slice(0, 6), f, "Z_val") if inner else (slice(0, 6), f"{LVC}/{ds}/test_full_P.npz", "S_test"))
        out["index"] = {"y_fit": (f, "y_fit")}
        if inner:
            out["index"].update({"fit_ix": (f, "fit_ix"), "val_ix": (f, "val_ix"), "y_val": (f, "y_val")})
        if arm:
            g = f"{LVC}/{ds}/{part}_{arm}.npz"
            out["fit"].append((slice(6, 10), g, "S_fit"))
            out[second].append((slice(6, 10), g, "S_val") if inner else (slice(6, 10), f"{LVC}/{ds}/test_full_{arm}.npz", "S_test"))
    return out


def check_images(images, expected):
    """[(file, member, identical)] for the recomputed images against the stored arrays (planes compared separately)."""
    res = []
    if not ARGS.manifest:
        return res
    for planes, f, member in expected:
        want = manifest()[f][member]
        res.append((f, member, sha_array(images[:, planes]) == want["sha256"]))
    return res


# ---------------------------------------------------------------------------------- reduction and classification
def reduce(F, others):
    """Coordinates in the row space of the training images (exact; see the module docstring)."""
    from scipy.linalg import qr
    F = F.reshape(len(F), -1).astype(np.float64)
    others = [B.reshape(len(B), -1).astype(np.float64) for B in others]
    n, p = F.shape
    if p <= n:
        return F, others, p, "pixels"
    Q, R = qr(F.T, mode="economic", overwrite_a=True)
    return np.ascontiguousarray(R.T), [B @ Q for B in others], p, "row space"


def fit(Z, y, C):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    _lift_evaluation_limit()
    m = LogisticRegression(C=C, solver="lbfgs", tol=TOL, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(Z, y)
    it = int(np.max(m.n_iter_))
    return m, it, int(any(issubclass(x.category, ConvergenceWarning) for x in w) or it >= MAX_ITER), gradient(m, Z, y, C)


_LIFTED = False


def _lift_evaluation_limit():
    """Pass ``maxfun=MAXFUN`` to the call of SciPy's minimizer made by scikit-learn's logistic regression."""
    global _LIFTED
    if _LIFTED:
        return
    import sklearn.linear_model._logistic as L
    minimize = L.optimize.minimize

    def patched(*a, **k):
        k["options"] = {**k.get("options", {}), "maxfun": MAXFUN}
        return minimize(*a, **k)
    L.optimize.minimize = patched
    _LIFTED = True


def gradient(m, Z, y, C):
    """Largest absolute component of the gradient of the penalized mean negative log-likelihood at the fitted
    coefficients and intercepts (the quantity that L-BFGS compares with the tolerance)."""
    n = len(y)
    P = m.predict_proba(Z)
    if P.shape[1] == 2:
        r = (P[:, 1] - (y == m.classes_[1])) / n
        g, g0 = Z.T @ r + m.coef_[0] / (C * n), r.sum()
        return float(max(np.abs(g).max(), abs(g0)))
    R_ = P.copy(); R_[np.arange(n), np.searchsorted(m.classes_, y)] -= 1.0; R_ /= n
    return float(max(np.abs(Z.T @ R_ + m.coef_.T / (C * n)).max(), np.abs(R_.sum(0)).max()))


def red_path(study, ds, rep, part):
    return Path(ARGS.work) / "reduced" / study / ds / f"{rep}__{part}.npz"


def json_path(stage, study, key):
    return Path(ARGS.work) / stage / f"{study}__{key}.json"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------ stages
def encode(task):
    study, ds = task
    out = json_path("encode", study, ds)
    if out.exists():
        return ds, "done before"
    sys.path.insert(0, str(Path(ARGS.repo)))
    from ts2i import make_encoder
    t0 = time.time()
    X, y, le = load_train(study, ds)
    Xt = load_test_series(study, ds)
    K = len(le.classes_)
    rec = {"study": study, "dataset": ds, "n_train": len(y), "n_test": len(Xt), "length": X.shape[1], "classes": K, "checks": [], "seconds": {}}
    for rep in representations(study):
        t1 = time.time()
        for seed in SEEDS[rep]:
            part = f"inner_s{seed}"
            fit_ix, val_ix = split(y, seed)
            if len(np.unique(y[fit_ix])) != K or len(np.unique(y[val_ix])) != K:
                raise ValueError(f"{ds} seed {seed}: a class is missing from a subset")
            enc = make_encoder(rep).fit(X[fit_ix])
            F, V = enc.transform(X[fit_ix]), enc.transform(X[val_ix])
            exp = stored(study, ds, rep, part)
            ok = check_images(F, exp["fit"]) + check_images(V, exp["val"])
            arrays = {"fit_ix": fit_ix, "val_ix": val_ix, "y_fit": y[fit_ix], "y_val": y[val_ix]}
            if ARGS.manifest:
                ok += [(f, member, sha_array(arrays[name]) == manifest()[f][member]["sha256"]) for name, (f, member) in exp["index"].items()]
            rec["checks"] += [{"representation": rep, "partition": part, "file": f, "member": member, "identical": bool(b)} for f, member, b in ok]
            Zf, (Zv,), p, how = reduce(F, [V])
            red_path(study, ds, rep, part).parent.mkdir(parents=True, exist_ok=True)
            np.savez(red_path(study, ds, rep, part), Zf=Zf, Zv=Zv, n_features=p, how=how)
        enc = make_encoder(rep).fit(X)
        A, B = enc.transform(X), enc.transform(Xt)
        exp = stored(study, ds, rep, "full")
        ok = check_images(A, exp["fit"]) + check_images(B, exp["test"])
        if ARGS.manifest:
            ok += [(f, member, sha_array(y) == manifest()[f][member]["sha256"]) for name, (f, member) in exp["index"].items()]
        rec["checks"] += [{"representation": rep, "partition": "full", "file": f, "member": member, "identical": bool(b)} for f, member, b in ok]
        Za, (Zb,), p, how = reduce(A, [B])
        np.savez(red_path(study, ds, rep, "full"), Za=Za, Zb=Zb, n_features=p, how=how)
        rec["seconds"][rep] = round(time.time() - t1, 2)
    rec["seconds"]["total"] = round(time.time() - t0, 2)
    rec["arrays_checked"] = len(rec["checks"])
    rec["arrays_identical"] = int(sum(c["identical"] for c in rec["checks"]))
    write_json(out, rec)
    return ds, f"{rec['arrays_identical']}/{rec['arrays_checked']} stored arrays reproduced, {rec['seconds']['total']:.0f} s"


def select(task):
    study, ds, rep, seed = task
    out = json_path("select", study, f"{ds}__{rep}__s{seed}")
    if out.exists():
        return f"{ds} {rep} s{seed}", "done before"
    from sklearn.metrics import log_loss
    t0 = time.time()
    _, y, le = load_train(study, ds)
    K = len(le.classes_)
    fit_ix, val_ix = split(y, seed)
    z = np.load(red_path(study, ds, rep, f"inner_s{seed}"))
    Zf, Zv = z["Zf"], z["Zv"]
    losses, its, nc, gr = {}, {}, {}, {}
    for C in C_GRID:
        m, it, bad, g = fit(Zf, y[fit_ix], C)
        losses[C] = float(log_loss(y[val_ix], m.predict_proba(Zv), labels=np.arange(K)))
        its[C], nc[C], gr[C] = it, bad, g
    best = min(losses.values())
    C_sel = min(C for C in C_GRID if losses[C] <= best + TIE)
    rec = {"study": study, "dataset": ds, "classes": K, "representation": rep, "seed": seed, "selected_C": C_sel, "validation_log_loss": best,
           "losses": {repr(C): losses[C] for C in C_GRID}, "iterations": {repr(C): its[C] for C in C_GRID},
           "not_converged": {repr(C): nc[C] for C in C_GRID}, "gradient": {repr(C): gr[C] for C in C_GRID},
           "n_features": int(z["n_features"]), "n_fit": len(fit_ix), "seconds": round(time.time() - t0, 2)}
    write_json(out, rec)
    return f"{ds} {rep} s{seed}", f"C {C_sel:g}, {rec['seconds']:.0f} s"


def final(task):
    study, ds, rep = task
    out = json_path("final", study, f"{ds}__{rep}")
    if out.exists():
        return f"{ds} {rep}", "done before"
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    t0 = time.time()
    _, y, le = load_train(study, ds)
    yt = load_test_labels(study, ds, le)
    K = len(le.classes_)
    sel = [json.loads(json_path("select", study, f"{ds}__{rep}__s{seed}").read_text(encoding="utf-8")) for seed in SEEDS[rep]]
    z = np.load(red_path(study, ds, rep, "full"))
    Za, Zb = z["Za"], z["Zb"]
    rows, cache = [], {}
    for s in sel:
        C = s["selected_C"]
        if C not in cache:
            m, it, bad, g = fit(Za, y, C)
            cache[C] = (m.predict(Zb), it, bad, g)
        pred, it, bad, g = cache[C]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bal = float(balanced_accuracy_score(yt, pred))
        rows.append({"study": study, "dataset": ds, "seed": s["seed"], "representation": rep,
                     "macro_f1": float(f1_score(yt, pred, average="macro", labels=np.arange(K), zero_division=0)),
                     "accuracy": float(accuracy_score(yt, pred)), "balanced_accuracy": bal,
                     "single_class_prediction": int(len(np.unique(pred)) == 1), "selected_C": C,
                     "validation_log_loss": s["validation_log_loss"], "n_features": s["n_features"], "n_train": len(y), "n_test": len(yt),
                     "classes": K, "iterations": it, "not_converged": bad, "gradient": g,
                     "selection_fits_not_converged": int(sum(s["not_converged"].values())),
                     "selection_largest_gradient": float(max(s["gradient"].values()))})
    write_json(out, {"study": study, "dataset": ds, "representation": rep, "rows": rows, "seconds": round(time.time() - t0, 2)})
    return f"{ds} {rep}", f"{len(rows)} rows, {time.time() - t0:.0f} s"


def collect(path):
    """Score file from the outputs of the stage ``final`` (one row per dataset, seed and representation)."""
    import csv
    rows = []
    for s in ARGS.study.split(","):
        for d in datasets(s):
            for rep in representations(s):
                rows += json.loads(json_path("final", s, f"{d}__{rep}").read_text(encoding="utf-8"))["rows"]
    rows.sort(key=lambda r: (r["study"], r["dataset"], r["seed"], r["representation"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", path, len(rows), "rows")


def cost(study, ds, rep, full=False):
    """Rough size of a task (rows x columns x classes), used only to order the tasks."""
    m = meta()["development"]["datasets"][ds] if study == "development" else meta()["independent"][ds]
    n = m["train_cases"] if full else round(0.75 * m["train_cases"])
    p = 1024 if rep in EXISTING else n
    return n * min(n, p) * (m["classes"] if m["classes"] > 2 else 1)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("encode", "select", "final", "collect"))
    ap.add_argument("--data", required=True); ap.add_argument("--manifest", default=""); ap.add_argument("--meta", required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--study", default="development,independent"); ap.add_argument("--datasets", default="")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--out", default="")
    ARGS = ap.parse_args()
    if ARGS.stage == "collect":
        return collect(ARGS.out)
    only = set(filter(None, ARGS.datasets.split(",")))
    pairs = [(s, d) for s in ARGS.study.split(",") for d in datasets(s) if not only or d in only]
    if ARGS.stage == "encode":
        tasks = pairs
    elif ARGS.stage == "select":       # one task per dataset, representation and division, the largest first
        tasks = sorted(((s, d, r, seed) for s, d in pairs for r in representations(s) for seed in SEEDS[r]), key=lambda k: -cost(*k[:3]))
    else:
        tasks = sorted(((s, d, r) for s, d in pairs for r in representations(s)), key=lambda k: -cost(*k, full=True))
    fn = {"encode": encode, "select": select, "final": final}[ARGS.stage]
    t0 = time.time()
    if ARGS.workers > 1:
        from multiprocessing import Pool
        with Pool(ARGS.workers) as pool:
            for i, (key, msg) in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
                print(f"[{ARGS.stage} {i}/{len(tasks)}] {key}: {msg}  ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            key, msg = fn(task)
            print(f"[{ARGS.stage} {i}/{len(tasks)}] {key}: {msg}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
