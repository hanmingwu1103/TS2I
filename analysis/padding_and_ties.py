"""Padding, ties and clipping in the datasets of the two studies on real data (descriptive; no network is trained).

Thirteen of the 17 datasets of the independent evaluation are equal-length versions of collections of series of
unequal length, in which each shorter series is extended at its end by low-amplitude noise.  The paper uses the series
as supplied.  This script describes the padding and two related properties of the data:

  padding      For each series of the independent evaluation, the padded part is estimated as the final run of blocks
               of ten consecutive observations, counted from the end of the series, whose standard deviation is less
               than one tenth of the largest block standard deviation of that series (Appendix C of the paper).  The
               output gives, by dataset, the share of series with such a tail, the median and the mean share of the
               length that it takes, the smallest and the largest class mean of this share over the training series
               (Section 6.4 of the paper), the difference in level between the tail and the rest of the series, and
               the share of windows that lie entirely in the tail.
  length only  Classification of the test series from the estimated original length alone, by a nearest-neighbor
               classifier whose number of neighbors is chosen by cross-validation on the training set.  Not reported
               in the paper.
  ties         For both studies, the number of distinct clipped arrival levels in the windows of each scale, relative
               to the window length, and the share of test arrival levels changed by clipping.
  sign         Whether the range of the lag difference contains zero, for the whole training set and for the three
               fitting subsets of each dataset.  If it does, clipping cannot change the sign of an increment
               (Appendix A of the paper).

    python analysis/padding_and_ties.py --data <folder of the datasets> [--meta analysis/realdata_studies_metadata.json]
           [--work <folder for one file per dataset>] [--out results/padding_and_ties.json]

``--meta`` lists the datasets and, relative to ``--data``, their files (see analysis/realdata_logistic_regression.py).
The datasets are not distributed with this repository.  Saved output: results/padding_and_ties.json;
analysis/summarize_padding_and_ties.py prints its summaries.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ts2i.representations import FRACTIONS, TimeSeriesImageEncoder, projections, window_layout  # noqa: E402

SEEDS = (101, 202, 303)
BLOCK, RATIO = 10, 0.10
DATA, META = None, None


def read_ts(path):
    X, y = [], []
    data = False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            data = True
            continue
        if not data:
            continue
        values, label = line.rsplit(":", 1)
        X.append([float(v) for v in values.split(",")]); y.append(label)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def load(study, name):
    if study == "development":
        z = np.load(DATA / META["development"]["datasets"][name]["cached_path"], allow_pickle=True)
        return (np.asarray(z["X_train"], dtype=np.float64), np.asarray(z["y_train"]).astype(str),
                np.asarray(z["X_test"], dtype=np.float64), np.asarray(z["y_test"]).astype(str))
    m = META["independent"][name]
    Xtr, ytr = read_ts(DATA / m["train_path"]); Xte, yte = read_ts(DATA / m["test_path"])
    return Xtr, ytr, Xte, yte


def tail_length(x):
    """Number of observations in the padded tail of one series (0 if none)."""
    n = len(x) // BLOCK
    if n < 4:
        return 0
    sd = x[len(x) - n * BLOCK:].reshape(n, BLOCK).std(axis=1)
    ref = sd.max()
    if ref <= 0:
        return 0
    k = 0
    for s in sd[::-1]:
        if s < RATIO * ref:
            k += 1
        else:
            break
    return k * BLOCK


def padding(X_train, y_train, X_test):
    X = np.concatenate([X_train, X_test]); T = X.shape[1]
    k = np.array([tail_length(x) for x in X]); share = k / T
    ktr = k[:len(X_train)]
    by_class = [float(np.mean(ktr[y_train == c]) / T) for c in sorted(set(y_train))]
    rel = []
    for x, kk in zip(X, k):
        if kk and kk < T - 5:
            head, tail = x[:T - kk], x[T - kk:]
            if head.std() > 0:
                rel.append(abs(tail.mean() - head.mean()) / head.std())
    N = T - 1
    inside = {}
    for f in FRACTIONS:
        starts, L = window_layout(N, f)
        # window k covers the positions starts[k] .. starts[k]+L-1, that is the observations starts[k] .. starts[k]+L
        inside[str(f)] = float(np.mean([(starts >= T - kk).mean() if kk else 0.0 for kk in k]))
    return {"series_with_tail": float(np.mean(k > 0)), "median_share": float(np.median(share)), "mean_share": float(share.mean()),
            "class_mean_share_min": min(by_class), "class_mean_share_max": max(by_class),
            "median_tail_minus_rest_in_sd": (float(np.median(rel)) if rel else None), "windows_inside_tail": inside,
            "original_length_train": (T - ktr).tolist(), "original_length_test": (T - k[len(X_train):]).tolist()}


def ties(X_train, X_test):
    enc = TimeSeriesImageEncoder("MQI").fit(X_train)
    low, high = enc.ranges_[0]
    out = {}
    X = np.concatenate([X_train, X_test])
    P = projections((X - enc.mean_) / enc.scale_)[:, 0]
    a = np.clip(P, low, high)
    Pt = P[len(X_train):]
    out["test_levels_clipped"] = float(np.mean((Pt < low) | (Pt > high))) if len(Pt) else 0.0
    N = a.shape[1]
    for f in FRACTIONS:
        starts, L = window_layout(N, f)
        idx = starts[:, None] + np.arange(L)[None, :]
        ratio_sum, any_tie, heavy, count = 0.0, 0, 0, 0
        step = max(1, int(2e7 // (32 * L)))
        for i in range(0, len(a), step):
            A = np.sort(a[i:i + step][:, idx], axis=-1)
            distinct = 1 + (np.diff(A, axis=-1) != 0).sum(axis=-1)
            r = distinct / L
            ratio_sum += float(r.sum()); any_tie += int((distinct < L).sum()); heavy += int((r < 0.5).sum()); count += r.size
        out[str(f)] = {"window_length": int(L), "mean_distinct_ratio": ratio_sum / count, "share_any_tie": any_tie / count, "share_ratio_below_half": heavy / count}
    return out


def sign_change(X_train, y_train):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    d = np.diff(X_train, axis=1); lo, hi = d.min(axis=1), d.max(axis=1)
    y = LabelEncoder().fit_transform(y_train)
    res = {"whole_training_set": bool(lo.min() <= 0.0 <= hi.max())}
    for seed in SEEDS:
        fit_ix, _ = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
        res[f"fitting_subset_seed_{seed}"] = bool(lo[fit_ix].min() <= 0.0 <= hi[fit_ix].max())
    res["series_with_both_signs"] = float(np.mean((lo < 0) & (hi > 0)))
    return res


def length_only(orig_train, y_train, orig_test, y_test):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_train); y, yt = le.transform(y_train), le.transform(y_test); K = len(le.classes_)
    Z, Zt = np.asarray(orig_train, dtype=float)[:, None], np.asarray(orig_test, dtype=float)[:, None]
    folds = int(min(5, np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    smallest_fit = min(len(tr) for tr, _ in cv.split(Z, y))
    grid = [k for k in (1, 5, 15, 31) if k <= smallest_fit - 1]
    cvscore = {}
    for k in grid:
        s = []
        for tr, va in cv.split(Z, y):
            m = KNeighborsClassifier(n_neighbors=k).fit(Z[tr], y[tr])
            s.append(f1_score(y[va], m.predict(Z[va]), average="macro", labels=np.arange(K), zero_division=0))
        cvscore[k] = float(np.mean(s))
    best = max(cvscore.values()); k = max(kk for kk, v in cvscore.items() if v >= best - 1e-12)
    m = KNeighborsClassifier(n_neighbors=k).fit(Z, y)
    return {"k": int(k), "folds": folds, "cv_macro_f1": cvscore, "test_macro_f1": float(f1_score(yt, m.predict(Zt), average="macro", labels=np.arange(K), zero_division=0)),
            "classes": int(K), "one_over_K": 1.0 / K}


def main():
    global DATA, META
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", default=str(REPO / "analysis" / "realdata_studies_metadata.json"))
    ap.add_argument("--work", default="padding_and_ties_work", help="folder for one file per dataset; finished datasets are kept")
    ap.add_argument("--out", default=str(REPO / "results" / "padding_and_ties.json"))
    ap.add_argument("--datasets", default="", help="comma-separated names (for a trial); the output file is then not written")
    args = ap.parse_args()
    DATA, META = Path(args.data), json.loads(Path(args.meta).read_text())
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.datasets.split(",")))
    res = {"development": {}, "independent": {}}
    for study, names in (("development", META["development"]["order"]), ("independent", sorted(META["independent"]))):
        for name in names:
            if only and name not in only:
                continue
            f = work / f"{study}__{name}.json"
            if f.exists():
                res[study][name] = json.loads(f.read_text()); continue
            Xtr, ytr, Xte, yte = load(study, name)
            r = {"n_train": len(Xtr), "n_test": len(Xte), "T": int(Xtr.shape[1]), "ties": ties(Xtr, Xte), "sign": sign_change(Xtr, ytr)}
            if study == "independent":
                p = padding(Xtr, ytr, Xte)
                r["length_only"] = length_only(p.pop("original_length_train"), ytr, p.pop("original_length_test"), yte)
                r["padding"] = p
            f.write_text(json.dumps(r, indent=1)); res[study][name] = r
            print(study, name, "done", flush=True)
    if not only:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
