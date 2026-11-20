"""Recompute the numerical results of the paper from the score files in ``results/``.

    python analysis/reproduce_tables.py

Scores are macro-F1 and accuracy on the test sets, one row per dataset (or simulation replicate), seed and
representation.  The script prints the entries of the tables of the simulation study and of the empirical
study.  The adjusted p-values of the comparison of MQI with the existing image encodings are not recomputed.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
TIE = 1e-8                                    # effects of at most this size count as ties
PROPOSED = ("MQI-LCF", "MQI-UDI", "MQI")
EXISTING = ("CWT", "RP", "GAF", "MTF")


def read(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = running
    return out


def sign_test(effects) -> float:
    """Exact two-sided sign test; ties are excluded."""
    wins, losses = int((effects > TIE).sum()), int((effects < -TIE).sum())
    return float(stats.binomtest(wins, wins + losses, 0.5).pvalue)


def wtl(effects) -> str:
    return f"{int((effects > TIE).sum())}/{int((np.abs(effects) <= TIE).sum())}/{int((effects < -TIE).sum())}"


def bootstrap_interval(effects, seed: int, B: int = 10000):
    """95% percentile interval for the mean from B resamples of the datasets.

    ``effects`` must be in the order of results/development_datasets.txt, the order used in the paper.
    """
    n = len(effects)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    return np.percentile(effects[idx].mean(axis=1), [2.5, 97.5])


def t_interval(effects):
    n = len(effects)
    half = stats.t.ppf(0.975, n - 1) * effects.std(ddof=1) / np.sqrt(n)
    return effects.mean() - half, effects.mean() + half


def development_datasets():
    return (RESULTS / "development_datasets.txt").read_text(encoding="utf-8").split()


def seed_means(rows, metric="macro_f1", seeds=None):
    """{representation: {dataset: mean over seeds}} in percent."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if seeds is None or int(r["seed"]) in seeds:
            acc[r["representation"]][r["dataset"]].append(100.0 * float(r[metric]))
    return {rep: {d: float(np.mean(v)) for d, v in by.items()} for rep, by in acc.items()}


# ----------------------------------------------------------------------------------------------- simulation
def simulation():
    rows = read("simulation_scores.csv")
    settings = ("A35", "A70", "NOISE", "TIES", "LEVEL", "NULL")
    f1, acc, single = defaultdict(dict), defaultdict(dict), defaultdict(int)
    for r in rows:
        key = (r["setting"], r["representation"])
        f1[key][int(r["replicate"])] = 100.0 * float(r["macro_f1"])
        acc[key][int(r["replicate"])] = 100.0 * float(r["accuracy"])
        single[key] += int(r["single_class_prediction"])
    print("Simulation study: mean macro-F1 (SD over 20 replicates), mean accuracy, models predicting a single class")
    for s in settings:
        for rep in ("MQI", "MQI-LCF", "MQI-UDI"):
            a, b = np.array(list(f1[(s, rep)].values())), np.array(list(acc[(s, rep)].values()))
            print(f"  {s:6s} {rep:8s} macro-F1 {a.mean():6.2f} ({a.std(ddof=1):5.2f})   accuracy {b.mean():6.2f}   single class {single[(s, rep)]}")
    reps = sorted(f1[("A35", "MQI")])
    primary = lambda table, rep: np.array([(table[("A35", rep)][r] + table[("A70", rep)][r]) / 2 for r in reps])
    print("  Prespecified contrasts (A35 and A70 averaged within each replicate; one-sided paired t tests, Holm adjustment):")
    pvals, lines = {}, {}
    for other in ("MQI", "MQI-UDI"):
        e = primary(f1, "MQI-LCF") - primary(f1, other)
        t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
        pvals[other] = float(stats.t.sf(t, len(e) - 1))
        lo, hi = t_interval(e)
        ea = primary(acc, "MQI-LCF") - primary(acc, other)
        lines[other] = (f"  MQI-LCF - {other:8s} mean {e.mean():6.2f}  95% interval [{lo:.2f}, {hi:.2f}]  t = {t:.2f}  range of the effects "
                        f"[{e.min():.2f}, {e.max():.2f}]  accuracy gain {ea.mean():.2f}")
    adj = holm(pvals)
    for other in lines:
        print(lines[other] + f"  adjusted p = {adj[other]:.2g}")


# ------------------------------------------------------------------------- comparison with existing encodings
def existing_encodings():
    dev, old = read("development_scores.csv"), read("existing_encodings_scores.csv")
    f1 = {**seed_means(dev, seeds=(101, 202)), **{k: v for k, v in seed_means(old).items() if k in EXISTING}}
    ac = {**seed_means(dev, "accuracy", seeds=(101, 202)), **{k: v for k, v in seed_means(old, "accuracy").items() if k in EXISTING}}
    mqi_old = seed_means(old)["MQI"]                     # MQI as saved with the existing encodings (six decimals)
    datasets = development_datasets()
    vec = lambda table: np.array([table[d] for d in datasets])
    print("\nComparison with existing image encodings, 30 development datasets, seeds 101 and 202")
    for rep in PROPOSED + EXISTING:
        line = f"  {rep:8s} macro-F1 {vec(f1[rep]).mean():6.2f}  accuracy {vec(ac[rep]).mean():6.2f}"
        if rep in EXISTING:
            e = vec(mqi_old) - vec(f1[rep])
            lo, hi = bootstrap_interval(e, 20261111)
            line += f"   MQI - {rep}: mean {e.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] W/T/L {wtl(e)}"
        if rep != "MQI-LCF":
            e = vec(f1["MQI-LCF"]) - vec(f1[rep])
            line += f"   MQI-LCF - {rep}: mean {e.mean():+6.2f} W/T/L {wtl(e)}"
        print(line)


# ------------------------------------------------------------------------------------------ development study
def development():
    rows = read("development_scores.csv")
    f1 = seed_means(rows)
    datasets = development_datasets()
    vec = lambda rep, table=f1: np.array([table[rep][d] for d in datasets])
    print("\nDevelopment study, 30 datasets, seeds 101, 202 and 303")
    print("  mean macro-F1: " + ", ".join(f"{rep} {vec(rep).mean():.2f}" for rep in ("MQI", "MQI-LCF", "MQI-UDI")))
    contrasts = {"MQI-LCF - MQI": ("MQI-LCF", "MQI"), "MQI-LCF - MQI-UDI": ("MQI-LCF", "MQI-UDI"), "MQI-UDI - MQI": ("MQI-UDI", "MQI")}
    effects = {k: vec(a) - vec(b) for k, (a, b) in contrasts.items()}
    adj = holm({k: sign_test(effects[k]) for k in ("MQI-LCF - MQI", "MQI-LCF - MQI-UDI")})
    for k, e in effects.items():
        lo, hi = bootstrap_interval(e, 20261115)
        p = f"  adjusted p = {adj[k]:.3f}" if k in adj else ""
        print(f"  {k:18s} mean {e.mean():+5.2f}  median {np.median(e):+5.2f}  W/T/L {wtl(e)}  95% interval [{lo:+.2f}, {hi:+.2f}]{p}")
    e = effects["MQI-LCF - MQI"]
    loo = [np.delete(e, i).mean() for i in range(len(e))]
    print(f"  MQI-LCF - MQI, one dataset left out: [{min(loo):+.2f}, {max(loo):+.2f}];  by seed: "
          + ", ".join(f"{s}: {np.mean([seed_means(rows, seeds=(s,))['MQI-LCF'][d] - seed_means(rows, seeds=(s,))['MQI'][d] for d in datasets]):+.2f}" for s in (101, 202, 303)))
    for metric in ("accuracy", "balanced_accuracy"):
        t = seed_means(rows, metric)
        print(f"  MQI-LCF - MQI in {metric}: {np.mean([t['MQI-LCF'][d] - t['MQI'][d] for d in datasets]):+.2f}")
    print("  macro-F1 by dataset (MQI, MQI-LCF, MQI-UDI):")
    for d in sorted(datasets):
        print(f"    {d:30s} {f1['MQI'][d]:6.2f} {f1['MQI-LCF'][d]:6.2f} {f1['MQI-UDI'][d]:6.2f}")


# --------------------------------------------------------------------------------------- independent evaluation
def independent():
    rows = read("independent_scores.csv")
    block_of = {r["dataset"]: r["block"] for r in rows}
    f1 = seed_means(rows)
    blocks = sorted(set(block_of.values()))

    def block_means(table, rep):
        return np.array([np.mean([v for d, v in table[rep].items() if block_of[d] == b]) for b in blocks])

    print("\nIndependent evaluation, 17 datasets in nine blocks, seeds 101, 202 and 303")
    print("  block-mean macro-F1: " + ", ".join(f"{rep} {block_means(f1, rep).mean():.2f}" for rep in ("MQI", "MQI-LCF", "MQI-UDI")))
    for a, b in (("MQI-LCF", "MQI"), ("MQI-LCF", "MQI-UDI"), ("MQI-UDI", "MQI")):
        e = block_means(f1, a) - block_means(f1, b)
        lo, hi = t_interval(e)
        line = (f"  {a + ' - ' + b:18s} mean {e.mean():+6.3f}  median {np.median(e):+5.2f}  SD {e.std(ddof=1):.2f}  95% interval [{lo:+.2f}, {hi:+.2f}]"
                f"  W/L {int((e > TIE).sum())}/{int((e < -TIE).sum())}")
        if (a, b) == ("MQI-LCF", "MQI"):
            t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
            line += f"  one-sided p = {stats.t.sf(t, len(e) - 1):.3f}"
            loo = [np.delete(e, i).mean() for i in range(len(e))]
            line += f"  one block left out [{min(loo):+.2f}, {max(loo):+.2f}]"
            by_block = "  effects by block: " + ", ".join(f"{bl} {x:+.2f}" for bl, x in zip(blocks, e))
        print(line)
    print(by_block)
    print("  MQI-LCF - MQI by seed: " + ", ".join(
        f"{s}: {(block_means(seed_means(rows, seeds=(s,)), 'MQI-LCF') - block_means(seed_means(rows, seeds=(s,)), 'MQI')).mean():+.2f}" for s in (101, 202, 303)))
    for metric in ("accuracy", "balanced_accuracy"):
        t = seed_means(rows, metric)
        print(f"  MQI-LCF - MQI in {metric}: {(block_means(t, 'MQI-LCF') - block_means(t, 'MQI')).mean():+.2f}")
    print("  macro-F1 by dataset (block; MQI, MQI-LCF, MQI-UDI):")
    for d in sorted(block_of, key=lambda d: (block_of[d], d)):
        print(f"    {d:26s} {block_of[d]:15s} {f1['MQI'][d]:6.2f} {f1['MQI-LCF'][d]:6.2f} {f1['MQI-UDI'][d]:6.2f}")


if __name__ == "__main__":
    simulation()
    existing_encodings()
    development()
    independent()
