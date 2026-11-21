"""Summaries of the logistic regression on real data.

The summaries are those of the corresponding analyses with the CNN (analysis/reproduce_tables.py, whose functions are
used here), without the tests and criteria of those analyses: the results of the logistic regression are descriptive.
They were fixed before the labels of the test series were read for this analysis.

    python analysis/summarize_realdata_logistic.py [repository] [realdata_scores_logistic.csv] [summary.json]
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import reproduce_tables as P  # noqa: E402  (bootstrap_interval, t_interval, wtl, seed_means, development_datasets, TIE)

PROPOSED, EXISTING = ("MQI", "MQI-LCF", "MQI-UDI"), ("CWT", "RP", "GAF", "MTF")
CONTRASTS = (("MQI-LCF", "MQI"), ("MQI-LCF", "MQI-UDI"), ("MQI-UDI", "MQI"))
OUT = {}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def existing(lr, cnn_dev, cnn_old):
    datasets = P.development_datasets()
    vec = lambda table, rep: np.array([table[rep][d] for d in datasets])
    f1, ac = P.seed_means(lr, seeds=(101, 202)), P.seed_means(lr, "accuracy", seeds=(101, 202))
    c1 = {**P.seed_means(cnn_dev, seeds=(101, 202)), **{k: v for k, v in P.seed_means(cnn_old).items() if k in EXISTING}}
    print("\nA. Seven representations, 30 development datasets, divisions of seeds 101 and 202 (as Table 3 of the paper)")
    res = {}
    for rep in ("MQI-LCF", "MQI-UDI", "MQI") + EXISTING:
        r = {"macro_f1": float(vec(f1, rep).mean()), "accuracy": float(vec(ac, rep).mean()), "macro_f1_cnn": float(vec(c1, rep).mean())}
        line = f"  {rep:8s} macro-F1 {r['macro_f1']:6.2f}  accuracy {r['accuracy']:6.2f}   (CNN macro-F1 {r['macro_f1_cnn']:6.2f})"
        if rep in EXISTING:
            e = vec(f1, "MQI") - vec(f1, rep)
            lo, hi = P.bootstrap_interval(e, 20261111)
            r["MQI_minus"] = {"mean": float(e.mean()), "interval": [float(lo), float(hi)], "wtl": P.wtl(e)}
            line += f"   MQI - {rep}: mean {e.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] W/T/L {P.wtl(e)}"
        if rep != "MQI-LCF":
            e = vec(f1, "MQI-LCF") - vec(f1, rep)
            r["MQI-LCF_minus"] = {"mean": float(e.mean()), "wtl": P.wtl(e)}
            line += f"   MQI-LCF - {rep}: mean {e.mean():+6.2f} W/T/L {P.wtl(e)}"
        res[rep] = r
        print(line)
    order = sorted(res, key=lambda k: -res[k]["macro_f1"]); order_cnn = sorted(res, key=lambda k: -res[k]["macro_f1_cnn"])
    print("  order of the means: logistic regression " + " > ".join(order) + ";  CNN " + " > ".join(order_cnn))
    OUT["existing"] = {"by_representation": res, "order": order, "order_cnn": order_cnn}


def development(lr, cnn_dev):
    datasets = P.development_datasets()
    f1, c1 = P.seed_means(lr, seeds=(101, 202, 303)), P.seed_means(cnn_dev)
    f1 = {k: v for k, v in f1.items() if k in PROPOSED}
    vec = lambda table, rep: np.array([table[rep][d] for d in datasets])
    print("\nB. Three proposed representations, 30 development datasets, divisions of seeds 101, 202 and 303 (as Table 4 of the paper, upper panel)")
    print("  mean macro-F1: " + ", ".join(f"{rep} {vec(f1, rep).mean():.2f} (CNN {vec(c1, rep).mean():.2f})" for rep in PROPOSED))
    res = {"means": {rep: float(vec(f1, rep).mean()) for rep in PROPOSED}, "means_cnn": {rep: float(vec(c1, rep).mean()) for rep in PROPOSED}, "contrasts": {}}
    for a, b in CONTRASTS:
        e, ec = vec(f1, a) - vec(f1, b), vec(c1, a) - vec(c1, b)
        lo, hi = P.bootstrap_interval(e, 20261115)
        loo = [np.delete(e, i).mean() for i in range(len(e))]
        res["contrasts"][f"{a} - {b}"] = {"mean": float(e.mean()), "median": float(np.median(e)), "wtl": P.wtl(e), "interval": [float(lo), float(hi)],
                                         "leave_one_out": [float(min(loo)), float(max(loo))], "mean_cnn": float(ec.mean()), "spearman_with_cnn": spearman(e, ec)}
        print(f"  {a + ' - ' + b:18s} mean {e.mean():+5.2f}  median {np.median(e):+5.2f}  W/T/L {P.wtl(e)}  95% interval [{lo:+.2f}, {hi:+.2f}]  one dataset left out "
              f"[{min(loo):+.2f}, {max(loo):+.2f}]   CNN mean {ec.mean():+5.2f};  Spearman correlation of the effects with those of the CNN {spearman(e, ec):+.2f}")
    by_seed = {s: float(np.mean([P.seed_means(lr, seeds=(s,))["MQI-LCF"][d] - P.seed_means(lr, seeds=(s,))["MQI"][d] for d in datasets])) for s in (101, 202, 303)}
    print("  MQI-LCF - MQI by division: " + ", ".join(f"{s}: {v:+.2f}" for s, v in by_seed.items()))
    res["by_seed"] = by_seed
    for metric in ("accuracy", "balanced_accuracy"):
        t = P.seed_means(lr, metric, seeds=(101, 202, 303))
        res[f"MQI-LCF - MQI in {metric}"] = float(np.mean([t["MQI-LCF"][d] - t["MQI"][d] for d in datasets]))
        print(f"  MQI-LCF - MQI in {metric}: {res[f'MQI-LCF - MQI in {metric}']:+.2f}")
    print("  macro-F1 by dataset, logistic regression (MQI, MQI-LCF, MQI-UDI) and CNN (MQI, MQI-LCF, MQI-UDI):")
    res["by_dataset"] = {}
    for d in sorted(datasets):
        res["by_dataset"][d] = {"logistic": [f1[r][d] for r in PROPOSED], "cnn": [c1[r][d] for r in PROPOSED]}
        print(f"    {d:30s} " + " ".join(f"{f1[r][d]:6.2f}" for r in PROPOSED) + "   " + " ".join(f"{c1[r][d]:6.2f}" for r in PROPOSED))
    OUT["development"] = res


def independent(lr, cnn_ind):
    block_of = {r["dataset"]: r["block"] for r in cnn_ind}
    blocks = sorted(set(block_of.values()))
    f1, c1 = P.seed_means(lr), P.seed_means(cnn_ind)

    def block_means(table, rep):
        return np.array([np.mean([v for d, v in table[rep].items() if block_of[d] == b]) for b in blocks])

    print("\nC. Three proposed representations, independent evaluation, 17 datasets in nine blocks, three divisions (as Table 4 of the paper, lower panel)")
    print("  block-mean macro-F1: " + ", ".join(f"{rep} {block_means(f1, rep).mean():.2f} (CNN {block_means(c1, rep).mean():.2f})" for rep in PROPOSED))
    res = {"means": {rep: float(block_means(f1, rep).mean()) for rep in PROPOSED}, "means_cnn": {rep: float(block_means(c1, rep).mean()) for rep in PROPOSED},
           "contrasts": {}, "blocks": blocks}
    for a, b in CONTRASTS:
        e, ec = block_means(f1, a) - block_means(f1, b), block_means(c1, a) - block_means(c1, b)
        lo, hi = P.t_interval(e)
        loo = [np.delete(e, i).mean() for i in range(len(e))]
        res["contrasts"][f"{a} - {b}"] = {"mean": float(e.mean()), "median": float(np.median(e)), "sd": float(e.std(ddof=1)), "interval": [float(lo), float(hi)],
                                         "wins": int((e > P.TIE).sum()), "losses": int((e < -P.TIE).sum()), "by_block": [float(x) for x in e],
                                         "leave_one_out": [float(min(loo)), float(max(loo))], "mean_cnn": float(ec.mean()), "by_block_cnn": [float(x) for x in ec],
                                         "spearman_with_cnn": spearman(e, ec)}
        print(f"  {a + ' - ' + b:18s} mean {e.mean():+6.2f}  median {np.median(e):+5.2f}  SD {e.std(ddof=1):.2f}  95% interval [{lo:+.2f}, {hi:+.2f}]  W/L {int((e > P.TIE).sum())}/{int((e < -P.TIE).sum())}"
              f"  one block left out [{min(loo):+.2f}, {max(loo):+.2f}]   CNN mean {ec.mean():+5.2f};  Spearman correlation with the CNN {spearman(e, ec):+.2f}")
        print("      by block: " + ", ".join(f"{bl} {x:+.2f} (CNN {xc:+.2f})" for bl, x, xc in zip(blocks, e, ec)))
    by_seed = {s: float((block_means(P.seed_means(lr, seeds=(s,)), "MQI-LCF") - block_means(P.seed_means(lr, seeds=(s,)), "MQI")).mean()) for s in (101, 202, 303)}
    print("  MQI-LCF - MQI by division: " + ", ".join(f"{s}: {v:+.2f}" for s, v in by_seed.items()))
    res["by_seed"] = by_seed
    for metric in ("accuracy", "balanced_accuracy"):
        t = P.seed_means(lr, metric)
        res[f"MQI-LCF - MQI in {metric}"] = float((block_means(t, "MQI-LCF") - block_means(t, "MQI")).mean())
        print(f"  MQI-LCF - MQI in {metric}: {res[f'MQI-LCF - MQI in {metric}']:+.2f}")
    print("  macro-F1 by dataset (block), logistic regression (MQI, MQI-LCF, MQI-UDI) and CNN (MQI, MQI-LCF, MQI-UDI):")
    res["by_dataset"] = {}
    for d in sorted(block_of, key=lambda d: (block_of[d], d)):
        res["by_dataset"][d] = {"block": block_of[d], "logistic": [f1[r][d] for r in PROPOSED], "cnn": [c1[r][d] for r in PROPOSED]}
        print(f"    {d:26s} {block_of[d]:15s} " + " ".join(f"{f1[r][d]:6.2f}" for r in PROPOSED) + "   " + " ".join(f"{c1[r][d]:6.2f}" for r in PROPOSED))
    OUT["independent"] = res


def diagnostics(rows):
    print("\nD. Diagnostics of the fits")
    res = {}
    for study in ("development", "independent"):
        r = [x for x in rows if x["study"] == study]
        sel = Counter((x["representation"], float(x["selected_C"])) for x in r)
        res[study] = {"rows": len(r), "final_fits_not_converged": int(sum(int(x["not_converged"]) for x in r)),
                      "selection_fits_not_converged": int(sum(int(x["selection_fits_not_converged"]) for x in r)),
                      "single_class_predictions": int(sum(int(x["single_class_prediction"]) for x in r)),
                      "max_iterations_final": int(max(int(x["iterations"]) for x in r)),
                      "largest_gradient_final": float(max(float(x["gradient"]) for x in r)),
                      "largest_gradient_selection": float(max(float(x["selection_largest_gradient"]) for x in r)),
                      "selected_C": {rep: {repr(C): n for (rp, C), n in sorted(sel.items()) if rp == rep} for rep in sorted({x["representation"] for x in r})}}
        print(f"  {study}: {len(r)} rows; final fits not converged {res[study]['final_fits_not_converged']}; fits of the selection not converged "
              f"{res[study]['selection_fits_not_converged']}; single-class predictions {res[study]['single_class_predictions']}; largest number of iterations of a final fit {res[study]['max_iterations_final']}; "
              f"largest absolute component of the gradient at termination: final fits {res[study]['largest_gradient_final']:.1e}, fits of the selection {res[study]['largest_gradient_selection']:.1e}")
        for rep, c in res[study]["selected_C"].items():
            print(f"    selected C, {rep:8s}: " + ", ".join(f"{k}: {v}" for k, v in c.items()))
    OUT["diagnostics"] = res


if __name__ == "__main__":
    rows = read(sys.argv[2] if len(sys.argv) > 2 else REPO / "results" / "realdata_scores_logistic.csv")
    cnn_dev, cnn_old, cnn_ind = P.read("development_scores.csv"), P.read("existing_encodings_scores.csv"), P.read("independent_scores.csv")
    dev, ind = [r for r in rows if r["study"] == "development"], [r for r in rows if r["study"] == "independent"]
    print("Logistic regression on the pixels, studies on real data (macro-F1 and accuracy in percent; all summaries descriptive)")
    existing(dev, cnn_dev, cnn_old)
    development(dev, cnn_dev)
    independent(ind, cnn_ind)
    diagnostics(rows)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(OUT, indent=1, sort_keys=True), encoding="utf-8")
