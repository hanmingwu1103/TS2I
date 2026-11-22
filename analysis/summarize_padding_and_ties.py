"""Summaries of the description of padding, ties and clipping (results/padding_and_ties.json).

    python analysis/summarize_padding_and_ties.py

Prints the statements of the paper that rest on this file (Appendix A: ranges of the lag difference; Appendix C:
estimated padded part; Section 6.4: class means of the padded part in the Asphalt block) and the tables behind
them, together with descriptive quantities that are not reported in the paper (classification from the estimated
length alone, ties among arrival levels, rank correlations with the effects).  Macro-F1 and shares are in percent.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
R = json.loads((REPO / "results" / "padding_and_ties.json").read_text())
META = json.loads((REPO / "analysis" / "realdata_studies_metadata.json").read_text())


def means(path):
    acc = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc[(r["dataset"], r["representation"])].append(100 * float(r["macro_f1"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def eff(s, d, a, b):
    return s[(d, a)] - s[(d, b)]


def rho(x, y):
    return float(spearmanr(x, y)[0])


dev, ind = means(REPO / "results" / "development_scores.csv"), means(REPO / "results" / "independent_scores.csv")
names = sorted(R["independent"])
# padded: the equal-length files of the sources ("_eq_" in the file name) and the two Aconity datasets
padded = [n for n in names if "_eq_" in META["independent"][n]["train_path"] or n.startswith("Aconity")]
ps = {n: R["independent"][n]["padding"] for n in names}
lines = []

lines.append("## Statements of the paper\n")
above = [n for n in padded if ps[n]["median_share"] > 0.5]
lines.append("- Padded datasets: %d of %d. Median estimated padded share above one half: %d of them; AsphaltRegularityUni %.0f%%." % (
    len(padded), len(names), len(above), 100 * ps["AsphaltRegularityUni"]["median_share"]))
lines.append("- Class means of the estimated padded share over the training series: " + "; ".join(
    "%s %.0f%% to %.0f%%" % (n, 100 * ps[n]["class_mean_share_min"], 100 * ps[n]["class_mean_share_max"]) for n in names if n.startswith("Asphalt")) + ".")
sign = {s: {n: R[s][n]["sign"] for n in R[s]} for s in R}
checked = sum(len([k for k in sign[s][n] if k != "series_with_both_signs"]) for s in sign for n in sign[s])
bad = [(s, n, k) for s in sign for n in sign[s] for k, v in sign[s][n].items() if k != "series_with_both_signs" and v is False]
lines.append("- Ranges of the lag difference (whole training set and three fitting subsets of each of the 47 datasets): %d checked, %d do not contain zero." % (checked, len(bad)))

lines.append("\n## Padding in the 17 datasets of the independent evaluation\n")
lines.append("| Dataset | Padded | T | Series with tail | Median share | Class means (min to max) | Tail level minus rest (SD) | Windows inside tail, short / long | Length only: macro-F1 | 100/K | MQI (CNN) |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for n in names:
    r = R["independent"][n]; p = r["padding"]; lo = r["length_only"]
    lines.append("| %s | %s | %d | %.0f%% | %.0f%% | %.0f%% to %.0f%% | %s | %.0f%% / %.0f%% | %s | %.1f | %.1f |" % (
        n, "yes" if n in padded else "no", r["T"], 100 * p["series_with_tail"], 100 * p["median_share"], 100 * p["class_mean_share_min"], 100 * p["class_mean_share_max"],
        ("%.2f" % p["median_tail_minus_rest_in_sd"]) if p["median_tail_minus_rest_in_sd"] is not None else "",
        100 * p["windows_inside_tail"]["0.1"], 100 * p["windows_inside_tail"]["0.3"],
        ("%.1f (k=%d)" % (100 * lo["test_macro_f1"], lo["k"])) if n in padded else "", 100 * lo["one_over_K"], ind[(n, "MQI")]))

lines.append("\n## Ties among the clipped arrival levels (all series, constants from the whole training set)\n")
lines.append("| Study | Dataset | T | Short: L, mean ratio, any tie, ratio < 1/2 | Long: L, mean ratio, any tie, ratio < 1/2 | Test levels clipped |")
lines.append("|---|---|---:|---|---|---:|")
for study in ("development", "independent"):
    for n in (META["development"]["order"] if study == "development" else names):
        t = R[study][n]["ties"]
        f = lambda s: "%d, %.2f, %.0f%%, %.0f%%" % (t[s]["window_length"], t[s]["mean_distinct_ratio"], 100 * t[s]["share_any_tie"], 100 * t[s]["share_ratio_below_half"])
        lines.append("| %s | %s | %d | %s | %s | %.2f%% |" % (study, n, R[study][n]["T"], f("0.1"), f("0.3"), 100 * t["test_levels_clipped"]))

lines.append("\n## Spearman correlations (descriptive)\n")
for label, sel in (("17 datasets", names), ("13 padded datasets", padded)):
    share = [ps[n]["median_share"] for n in sel]
    c = {k: rho(share, [eff(ind, n, a, b) for n in sel]) for k, a, b in (("MQI-LCF - MQI", "MQI-LCF", "MQI"), ("MQI-UDI - MQI", "MQI-UDI", "MQI"), ("MQI-LCF - MQI-UDI", "MQI-LCF", "MQI-UDI"))}
    lines.append("- Median padded share against the effects, %s: " % label + "; ".join("%s %+.2f" % kv for kv in c.items()))
for label, s, sel, sc in (("30 development datasets", "development", META["development"]["order"], dev), ("17 datasets of the independent evaluation", "independent", names, ind)):
    heavy = [R[s][n]["ties"]["0.3"]["share_ratio_below_half"] for n in sel]
    c = {"heavy ties (long scale) vs MQI-LCF - MQI": rho(heavy, [eff(sc, n, "MQI-LCF", "MQI") for n in sel]), "heavy ties (long scale) vs MQI-LCF - MQI-UDI": rho(heavy, [eff(sc, n, "MQI-LCF", "MQI-UDI") for n in sel])}
    lines.append("- Share of long windows with a ratio below 1/2 against the effects, %s: " % label + "; ".join("%s %+.2f" % kv for kv in c.items()))
print("\n".join(lines))
