"""One replicate of one setting of the simulation study, for the three proposed representations.

    python examples/simulation_example.py --replicate 1 --setting A35

The script generates the training and test series of the replicate, trains the common CNN on the images of
each representation and prints macro-F1 and accuracy next to the values saved from the study
(results/simulation_scores.csv).  The series and their images are reproduced exactly; the fitted networks
depend on the PyTorch build, so the scores can differ from the saved ones, in particular near the chance level.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ts2i import simulation                      # noqa: E402
from ts2i.pipeline import evaluate               # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--replicate", type=int, default=1, help="1, ..., 20")
parser.add_argument("--setting", default="A35", choices=[s["id"] for s in simulation.SETTINGS])
args = parser.parse_args()

X_train, y_train = simulation.generate(args.replicate, args.setting, "train")
X_test, y_test = simulation.generate(args.replicate, args.setting, "test")
with open(ROOT / "results" / "simulation_scores.csv", newline="", encoding="utf-8") as f:
    saved = {r["representation"]: r for r in csv.DictReader(f) if int(r["replicate"]) == args.replicate and r["setting"] == args.setting}

for representation in ("MQI", "MQI-LCF", "MQI-UDI"):
    out = evaluate(representation, X_train, y_train, X_test, y_test, seed=61000 + args.replicate, sort_subsets=True)
    print(f"{representation:8s} macro-F1 {100 * out['macro_f1']:6.2f} (saved {100 * float(saved[representation]['macro_f1']):6.2f})   "
          f"accuracy {100 * out['accuracy']:6.2f} (saved {100 * float(saved[representation]['accuracy']):6.2f})   "
          f"learning rate {out['learning_rate']}, {out['epochs']} epochs")
