import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import Classifier

R = Path(__file__).parent / "results"


def load(name: str) -> dict | list:
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))


runs = load("experiments")
results = load("final_results")
vocab = load("vocabulary")
assert len(runs) == 9
assert all(len(r["history"]) == 4 for r in runs)
splits = {n: pd.read_csv(R / f"{n}_split.csv") for n in ["train", "validation", "test"]}
sets = {n: set(d.text) for n, d in splits.items()}
assert not (
    sets["train"] & sets["validation"]
    or sets["train"] & sets["test"]
    or sets["validation"] & sets["test"]
)
for result in results:
    kind = result["model"]
    pred = pd.read_csv(R / f"{kind}_predictions.csv")
    assert pred.actual.tolist() == splits["test"].label.tolist()
    assert abs(accuracy_score(pred.actual, pred.predicted) - result["accuracy"]) < 1e-10
    assert (
        abs(f1_score(pred.actual, pred.predicted, average="macro") - result["macro_f1"])
        < 1e-10
    )
    assert (
        sum(p.numel() for p in Classifier(len(vocab), kind).parameters())
        == result["parameters"]
    )
    chosen = max(
        [r for r in runs if r["model"] == kind],
        key=lambda r: r["best_validation_macro_f1"],
    )
    assert (
        chosen["learning_rate"] == result["learning_rate"]
        and chosen["batch_size"] == result["batch_size"]
    )
checks = dict(
    experiments=9,
    epochs_per_run=4,
    disjoint_normalized_text=True,
    prediction_metrics_recomputed=True,
    parameter_counts_verified=True,
    selection_matches_validation=True,
)
(R / "verification.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
