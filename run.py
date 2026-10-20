"""Train and compare manual LSTM cells on rating-derived review sentiment."""

import argparse
import copy
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import CELL_TYPES, Classifier
from preprocessing import (
    CLASS_NAMES,
    MAX_LENGTH,
    build_vocabulary,
    clean,
    encode_dataset,
    stratified_subset,
)

ROOT = Path(__file__).resolve().parent
HYPERPARAMETERS = ((0.001, 128), (0.003, 128), (0.001, 64))


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "results"
    train_size: int = 12_000
    validation_size: int = 2_000
    test_size: int = 4_000
    epochs: int = 4
    seed: int = 42


def save_json(output_dir: Path, name: str, value: object) -> None:
    (output_dir / f"{name}.json").write_text(
        json.dumps(value, indent=2), encoding="utf-8"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(
    config: ExperimentConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict]:
    frames = {}
    for name, filename in (
        ("train", "drugsComTrain_raw.tsv"),
        ("test", "drugsComTest_raw.tsv"),
    ):
        source = config.data_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Dataset file not found: {source}. See setup.ps1.")
        frame = pd.read_csv(source, sep="\t")
        required = {"review", "rating"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{source.name} must contain {sorted(required)}.")
        if not frame["rating"].between(1, 10).all():
            raise ValueError(f"{source.name} contains missing or invalid ratings.")
        frames[name] = frame

    original_counts = {name: len(frame) for name, frame in frames.items()}
    for name, frame in frames.items():
        frame["text"] = frame["review"].map(clean)
        frame["label"] = np.select(
            [frame["rating"] <= 4, frame["rating"] <= 6], [0, 1], default=2
        )
        frames[name] = frame[frame["text"].str.len() > 0].drop_duplicates("text").copy()

    train, test = frames["train"], frames["test"]
    overlap = test["text"].isin(set(train["text"]))
    removed_overlap = int(overlap.sum())
    test = test[~overlap].copy()
    training, validation = train_test_split(
        train, test_size=0.15, stratify=train["label"], random_state=config.seed
    )
    splits = {
        "train": stratified_subset(training, config.train_size, config.seed),
        "validation": stratified_subset(
            validation, config.validation_size, config.seed
        ),
        "test": stratified_subset(test, config.test_size, config.seed),
    }
    vocabulary = build_vocabulary(splits["train"]["text"])
    summary = {
        "original": original_counts,
        "removed_test_overlap": removed_overlap,
        "cleaned_train": len(train),
        "cleaned_test": len(test),
        "splits": {
            name: {
                "size": len(frame),
                "classes": {
                    str(label): int(count)
                    for label, count in frame["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            }
            for name, frame in splits.items()
        },
        "vocabulary": len(vocabulary),
        "max_length": MAX_LENGTH,
        "labels": {"0": "negative (1-4)", "1": "neutral (5-6)", "2": "positive (7-10)"},
        "train_truncation_fraction": float(
            (splits["train"]["text"].str.split().str.len() > MAX_LENGTH).mean()
        ),
    }
    return splits, vocabulary, summary


def evaluate(
    model: Classifier, dataset: TensorDataset, device: str
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for tokens, labels in DataLoader(dataset, batch_size=256):
            predicted.extend(model(tokens.to(device)).argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
    }
    return metrics, actual, predicted


def train_run(
    kind: str,
    learning_rate: float,
    batch_size: int,
    vocabulary_size: int,
    datasets: dict[str, TensorDataset],
    criterion: nn.Module,
    config: ExperimentConfig,
    device: str,
) -> tuple[dict, dict]:
    set_seed(config.seed)
    model = Classifier(vocabulary_size, kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_state = None
    best_score = -1.0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, labels in DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True
        ):
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(tokens), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        metrics, _, _ = evaluate(model, datasets["validation"], device)
        epoch_result = {
            "epoch": epoch,
            "loss": total_loss / len(datasets["train"]),
            **metrics,
        }
        history.append(epoch_result)
        print(
            f"{kind} lr={learning_rate} batch={batch_size} {epoch_result}", flush=True
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
        save_json(
            config.output_dir,
            "progress",
            {
                "model": kind,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "history": history,
            },
        )

    record = {
        "model": kind,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "initialization": "PyTorch defaults",
        "seed": config.seed,
        "epochs": config.epochs,
        "embedding": 64,
        "hidden": 64,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_macro_f1": best_score,
        "seconds": time.perf_counter() - started,
        "history": history,
    }
    return record, best_state


def run_experiments(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_json(
        config.output_dir,
        "environment",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
    )
    print(f"Training device: {device}", flush=True)
    splits, vocabulary, summary = prepare_data(config)
    save_json(config.output_dir, "vocabulary", vocabulary)
    save_json(config.output_dir, "data_summary", summary)
    for name, frame in splits.items():
        frame.to_csv(config.output_dir / f"{name}_split.csv", index=False)

    datasets = {
        name: encode_dataset(frame, vocabulary) for name, frame in splits.items()
    }
    counts = np.bincount(splits["train"]["label"], minlength=3)
    if (counts == 0).any():
        raise ValueError("The training subset must contain all three classes.")
    weights = torch.tensor(
        len(splits["train"]) / (3 * counts), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    runs, selected = [], {}
    for kind in CELL_TYPES:
        best_score = -1.0
        for learning_rate, batch_size in HYPERPARAMETERS:
            record, state = train_run(
                kind,
                learning_rate,
                batch_size,
                len(vocabulary),
                datasets,
                criterion,
                config,
                device,
            )
            runs.append(record)
            save_json(config.output_dir, "experiments", runs)
            if record["best_validation_macro_f1"] > best_score:
                best_score = record["best_validation_macro_f1"]
                selected[kind] = (record, state)
        torch.save(selected[kind][1], config.output_dir / f"{kind}.pt")

    # Select every configuration before inspecting held-out test performance.
    final_results = []
    for kind, (record, state) in selected.items():
        model = Classifier(len(vocabulary), kind).to(device)
        model.load_state_dict(state)
        metrics, actual, predicted = evaluate(model, datasets["test"], device)
        final_results.append(
            {
                "model": kind,
                "learning_rate": record["learning_rate"],
                "batch_size": record["batch_size"],
                "parameters": record["parameters"],
                **metrics,
            }
        )
        save_json(
            config.output_dir,
            f"{kind}_classification",
            classification_report(
                actual,
                predicted,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        )
        pd.DataFrame(
            {
                "text": splits["test"]["text"].tolist(),
                "actual": actual,
                "predicted": predicted,
            }
        ).to_csv(config.output_dir / f"{kind}_predictions.csv", index=False)

    save_json(config.output_dir, "final_results", final_results)
    pd.DataFrame(final_results).to_csv(
        config.output_dir / "comparison.csv", index=False
    )
    majority_class = splits["train"]["label"].mode()[0]
    save_json(
        config.output_dir,
        "baseline",
        {
            "majority_accuracy": float(
                (splits["test"]["label"] == majority_class).mean()
            ),
            "majority_macro_f1": f1_score(
                splits["test"]["label"],
                [majority_class] * len(splits["test"]),
                average="macro",
            ),
        },
    )
    from style_figures import main as plot_results

    plot_results(config.output_dir)
    print(json.dumps(final_results, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if min(args.train_size, args.validation_size, args.test_size, args.epochs) < 1:
        parser.error("Subset sizes and epoch count must be positive.")
    run_experiments(ExperimentConfig(**vars(args)))


if __name__ == "__main__":
    main()
