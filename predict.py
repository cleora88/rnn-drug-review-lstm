"""Classify a new review using a saved checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from models import CELL_TYPES, Classifier
from preprocessing import CLASS_NAMES, clean, encode_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Review text to classify.")
    parser.add_argument("--model", choices=CELL_TYPES, default="cifg")
    args = parser.parse_args()
    normalized_text = clean(args.text)
    if not normalized_text:
        parser.error("The review must contain at least one supported token.")

    results_dir = Path(__file__).resolve().parent / "results"
    checkpoint = results_dir / f"{args.model}.pt"
    vocabulary_file = results_dir / "vocabulary.json"
    if not checkpoint.exists() or not vocabulary_file.exists():
        parser.error("Model files are missing. Run python run.py before prediction.")

    vocabulary = json.loads(vocabulary_file.read_text(encoding="utf-8"))
    model = Classifier(len(vocabulary), args.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = (
            model(encode_texts([normalized_text], vocabulary))
            .softmax(dim=-1)[0]
            .tolist()
        )
    print(json.dumps(dict(zip(CLASS_NAMES, probabilities)), indent=2))
    print("Class-weighted training scores are not calibrated confidence estimates.")


if __name__ == "__main__":
    main()
