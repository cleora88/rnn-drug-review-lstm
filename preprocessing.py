"""Text normalization, stratified sampling and token encoding."""

import html
import re
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import TensorDataset

CLASS_NAMES = ("negative", "neutral", "positive")
MAX_LENGTH = 96
VOCABULARY_SIZE = 12_000
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[!?]")


def clean(text: str) -> str:
    """Decode HTML and retain lowercase word, number and sentiment tokens."""
    return " ".join(TOKEN_PATTERN.findall(html.unescape(str(text)).lower()))


def stratified_subset(frame: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= size:
        return frame
    subset, _ = train_test_split(
        frame, train_size=size, stratify=frame["label"], random_state=seed
    )
    return subset


def build_vocabulary(texts: pd.Series) -> dict[str, int]:
    counts = Counter(" ".join(texts).split())
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update(
        (word, index + 2)
        for index, (word, _) in enumerate(counts.most_common(VOCABULARY_SIZE - 2))
    )
    return vocabulary


def encode_texts(
    texts: list[str], vocabulary: dict[str, int], max_length: int = MAX_LENGTH
) -> Tensor:
    tokens = torch.zeros((len(texts), max_length), dtype=torch.long)
    for row, text in enumerate(texts):
        indices = [vocabulary.get(word, 1) for word in text.split()[:max_length]]
        tokens[row, : len(indices)] = torch.tensor(indices, dtype=torch.long)
    return tokens


def encode_dataset(frame: pd.DataFrame, vocabulary: dict[str, int]) -> TensorDataset:
    tokens = encode_texts(frame["text"].tolist(), vocabulary)
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    return TensorDataset(tokens, labels)
