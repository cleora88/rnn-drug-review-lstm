"""Check token handling and reproducible stratified sampling."""

import unittest

import pandas as pd
import torch

from preprocessing import clean, encode_texts, stratified_subset


class PreprocessingTests(unittest.TestCase):
    def test_html_and_negation(self):
        self.assertEqual(
            clean("It DIDN'T help &amp; made me sick!"), "it didn't help made me sick !"
        )

    def test_unknown_words_and_padding(self):
        vocabulary = {"<pad>": 0, "<unk>": 1, "helped": 2}
        tokens = encode_texts(["helped unfamiliar"], vocabulary, max_length=4)
        torch.testing.assert_close(tokens, torch.tensor([[2, 1, 0, 0]]))

    def test_truncation_preserves_prefix(self):
        vocabulary = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
        tokens = encode_texts(["a b a b"], vocabulary, max_length=2)
        torch.testing.assert_close(tokens, torch.tensor([[2, 3]]))

    def test_stratified_subset_is_reproducible(self):
        frame = pd.DataFrame({"label": [0] * 6 + [1] * 6 + [2] * 6})
        first = stratified_subset(frame, 9)
        second = stratified_subset(frame, 9)
        self.assertEqual(first.index.tolist(), second.index.tolist())
        self.assertEqual(first["label"].value_counts().to_dict(), {0: 3, 1: 3, 2: 3})


if __name__ == "__main__":
    unittest.main()
