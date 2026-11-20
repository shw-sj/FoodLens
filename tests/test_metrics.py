"""Metric regression tests.

The reference values in ``test_matches_pycocoevalcap`` were produced by
``pycocoevalcap`` (Bleu / Rouge / Cider) on the same tokenized strings, with
the tokenizer's output joined by spaces.  They lock the BLEU-4, ROUGE-L and
CIDEr-D implementations to the official COCO code.

Run: ``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import unittest

from src.common.metrics import (
    average_length,
    bleu4,
    cider_d,
    compute_all,
    repetition_rate,
    rouge_l,
    tokenize,
)

PREDICTIONS = [
    "a plate of fried chicken with rice and vegetables",
    "fresh salad with tomatoes lettuce and cheese",
    "grilled salmon with lemon",
    "a bowl of soup",
]
REFERENCES = [
    ["fried chicken with rice and vegetables", "a plate of fried chicken and rice"],
    ["salad with fresh tomatoes and lettuce", "a green salad with tomatoes and cheese", "fresh vegetable salad"],
    ["grilled salmon with lemon slices", "salmon grilled with lemon"],
    ["a bowl of vegetable soup", "soup in a bowl"],
]


class TokenizerTest(unittest.TestCase):
    def test_lowercase_punctuation_hyphen(self):
        self.assertEqual(
            tokenize("Stir-fried tofu, 2 pieces (hot!)"),
            ["stir-fried", "tofu", "2", "pieces", "hot"],
        )

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])


class MetricParityTest(unittest.TestCase):
    def test_matches_pycocoevalcap(self):
        # pycocoevalcap on the tokenized strings:
        # BLEU-4 72.4157734214586, ROUGE-L 82.17687074829931, CIDEr 4.5421366965215615
        self.assertAlmostEqual(bleu4(PREDICTIONS, REFERENCES), 72.4157734214586, places=6)
        self.assertAlmostEqual(rouge_l(PREDICTIONS, REFERENCES), 82.17687074829931, places=6)
        self.assertAlmostEqual(cider_d(PREDICTIONS, REFERENCES), 4.5421366965215615, places=6)

    def test_perfect_caption(self):
        self.assertAlmostEqual(bleu4(PREDICTIONS, [[p] for p in PREDICTIONS]), 100.0, places=6)
        self.assertAlmostEqual(rouge_l(PREDICTIONS, [[p] for p in PREDICTIONS]), 100.0, places=6)
        self.assertAlmostEqual(cider_d(PREDICTIONS, [[p] for p in PREDICTIONS]), 10.0, places=6)

    def test_disjoint_is_low(self):
        preds = ["zebra quantum rocket"]
        refs = [["apple banana cherry"]]
        self.assertLess(bleu4(preds, refs), 1.0)
        self.assertLess(rouge_l(preds, refs), 1.0)
        self.assertLess(cider_d(preds, refs), 1.0)

    def test_cider_scale(self):
        self.assertAlmostEqual(
            cider_d(PREDICTIONS, REFERENCES, scale=100.0),
            cider_d(PREDICTIONS, REFERENCES, scale=10.0) * 10.0,
            places=6,
        )

    def test_cider_clip_is_recorded(self):
        clipped = cider_d(PREDICTIONS, REFERENCES, clip=10)
        self.assertGreaterEqual(clipped, 0.0)
        self.assertLessEqual(clipped, 10.0)


class AuxiliaryTest(unittest.TestCase):
    def test_repetition(self):
        self.assertEqual(repetition_rate(["a b c d"], n=4), 0.0)
        self.assertGreater(repetition_rate(["a a a a a a"], n=4), 0.0)

    def test_average_length(self):
        self.assertAlmostEqual(average_length(["a b", "a b c"]), 2.5)

    def test_compute_all_keys(self):
        result = compute_all(PREDICTIONS, REFERENCES)
        for key in ("n", "bleu4", "rouge_l", "cider_d", "rep4", "avg_len"):
            self.assertIn(key, result)
        self.assertEqual(int(result["n"]), len(PREDICTIONS))

    def test_empty_input(self):
        self.assertEqual(bleu4([], []), 0.0)
        self.assertEqual(rouge_l([], []), 0.0)
        self.assertEqual(cider_d([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
