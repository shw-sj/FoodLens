"""Interface tests for the ViT/SCST hand-off.

These tests use randomly initialised, tiny models (no pretrained download) and
lock the contract that the SCST owner receives:

* ``sequence_log_prob`` is differentiable and equals a manual
  ``log_softmax`` sum.
* ``greedy_decode`` is deterministic and matches a manual argmax loop.
* ``beam_search_decode`` respects ``max_len`` and returns valid ids.

Run: ``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.task1_caption.beam_search import beam_search_decode, greedy_decode
from src.task1_caption.models import build_model, sequence_log_prob


def _tiny_vit(vocab_size: int = 30, max_len: int = 8):
    torch.manual_seed(0)
    config = {
        "model": {
            "name": "vit_transformer",
            "pretrained": False,
            "d_model": 32,
            "nhead": 2,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "pad_id": 0,
        }
    }
    model = build_model(config, vocab_size=vocab_size, max_len=max_len)
    model.eval()
    return model


class SequenceLogProbTest(unittest.TestCase):
    def test_matches_manual_log_softmax_and_has_grad(self):
        model = _tiny_vit()
        model.train()
        images = torch.randn(2, 3, 224, 224)
        tokens = torch.tensor(
            [
                [1, 5, 6, 2],      # sos 5 6 eos
                [1, 7, 2, 0],      # sos 7 eos pad
            ],
            dtype=torch.long,
        )
        log_prob = sequence_log_prob(model, images, tokens, pad_id=0)

        logits = model(images, tokens[:, :-1])
        manual = F.log_softmax(logits.float(), dim=-1)
        manual = manual.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = tokens[:, 1:].ne(0).float()
        manual = (manual * mask).sum(dim=-1)

        self.assertTrue(torch.allclose(log_prob, manual, atol=1e-4))
        loss = -log_prob.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))


class GreedyTest(unittest.TestCase):
    def test_deterministic_and_matches_manual_argmax(self):
        model = _tiny_vit(max_len=6)
        images = torch.randn(2, 3, 224, 224)
        first = greedy_decode(model, images, max_len=5)
        second = greedy_decode(model, images, max_len=5)
        self.assertTrue(torch.equal(first, second))

        # Manual greedy loop.
        with torch.no_grad():
            state = model.init_decode_state(images)
            prev = torch.full((2,), 1, dtype=torch.long)
            rows = []
            for _ in range(5):
                logits, state = model.decode_step(state, prev)
                prev = logits.argmax(dim=-1)
                rows.append(prev)
        manual = torch.stack(rows, dim=1)
        self.assertTrue(torch.equal(first, manual))


class BeamTest(unittest.TestCase):
    def test_beam_respects_max_len(self):
        model = _tiny_vit(max_len=6)
        image = torch.randn(1, 3, 224, 224)
        tokens, score = beam_search_decode(
            model, image, beam_size=3, max_len=4, length_penalty=1.0
        )
        self.assertLessEqual(len(tokens), 4)
        self.assertTrue(all(0 <= token < 30 for token in tokens))
        self.assertIsInstance(score, float)


class ResNetInterfaceTest(unittest.TestCase):
    def test_greedy_and_log_prob(self):
        torch.manual_seed(1)
        config = {
            "model": {
                "name": "resnet18_gru",
                "pretrained": False,
                "embed_dim": 16,
                "hidden_dim": 32,
                "dropout": 0.0,
                "pad_id": 0,
            }
        }
        model = build_model(config, vocab_size=20, max_len=6)
        model.eval()
        images = torch.randn(2, 3, 64, 64)
        tokens = torch.tensor([[1, 5, 2], [1, 6, 2]], dtype=torch.long)
        lp = sequence_log_prob(model, images, tokens)
        self.assertEqual(lp.shape, (2,))
        out = greedy_decode(model, images, max_len=4)
        self.assertEqual(out.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
