"""Unified caption metrics: BLEU-4, ROUGE-L and CIDEr-D.

The three metrics live in one file so every experiment (task-1 baseline,
beam search, SCST hand-off) uses exactly the same tokenizer and scoring
conventions.  Changing anything here changes the frozen evaluation protocol,
so bump ``METRIC_VERSION`` / ``TOKENIZER_VERSION`` and update
``configs/eval_protocol.yaml`` when you do.

Conventions
-----------
* Public functions take a list of prediction strings and a list of reference
  groups.  A reference group is either a string (one reference) or a sequence
  of strings (multiple references for the same image).
* ``bleu4`` is corpus-level BLEU-4 with multi-reference clipping, closest
  reference length, brevity penalty, and the COCO epsilon smoothing
  (``+1e-15`` / ``+1e-9``).
* ``rouge_l`` follows the COCO ``pycocoevalcap`` ROUGE-L implementation:
  beta = 1.2, max precision and max recall are taken independently over the
  reference set, then combined into an F-measure.
* ``cider_d`` follows the COCO ``pycocoevalcap`` CIDEr implementation:
  tf-idf over 1-4 grams, image-level document frequency over the union of
  reference n-grams, ``N = number of images``, no count clipping by default,
  Gaussian length penalty (sigma = 6), and the
  ``sum(min(tf_hyp, tf_ref) * tf_ref) / (||tf_hyp|| * ||tf_ref||)``
  similarity.  The optional ``clip`` argument enables count clipping at N
  (CIDEr-D paper setting); the official COCO code does not clip, so the
  default is 0.  The native COCO scale is used (x10); set ``scale=100`` if
  your report uses a 0-100 table.
* ``bleu4`` and ``rouge_l`` are reported on a 0-100 scale by default, which
  is the usual convention in captioning papers.  CIDEr-D keeps the official
  x10 scale by default.  All scales are recorded by ``evaluate.py``.

The regex tokenizer is a deterministic stand-in for the Java PTB tokenizer
used by the official COCO tool.  It lower-cases, keeps words/numbers and
internal apostrophes/hyphens, and drops punctuation.  Use the same tokenizer
for all three metrics and for building the vocabulary.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Union

METRIC_VERSION = "task1-metrics-v1"
TOKENIZER_VERSION = "regex-lower-alnum-v1"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

ReferenceGroup = Union[str, Sequence[str]]

BLEU_EPSILON = 1e-15
BLEU_SMALL = 1e-9


def tokenize(text: str) -> List[str]:
    """Tokenize one caption with the frozen task-1 tokenizer."""
    if text is None:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _normalize_references(references: Iterable[ReferenceGroup]) -> List[List[str]]:
    groups: List[List[str]] = []
    for refs in references:
        if refs is None:
            groups.append([])
        elif isinstance(refs, str):
            groups.append([refs])
        else:
            groups.append([str(r) for r in refs])
    return groups


def _check_lengths(predictions: Sequence[str], references: Sequence[ReferenceGroup]) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length, "
            f"got {len(predictions)} vs {len(references)}"
        )


# ---------------------------------------------------------------------------
# BLEU-4 (corpus level, pycocoevalcap-compatible)
# ---------------------------------------------------------------------------
def bleu4(
    predictions: Sequence[str],
    references: Sequence[ReferenceGroup],
    max_order: int = 4,
    scale: float = 100.0,
) -> float:
    """Corpus-level BLEU-4 with COCO epsilon smoothing."""
    _check_lengths(predictions, references)
    ref_groups = _normalize_references(references)
    if not predictions:
        return 0.0

    correct = [0.0] * max_order
    guess = [0.0] * max_order
    test_len = 0
    ref_len = 0.0

    for pred, refs in zip(predictions, ref_groups):
        ptoks = tokenize(pred)
        rtoks = [tokenize(r) for r in refs]
        test_len += len(ptoks)

        if rtoks:
            # COCO "closest" reference length: smallest |len - test_len|,
            # ties broken by the shorter reference.
            ref_len += min((abs(len(r) - len(ptoks)), len(r)) for r in rtoks)[1]

        for n in range(1, max_order + 1):
            guess[n - 1] += max(0.0, float(len(ptoks) - n + 1))
            if len(ptoks) < n:
                continue
            cand_counts = Counter(_ngrams(ptoks, n))
            max_ref_counts = Counter()
            for rt in rtoks:
                for gram, count in Counter(_ngrams(rt, n)).items():
                    if count > max_ref_counts[gram]:
                        max_ref_counts[gram] = count
            correct[n - 1] += sum(
                min(count, max_ref_counts.get(gram, 0))
                for gram, count in cand_counts.items()
            )

    bleu = 1.0
    for k in range(max_order):
        bleu *= (correct[k] + BLEU_EPSILON) / (guess[k] + BLEU_SMALL)

    ratio = (test_len + BLEU_EPSILON) / (ref_len + BLEU_SMALL)
    if ratio < 1.0:
        bleu *= math.exp(1.0 - 1.0 / ratio)

    score = bleu ** (1.0 / max_order)
    return score * scale


# ---------------------------------------------------------------------------
# ROUGE-L (pycocoevalcap-compatible: beta=1.2, independent max P / max R)
# ---------------------------------------------------------------------------
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length; O(len(a)*len(b)) time, O(len(b))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(
    predictions: Sequence[str],
    references: Sequence[ReferenceGroup],
    beta: float = 1.2,
    scale: float = 100.0,
) -> float:
    """Mean sentence-level ROUGE-L F-measure (COCO implementation)."""
    _check_lengths(predictions, references)
    ref_groups = _normalize_references(references)
    if not predictions:
        return 0.0

    scores: List[float] = []
    for pred, refs in zip(predictions, ref_groups):
        ptoks = tokenize(pred)
        if not ptoks or not refs:
            scores.append(0.0)
            continue

        precisions: List[float] = []
        recalls: List[float] = []
        for ref in refs:
            rtoks = tokenize(ref)
            if not rtoks:
                continue
            lcs = _lcs_length(ptoks, rtoks)
            precisions.append(lcs / len(ptoks))
            recalls.append(lcs / len(rtoks))

        if not precisions:
            scores.append(0.0)
            continue
        precision = max(precisions)
        recall = max(recalls)
        if precision == 0.0 or recall == 0.0:
            scores.append(0.0)
            continue
        beta2 = beta * beta
        f = (1.0 + beta2) * precision * recall / (recall + beta2 * precision)
        scores.append(f)

    return (sum(scores) / len(scores)) * scale if scores else 0.0


# ---------------------------------------------------------------------------
# CIDEr-D (pycocoevalcap-compatible)
# ---------------------------------------------------------------------------
def cider_d(
    predictions: Sequence[str],
    references: Sequence[ReferenceGroup],
    max_order: int = 4,
    sigma: float = 6.0,
    clip: int = 0,
    scale: float = 10.0,
) -> float:
    """CIDEr score on the official COCO scale (x10 by default).

    ``clip > 0`` caps every n-gram count at ``clip`` (the CIDEr-D paper uses
    10).  The official COCO scorer does not clip, so the default is 0.
    ``sigma <= 0`` disables the Gaussian length penalty.
    """
    _check_lengths(predictions, references)
    ref_groups = _normalize_references(references)
    n_images = len(predictions)
    if n_images == 0:
        return 0.0

    cand_tokens = [tokenize(p) for p in predictions]
    ref_tokens = [[tokenize(r) for r in refs] for refs in ref_groups]

    # Image-level document frequency over the union of reference n-grams.
    df: Counter = Counter()
    for refs in ref_tokens:
        unique_grams = set()
        for ref in refs:
            for n in range(1, max_order + 1):
                unique_grams.update(_ngrams(ref, n))
        for gram in unique_grams:
            df[gram] += 1

    ref_len_log = math.log(float(max(1, n_images)))

    def counts_to_vec(tokens: Sequence[str]):
        counts: Counter = Counter()
        for n in range(1, max_order + 1):
            for gram in _ngrams(tokens, n):
                counts[gram] += 1
        if clip and clip > 0:
            counts = Counter({g: min(c, clip) for g, c in counts.items()})

        vecs = [defaultdict(float) for _ in range(max_order)]
        norms = [0.0] * max_order
        length = 0
        for gram, count in counts.items():
            n = len(gram)
            idf = ref_len_log - math.log(float(max(1.0, df.get(gram, 0))))
            value = float(count) * idf
            vecs[n - 1][gram] = value
            norms[n - 1] += value * value
            if n == 1:
                length += count
        norms = [math.sqrt(v) for v in norms]
        return vecs, norms, length

    def similarity(
        hyp_vecs, hyp_norms, hyp_len, ref_vecs, ref_norms, ref_len
    ) -> List[float]:
        delta = float(hyp_len - ref_len)
        gauss = (
            math.exp(-(delta * delta) / (2.0 * sigma * sigma))
            if sigma and sigma > 0
            else 1.0
        )
        vals: List[float] = []
        for n in range(max_order):
            numerator = 0.0
            for gram, hyp_value in hyp_vecs[n].items():
                ref_value = ref_vecs[n].get(gram, 0.0)
                numerator += min(hyp_value, ref_value) * ref_value
            if hyp_norms[n] > 0.0 and ref_norms[n] > 0.0:
                numerator /= hyp_norms[n] * ref_norms[n]
            else:
                numerator = 0.0
            vals.append(numerator * gauss)
        return vals

    image_scores: List[float] = []
    for cand, refs in zip(cand_tokens, ref_tokens):
        if not refs:
            image_scores.append(0.0)
            continue
        c_vecs, c_norms, c_len = counts_to_vec(cand)
        per_order = [0.0] * max_order
        for ref in refs:
            r_vecs, r_norms, r_len = counts_to_vec(ref)
            sims = similarity(c_vecs, c_norms, c_len, r_vecs, r_norms, r_len)
            for n in range(max_order):
                per_order[n] += sims[n]
        image_scores.append(sum(per_order) / (max_order * len(refs)))

    return (sum(image_scores) / n_images) * scale


# ---------------------------------------------------------------------------
# Auxiliary numbers reported next to the three main metrics
# ---------------------------------------------------------------------------
def repetition_rate(predictions: Sequence[str], n: int = 4) -> float:
    """Mean fraction of duplicate n-grams (0 = no repetition, 1 = all same).

    Sentences shorter than ``n`` tokens are skipped.  Returns 0 when no
    sentence is long enough.
    """
    scores: List[float] = []
    for pred in predictions:
        grams = _ngrams(tokenize(pred), n)
        if not grams:
            continue
        scores.append(1.0 - len(set(grams)) / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def average_length(predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    return sum(len(tokenize(p)) for p in predictions) / len(predictions)


def compute_all(
    predictions: Sequence[str],
    references: Sequence[ReferenceGroup],
    bleu_max_order: int = 4,
    rouge_beta: float = 1.2,
    cider_max_order: int = 4,
    cider_sigma: float = 6.0,
    cider_clip: int = 0,
    bleu_scale: float = 100.0,
    rouge_scale: float = 100.0,
    cider_scale: float = 10.0,
    rep_n: int = 4,
) -> Dict[str, float]:
    """Return every headline number used in the task-1 tables."""
    return {
        "n": float(len(predictions)),
        "bleu4": bleu4(
            predictions, references, max_order=bleu_max_order, scale=bleu_scale
        ),
        "rouge_l": rouge_l(
            predictions, references, beta=rouge_beta, scale=rouge_scale
        ),
        "cider_d": cider_d(
            predictions,
            references,
            max_order=cider_max_order,
            sigma=cider_sigma,
            clip=cider_clip,
            scale=cider_scale,
        ),
        "rep4": repetition_rate(predictions, n=rep_n),
        "avg_len": average_length(predictions),
    }
