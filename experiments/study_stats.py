"""Statistics of the study: classification metrics, kappa, McNemar, Holm, bootstraps.

Everything is deterministic given a seed. Conditions compared within one bootstrap see the
same resamples, which is what makes the differences paired.
"""
from __future__ import annotations

import math
from typing import Any, Iterator, Sequence

import numpy as np

DEFAULT_RESAMPLES = 10_000
_CHUNK = 500


def counts(pred: np.ndarray, gold: np.ndarray) -> dict[str, int]:
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    return {"tp": int((pred & gold).sum()), "fp": int((pred & ~gold).sum()),
            "fn": int((~pred & gold).sum()), "tn": int((~pred & ~gold).sum())}


def prf(tp: Any, fp: Any, fn: Any) -> tuple[Any, Any, Any]:
    """Precision, recall, F1 for scalars or arrays. An empty denominator gives 0."""
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp + fp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp + fn), where=(tp + fn) > 0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return precision, recall, f1


def metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    c = counts(pred, gold)
    precision, recall, f1 = prf(c["tp"], c["fp"], c["fn"])
    n = sum(c.values())
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0, "kept": c["tp"] + c["fp"], **c}


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa. When chance agreement is 1 (one label only), agreement is perfect: 1.0."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two label lists of the same non-zero length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    chance = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    return 1.0 if chance == 1 else (observed - chance) / (1 - chance)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(ordered), 0.0, {}
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_p(differences: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for a difference of zero, never exactly 0."""
    b = len(differences)
    below, above = int((differences <= 0).sum()), int((differences >= 0).sum())
    return min(1.0, 2 * (min(below, above) + 1) / (b + 1))


def _chunks(total: int) -> Iterator[int]:
    for start in range(0, total, _CHUNK):
        yield min(_CHUNK, total - start)


def _collect(weight_chunks: Iterator[np.ndarray], preds: dict[str, np.ndarray],
             gold: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Precision, recall and F1 per condition and resample, from weights (resamples x pairs)."""
    gold = np.asarray(gold, bool)
    parts: dict[str, dict[str, list[np.ndarray]]] = {k: {"tp": [], "fp": [], "fn": []} for k in preds}
    for weights in weight_chunks:
        for name, pred in preds.items():
            pred = np.asarray(pred, bool)
            parts[name]["tp"].append((weights * (pred & gold)).sum(axis=1))
            parts[name]["fp"].append((weights * (pred & ~gold)).sum(axis=1))
            parts[name]["fn"].append((weights * (~pred & gold)).sum(axis=1))
    out = {}
    for name, p in parts.items():
        precision, recall, f1 = prf(*(np.concatenate(p[key]) for key in ("tp", "fp", "fn")))
        out[name] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def pair_bootstrap(preds: dict[str, np.ndarray], gold: np.ndarray, resamples: int = DEFAULT_RESAMPLES,
                   seed: int = 20261106) -> dict[str, dict[str, np.ndarray]]:
    """Resample pairs with replacement. Returns {condition: {metric: array(resamples)}}."""
    n = len(gold)
    rng = np.random.default_rng(seed)

    def weights() -> Iterator[np.ndarray]:
        for size in _chunks(resamples):
            yield rng.multinomial(n, np.full(n, 1 / n), size=size)

    return _collect(weights(), preds, gold)


def item_bootstrap(preds: dict[str, np.ndarray], gold: np.ndarray, strata: Sequence[str], names_a: Sequence[str],
                   names_b: Sequence[str], resamples: int = DEFAULT_RESAMPLES,
                   seed: int = 20261106) -> dict[str, dict[str, np.ndarray]]:
    """Resample product names instead of pairs, because pairs share products.

    Within each set the distinct product names are drawn with replacement. A pair survives
    when both of its products were drawn, with weight count(a) x count(b), so its expected
    weight stays close to 1. The same metrics are then computed on these weights.
    """
    rng = np.random.default_rng(seed)
    n = len(gold)
    groups = []
    for stratum in sorted(set(strata)):
        rows = np.array([i for i in range(n) if strata[i] == stratum])
        names = sorted({names_a[i] for i in rows} | {names_b[i] for i in rows})
        index = {name: k for k, name in enumerate(names)}
        groups.append((rows, np.array([index[names_a[i]] for i in rows]),
                       np.array([index[names_b[i]] for i in rows]), len(names)))

    def weights() -> Iterator[np.ndarray]:
        for size in _chunks(resamples):
            out = np.zeros((size, n), dtype=np.int64)
            for rows, idx_a, idx_b, m in groups:
                drawn = rng.multinomial(m, np.full(m, 1 / m), size=size)
                out[:, rows] = drawn[:, idx_a] * drawn[:, idx_b]
            yield out

    return _collect(weights(), preds, gold)


def mean_bootstrap(values: dict[str, np.ndarray], resamples: int = DEFAULT_RESAMPLES,
                   seed: int = 20261106) -> dict[str, np.ndarray]:
    """Resample units (here: bundles) for paired means. Each value array has one row per unit."""
    n = len(next(iter(values.values())))
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in values}
    for size in _chunks(resamples):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size)
        for name, array in values.items():
            parts[name].append((weights * np.asarray(array, float)).sum(axis=1) / n)
    return {k: np.concatenate(v) for k, v in parts.items()}
