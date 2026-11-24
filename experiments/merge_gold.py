#!/usr/bin/env python3
"""Merge two annotators' labels: agreement statistics first, then the final gold files.

  python experiments/merge_gold.py agree --sets dev dev_de t1 t2 t3
      validates eval/gold/annotator_{a,b}/<set>.csv, writes eval/gold/agreement.json and
      eval/gold/disagreements_<set>.csv for the blind adjudicator
  python experiments/merge_gold.py finalize --sets dev dev_de t1 t2 t3
      after eval/gold/adjudicated_<set>.csv exists: writes eval/gold/final_<set>.csv and
      stops with an error on any unresolved row
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import study_stats  # noqa: E402

SEED = 20261106
DISAGREEMENT_HEADER = ["pair_id", "product_1", "product_2", "label_1", "confidence_1", "reason_1",
                       "label_2", "confidence_2", "reason_2"]


class GoldError(Exception):
    """A problem that must stop the merge."""


def read_annotator(annotator: str, set_id: str, expected_ids: list[str]) -> dict[str, dict[str, str]]:
    """One annotator's file, with every id present exactly once and every label valid."""
    path = common.gold_dir() / f"annotator_{annotator}" / f"{set_id}.csv"
    if not path.exists():
        raise GoldError(f"{path} is missing")
    rows = common.read_csv(path)
    ids = [r["pair_id"] for r in rows]
    problems = []
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate ids: {duplicates[:5]}")
    missing = sorted(set(expected_ids) - set(ids))
    if missing:
        problems.append(f"{len(missing)} ids missing, for example {missing[:5]}")
    extra = sorted(set(ids) - set(expected_ids))
    if extra:
        problems.append(f"{len(extra)} unknown ids, for example {extra[:5]}")
    bad = sorted({r["pair_id"] for r in rows if r["label"] not in common.GOLD_LABELS})
    if bad:
        problems.append(f"{len(bad)} invalid labels, for example {bad[:5]}")
    if problems:
        raise GoldError(f"{path}: " + "; ".join(problems))
    return {r["pair_id"]: r for r in rows}


def confusion(labels_a: list[str], labels_b: list[str]) -> dict[str, dict[str, int]]:
    table = {x: {y: 0 for y in common.GOLD_LABELS} for x in common.GOLD_LABELS}
    for x, y in zip(labels_a, labels_b):
        table[x][y] += 1
    return table


def agreement(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    binary_a = [x == "complement" for x in labels_a]
    binary_b = [x == "complement" for x in labels_b]
    agree = sum(x == y for x, y in zip(labels_a, labels_b))
    return {
        "n": len(labels_a),
        "kappa_3class": study_stats.cohen_kappa(labels_a, labels_b),
        "kappa_binary": study_stats.cohen_kappa(binary_a, binary_b),
        "percent_agreement": 100 * agree / len(labels_a),
        "n_disagreements": len(labels_a) - agree,
        "confusion": confusion(labels_a, labels_b),
    }


def disagreement_rows(set_id: str, pairs: list[dict[str, Any]], a: dict[str, dict[str, str]],
                      b: dict[str, dict[str, str]], seed: int = SEED) -> list[dict[str, str]]:
    """Rows for the adjudicator; a seeded coin per row hides which annotator is which."""
    rng = random.Random(f"{seed}:{set_id}")
    rows = []
    for pair in pairs:
        pid = pair["pair_id"]
        if a[pid]["label"] == b[pid]["label"]:
            continue
        first, second = (a[pid], b[pid]) if rng.random() < 0.5 else (b[pid], a[pid])
        rows.append({"pair_id": pid, "product_1": pair["a"], "product_2": pair["b"],
                     "label_1": first["label"], "confidence_1": first.get("confidence", ""),
                     "reason_1": first.get("reason", ""), "label_2": second["label"],
                     "confidence_2": second.get("confidence", ""), "reason_2": second.get("reason", "")})
    return rows


def load_set(set_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    pairs = common.read_pairs(set_id)
    ids = [p["pair_id"] for p in pairs]
    return pairs, read_annotator("a", set_id, ids), read_annotator("b", set_id, ids)


def run_agree(set_ids: list[str]) -> dict[str, Any]:
    path = common.gold_dir() / "agreement.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for set_id in set_ids:
        pairs, a, b = load_set(set_id)
        ids = [p["pair_id"] for p in pairs]
        report[set_id] = agreement([a[i]["label"] for i in ids], [b[i]["label"] for i in ids])
        rows = disagreement_rows(set_id, pairs, a, b)
        common.write_csv(common.gold_dir() / f"disagreements_{set_id}.csv", DISAGREEMENT_HEADER, rows)
        print(f"{set_id}: n={len(ids)} kappa_3={report[set_id]['kappa_3class']:.3f} "
              f"kappa_bin={report[set_id]['kappa_binary']:.3f} disagreements={len(rows)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def merge_final(set_id: str, pairs: list[dict[str, Any]], a: dict[str, dict[str, str]],
                b: dict[str, dict[str, str]], adjudicated: dict[str, str]) -> list[dict[str, str]]:
    """Final labels. An adjudicated row must exist for every disagreement and nothing else."""
    ids = [p["pair_id"] for p in pairs]
    open_ids = {i for i in ids if a[i]["label"] != b[i]["label"]}
    unresolved = sorted(open_ids - set(adjudicated))
    if unresolved:
        raise GoldError(f"{set_id}: {len(unresolved)} unresolved rows, for example {unresolved[:5]}")
    stray = sorted(set(adjudicated) - open_ids)
    if stray:
        raise GoldError(f"{set_id}: adjudicated rows that were no disagreement: {stray[:5]}")
    invalid = sorted(i for i, label in adjudicated.items() if label not in common.GOLD_LABELS)
    if invalid:
        raise GoldError(f"{set_id}: invalid adjudicated labels, for example {invalid[:5]}")
    return [{"pair_id": i, "label": adjudicated[i] if i in open_ids else a[i]["label"]} for i in ids]


def run_finalize(set_ids: list[str]) -> None:
    for set_id in set_ids:
        pairs, a, b = load_set(set_id)
        path = common.gold_dir() / f"adjudicated_{set_id}.csv"
        adjudicated = {r["pair_id"]: r["label"] for r in common.read_csv(path)} if path.exists() else {}
        rows = merge_final(set_id, pairs, a, b, adjudicated)
        common.write_csv(common.final_gold_path(set_id), ["pair_id", "label"], rows)
        print(f"{set_id}: wrote {len(rows)} final labels ({len(adjudicated)} adjudicated)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("agree", "finalize"))
    parser.add_argument("--sets", nargs="+", default=[*common.DEV_SETS, *common.TEST_SETS])
    args = parser.parse_args()
    try:
        run_agree(args.sets) if args.mode == "agree" else run_finalize(args.sets)
    except GoldError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
