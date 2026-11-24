#!/usr/bin/env python3
"""Blind rating sheets from the final copy texts: one shuffled CSV per rater, plus a private key.

Writes eval/gold/blind_rating/R1.csv and R2.csv (condition and source hidden) and
experiments/results/rating_key.csv. Identical texts still get one row per condition.
Empty outputs are not sent to raters; they count as not publishable.

The raters return eval/gold/blind_rating/R1_scored.csv and R2_scored.csv with the columns
rating_id,faithful,grammatical,natural,publishable (natural 1-5, the others yes/no).
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
import run_copy  # noqa: E402

SHEET_HEADER = ["rating_id", "language", "product_1", "product_2", "discount_pct", "title", "copy"]
KEY_HEADER = ["rating_id", "bundle_id", "set", "condition", "language", "source"]
ID_SEED, R1_SEED, R2_SEED = 20261106, 20261107, 20261108


def sheet_dir() -> Path:
    return common.gold_dir() / "blind_rating"


def key_path() -> Path:
    return common.results_dir() / "rating_key.csv"


def latest_finals(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(bundle, condition, language) -> the last final record."""
    return {(r["bundle_id"], r["condition"], r["lang"]): r for r in common.read_jsonl(path) if r["kind"] == "final"}


def build_rows(finals: dict[tuple[str, str, str], dict[str, Any]],
               bundles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sheet rows and key rows, with ids that do not follow any order of conditions."""
    by_id = {b["id"]: b for b in bundles}
    keys = sorted(k for k, r in finals.items() if r["source"] != "none" and (r["title"].strip() or r["copy"].strip()))
    random.Random(ID_SEED).shuffle(keys)
    sheet, key = [], []
    for number, (bundle_id, condition, lang) in enumerate(keys, start=1):
        record, bundle = finals[(bundle_id, condition, lang)], by_id[bundle_id]
        rating_id = f"r{number:04d}"
        sheet.append({"rating_id": rating_id, "language": lang, "product_1": bundle["anchor"],
                      "product_2": bundle["pull"], "discount_pct": bundle["pct"], "title": record["title"],
                      "copy": record["copy"]})
        key.append({"rating_id": rating_id, "bundle_id": bundle_id, "set": record["set"], "condition": condition,
                    "language": lang, "source": record["source"]})
    return sheet, key


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = sorted(rows, key=lambda r: r["rating_id"])
    random.Random(seed).shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", default=None, help="run log (default: results/copy_runs.jsonl)")
    args = parser.parse_args()
    finals = latest_finals(Path(args.runs) if args.runs else run_copy.runs_path())
    bundles = json.loads(run_copy.bundles_path().read_text(encoding="utf-8"))
    expected = len(bundles) * len(run_copy.CONDITIONS) * len(run_copy.LANGS)
    if len(finals) != expected:
        sys.exit(f"Copy runs are incomplete: {len(finals)} final texts, expected {expected}.")
    sheet, key = build_rows(finals, bundles)
    common.write_csv(sheet_dir() / "R1.csv", SHEET_HEADER, shuffled(sheet, R1_SEED))
    common.write_csv(sheet_dir() / "R2.csv", SHEET_HEADER, shuffled(sheet, R2_SEED))
    common.write_csv(key_path(), KEY_HEADER, key)
    print(f"{len(sheet)} texts for rating, {expected - len(sheet)} empty and not sent", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
