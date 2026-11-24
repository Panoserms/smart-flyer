#!/usr/bin/env python3
"""Pick the triage prompt: all three variants on a seeded dev sample, best macro-F1 wins.

Runs plain (no parser information) on 50 dev + 30 dev_de pairs, via run_triage, so it is
resumable. Scores the 3 classes against eval/gold/final_dev*.csv. A tie goes to the variant
that used fewer tokens. Writes experiments/results/prompt_selection.json.
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
import run_triage  # noqa: E402
import study_stats  # noqa: E402
import triage_prompts as prompts  # noqa: E402

SAMPLE_SEED = 20261106
SAMPLE_SIZES = {"dev": 50, "dev_de": 30}


def draw_sample(seed: int = SAMPLE_SEED, sizes: dict[str, int] = SAMPLE_SIZES) -> dict[str, list[str]]:
    """The same pair ids for every variant, drawn per set from the sorted ids."""
    out = {}
    for set_id, size in sizes.items():
        ids = sorted(p["pair_id"] for p in common.read_pairs(set_id))
        out[set_id] = random.Random(f"{seed}:{set_id}").sample(ids, min(size, len(ids)))
    return out


def macro_f1(gold: list[str], verdicts: list[str]) -> dict[str, Any]:
    """Unweighted mean F1 over the 3 classes. INVALID belongs to no class, so it only costs recall."""
    per_class = {}
    for label in common.GOLD_LABELS:
        verdict = common.LABEL_TO_VERDICT[label]
        tp = sum(g == label and v == verdict for g, v in zip(gold, verdicts))
        fp = sum(g != label and v == verdict for g, v in zip(gold, verdicts))
        fn = sum(g == label and v != verdict for g, v in zip(gold, verdicts))
        per_class[label] = float(study_stats.prf(tp, fp, fn)[2])
    return {"macro_f1": sum(per_class.values()) / len(per_class), "per_class_f1": per_class}


def score_variant(records_by_set: dict[str, list[dict[str, Any]]],
                  gold_by_set: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Quality and cost of one variant over the sample, pooled and per set."""
    records = [r for rs in records_by_set.values() for r in rs]
    gold = [gold_by_set[r["set"]][r["pair_id"]] for r in records]
    verdicts = [r["verdict"] for r in records]
    prompt_tokens = sum(r.get("prompt_eval_count") or 0 for r in records)
    output_tokens = sum(r.get("eval_count") or 0 for r in records)
    return {
        **macro_f1(gold, verdicts),
        "accuracy": sum(common.LABEL_TO_VERDICT[g] == v for g, v in zip(gold, verdicts)) / len(records),
        "n": len(records),
        "invalid": verdicts.count("INVALID"),
        "errors": sum(bool(r.get("error")) for r in records),
        "calls": len(records),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokens": prompt_tokens + output_tokens,
        "seconds": round(sum(r.get("wall_ms") or 0 for r in records) / 1000, 1),
        "by_set": {s: macro_f1([gold_by_set[s][r["pair_id"]] for r in rs], [r["verdict"] for r in rs])["macro_f1"]
                   for s, rs in records_by_set.items()},
    }


def pick_winner(results: dict[str, dict[str, Any]]) -> tuple[str, bool]:
    """Best macro-F1; on a tie the cheaper variant. Returns (winner, whether there was a tie)."""
    best = max(round(r["macro_f1"], 6) for r in results.values())
    tied = [name for name, r in results.items() if round(r["macro_f1"], 6) == best]
    return min(tied, key=lambda name: results[name]["tokens"]), len(tied) > 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy, else working copy)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--dry-run", action="store_true", help="draw the sample and print the plan only")
    args = parser.parse_args()

    sample = draw_sample(args.seed)
    for set_id, ids in sample.items():
        path = common.results_dir() / f"prompt_sample_ids_{set_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    total = sum(len(ids) for ids in sample.values())
    print(f"Sample: {({s: len(ids) for s, ids in sample.items()})}; {total * len(prompts.VARIANTS)} calls in all",
          flush=True)
    if args.dry_run:
        return 0

    gold = {set_id: common.read_gold(set_id) for set_id in sample}
    results = {}
    for variant in prompts.VARIANTS:
        records = {set_id: run_triage.run_set(set_id, "plain", variant, pipeline_path=args.pipeline, ids=ids)["records"]
                   for set_id, ids in sample.items()}
        results[variant] = score_variant(records, gold)
    winner, tie = pick_winner(results)
    out = {"seed": args.seed, "sample": sample, "variants": results, "winner": winner, "tie": tie,
           "rule": "highest macro-F1 over the 3 classes; on a tie, fewest tokens"}
    common.prompt_selection_path().write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for name, r in results.items():
        print(f"{name:12s} macro-F1 {r['macro_f1']:.3f}  invalid {r['invalid']}  calls {r['calls']}  "
              f"tokens {r['tokens']}  {r['seconds']} s", flush=True)
    print(f"Winner: {winner}{' (tie, cheaper)' if tie else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
