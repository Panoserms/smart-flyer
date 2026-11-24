#!/usr/bin/env python3
"""Robustness runs on a seeded 100-pair subset of the pooled test sets.

LLM (resumable, same record format as run_triage): plain and informed with the product
order swapped, plain with letter case flipped. Rules (CPU only): the stage is recomputed
under both. Base verdicts come from the main triage runs; analyze.py computes flip rates.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl_features  # noqa: E402
import common  # noqa: E402
import run_triage  # noqa: E402
import triage_prompts as prompts  # noqa: E402

SUBSET_SEED = 20261106
SUBSET_SIZE = 100
KINDS = {"swap_plain": ("swap", "plain"), "swap_informed": ("swap", "informed"), "case_plain": ("case", "plain")}


def ids_path() -> Path:
    return common.results_dir() / "robustness_ids.json"


def rules_path() -> Path:
    return common.results_dir() / "robustness_rules.jsonl"


def llm_path(kind: str, variant: str) -> Path:
    return common.results_dir() / f"robustness_{kind}_{variant}.jsonl"


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split total proportionally over the sets, largest remainders first."""
    grand = sum(sizes.values())
    exact = {k: total * v / grand for k, v in sizes.items()}
    out = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(sizes, key=lambda k: (exact[k] - out[k], k), reverse=True)
    for key in by_remainder[: total - sum(out.values())]:
        out[key] += 1
    return out


def draw_subset(pairs_by_set: dict[str, list[dict[str, Any]]], total: int = SUBSET_SIZE,
                seed: int = SUBSET_SEED) -> dict[str, list[str]]:
    shares = allocate({s: len(pairs) for s, pairs in pairs_by_set.items()}, total)
    return {s: random.Random(f"{seed}:{s}").sample(sorted(p["pair_id"] for p in pairs), shares[s])
            for s, pairs in pairs_by_set.items()}


def flip_case(name: str, pipeline: ModuleType) -> str:
    """ALL-CAPS names become title case, any other name becomes upper case."""
    return pipeline.smart_title(name) if name.upper() == name else name.upper()


def swapped(pair: dict[str, Any]) -> dict[str, Any]:
    return {**pair, "a": pair["b"], "b": pair["a"], "lang_a": pair["lang_b"], "lang_b": pair["lang_a"],
            "price_a": pair["price_b"], "price_b": pair["price_a"]}


def case_flipped(pair: dict[str, Any], pipeline: ModuleType) -> dict[str, Any]:
    return {**pair, "a": flip_case(pair["a"], pipeline), "b": flip_case(pair["b"], pipeline)}


def transformed_views(pairs: list[dict[str, Any]], catalog: list[str] | None,
                      pipeline: ModuleType) -> dict[str, Any]:
    """The rule stage on the pair as is, swapped and case-flipped."""
    swap = [swapped(p) for p in pairs]
    case = [case_flipped(p, pipeline) for p in pairs]
    return {
        "swap_pairs": {p["pair_id"]: p for p in swap},
        "case_pairs": {p["pair_id"]: p for p in case},
        "base": cl_features.analyse_pairs(pairs, catalog, pipeline),
        "swap": cl_features.analyse_pairs(swap, catalog, pipeline),
        "case": cl_features.analyse_pairs(case, catalog, pipeline),
    }


def jobs_for(kind: str, pairs: list[dict[str, Any]], set_id: str, views: dict[str, Any],
             variant: str) -> list[dict[str, Any]]:
    transform, condition = KINDS[kind]
    jobs = []
    for pair in pairs:
        pid = pair["pair_id"]
        shown = views[f"{transform}_pairs"][pid]
        analysis = views[transform][pid] if condition == "informed" else None
        jobs.append({"pair_id": pid, "set": set_id, "condition": condition, "transform": transform,
                     "prompt": prompts.build_prompt(variant, shown["a"], shown["b"], analysis)})
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", default="auto", choices=(*prompts.VARIANTS, "auto"))
    parser.add_argument("--sets", nargs="+", default=list(common.TEST_SETS))
    parser.add_argument("--n", type=int, default=SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy, else working copy)")
    parser.add_argument("--rules-only", action="store_true", help="only the CPU part, no LLM calls")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    variant = common.winner_variant() if args.variant == "auto" else args.variant
    pipeline = cl_features.load_pipeline(common.resolve_pipeline_path(args.pipeline, args.sets))
    pairs_by_set = {s: common.read_pairs(s) for s in args.sets}
    subset = draw_subset(pairs_by_set, args.n, args.seed)
    ids_path().parent.mkdir(parents=True, exist_ok=True)
    ids_path().write_text(json.dumps({"seed": args.seed, "ids": subset}, indent=2) + "\n", encoding="utf-8")
    print(f"Subset: {({s: len(ids) for s, ids in subset.items()})}", flush=True)

    rules, jobs = [], {kind: [] for kind in KINDS}
    for set_id, ids in subset.items():
        wanted = set(ids)
        pairs = [p for p in pairs_by_set[set_id] if p["pair_id"] in wanted]
        views = transformed_views(pairs, common.read_catalog(set_id), pipeline)
        rules += [{"pair_id": p["pair_id"], "set": set_id, "base_reject": views["base"][p["pair_id"]]["cl_reject"],
                   "swap_reject": views["swap"][p["pair_id"]]["cl_reject"],
                   "case_reject": views["case"][p["pair_id"]]["cl_reject"]} for p in pairs]
        for kind in KINDS:
            jobs[kind] += jobs_for(kind, pairs, set_id, views, variant)
    rules_path().write_text("".join(json.dumps(r) + "\n" for r in rules), encoding="utf-8")
    print(f"Rule stage recomputed for {len(rules)} pairs", flush=True)
    if args.rules_only:
        return 0
    for kind, kind_jobs in jobs.items():
        run_triage.run_jobs(kind_jobs, llm_path(kind, variant), variant=variant, label=f"robustness {kind}",
                            retry_errors=args.retry_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
