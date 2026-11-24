#!/usr/bin/env python3
"""Run the triage LLM over one set: plain (raw names) or informed (parser analysis added).

Resumable: one JSON line per finished call in experiments/results/, and pair ids already
there are skipped. A failed call (after retries) is logged as INVALID with an "error"
field; --retry-errors runs those ids again and the new line replaces the old one.

  python experiments/run_triage.py --set t1 --condition plain --variant auto
  python experiments/run_triage.py --set dev --condition informed --variant examples --ids-file ids.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl_features  # noqa: E402
import common  # noqa: E402
import triage_prompts as prompts  # noqa: E402

CONDITIONS = ("plain", "informed")
MAX_CONSECUTIVE_ERRORS = 5
_WORD_RE = re.compile(r"\b(" + "|".join(prompts.ANSWER_WORDS) + r")\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*:", re.IGNORECASE)
_AFTER_ANSWER_RE = re.compile(r"[\s*_`\"'\[(]*(" + "|".join(prompts.ANSWER_WORDS) + r")\b", re.IGNORECASE)


def parse_verdict(text: str, variant: str) -> str:
    """First of the three words; for "reasoning", the word after the last "Answer:"."""
    if variant == "reasoning":
        markers = list(_ANSWER_RE.finditer(text))
        if not markers:
            return "INVALID"
        match = _AFTER_ANSWER_RE.match(text, markers[-1].end())
    else:
        match = _WORD_RE.search(text)
    return match.group(1).upper() if match else "INVALID"


def results_path(set_id: str, condition: str, variant: str, out_dir: str | Path | None = None) -> Path:
    base = Path(out_dir) if out_dir else common.results_dir()
    return base / f"triage_{set_id}_{condition}_{variant}.jsonl"


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def run_jobs(jobs: list[dict[str, Any]], out_path: Path, *, variant: str, label: str,
             retry_errors: bool = False) -> dict[str, Any]:
    """Run jobs ({pair_id, prompt, ...extra fields}) one at a time, appending to out_path."""
    done = common.latest_by_pair(out_path)
    for job in jobs:
        old = done.get(job["pair_id"])
        if old and old.get("prompt_sha") != common.sha1(job["prompt"]):
            sys.exit(f"{out_path}: {job['pair_id']} was run with a different prompt. "
                     "Mixed prompts would make the results meaningless; move the file away to restart.")
    todo = [j for j in jobs if j["pair_id"] not in done or (retry_errors and done[j["pair_id"]].get("error"))]
    num_predict = prompts.NUM_PREDICT[variant]
    print(f"[{label}] {len(jobs)} pairs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    started, consecutive_errors = time.perf_counter(), 0
    for n, job in enumerate(todo, start=1):
        reply = common.generate(job["prompt"], num_predict=num_predict)
        record = {k: v for k, v in job.items() if k != "prompt"}
        record.update(variant=variant, raw=reply["response"], verdict=parse_verdict(reply["response"], variant),
                      prompt_eval_count=reply["prompt_eval_count"], eval_count=reply["eval_count"],
                      total_duration=reply["total_duration"], load_duration=reply["load_duration"],
                      wall_ms=reply["wall_ms"], attempts=reply["attempts"],
                      prompt_sha=common.sha1(job["prompt"]))
        if reply["error"]:
            record["error"] = reply["error"]
        common.append_jsonl(out_path, record)
        done[job["pair_id"]] = record
        consecutive_errors = consecutive_errors + 1 if reply["error"] else 0
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            sys.exit(f"[{label}] {consecutive_errors} failed calls in a row: Ollama looks down. Stopping so an "
                     "outage is not logged as model answers. Fix it, then rerun with --retry-errors.")
        if n % 25 == 0 or n == len(todo):
            elapsed = time.perf_counter() - started
            eta = elapsed / n * (len(todo) - n)
            print(f"[{label}] {n}/{len(todo)} calls, {elapsed / n:.1f} s/call, "
                  f"elapsed {_format_seconds(elapsed)}, ETA {_format_seconds(eta)}", flush=True)
    records = [done[j["pair_id"]] for j in jobs]
    invalid = sum(r["verdict"] == "INVALID" for r in records)
    errors = sum(bool(r.get("error")) for r in records)
    print(f"[{label}] finished: {len(records)} records, {invalid} INVALID, {errors} call errors", flush=True)
    return {"records": records, "ran": len(todo), "invalid": invalid, "errors": errors}


def warn_if_example_leaks(pairs: list[dict[str, Any]], variant: str) -> None:
    """The invented examples must be in no dataset. This prints only the example's own name."""
    if variant != "examples":
        return
    invented = {name.casefold() for a, b, _ in prompts.EXAMPLES for name in (a, b)}
    found = sorted(invented & {p[key].casefold() for p in pairs for key in ("a", "b")})
    if found:
        print(f"WARNING: invented prompt example(s) also occur in this set: {found}", file=sys.stderr, flush=True)


def select_pairs(set_id: str, ids: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    pairs = common.read_pairs(set_id)
    if ids is not None:
        by_id = {p["pair_id"]: p for p in pairs}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            sys.exit(f"{len(unknown)} ids are not in {set_id}, for example {unknown[:3]}")
        pairs = [by_id[i] for i in ids]
    return pairs[:limit] if limit else pairs


def run_set(set_id: str, condition: str, variant: str, *, pipeline_path: str | Path | None = None,
            ids: list[str] | None = None, limit: int | None = None, out_dir: str | Path | None = None,
            retry_errors: bool = False) -> dict[str, Any]:
    """Run one set under one condition and variant. Importable for select_prompt.py."""
    path = common.resolve_pipeline_path(pipeline_path, [set_id])
    pairs = select_pairs(set_id, ids, limit)
    warn_if_example_leaks(pairs, variant)
    analysis: dict[str, dict[str, Any]] = {}
    if condition == "informed":
        analysis = cl_features.analyse_pairs(pairs, common.read_catalog(set_id), cl_features.load_pipeline(path))
    jobs = [{"pair_id": p["pair_id"], "set": set_id, "condition": condition,
             "prompt": prompts.build_prompt(variant, p["a"], p["b"], analysis.get(p["pair_id"]))}
            for p in pairs]
    return run_jobs(jobs, results_path(set_id, condition, variant, out_dir), variant=variant,
                    label=f"{set_id} {condition} {variant}", retry_errors=retry_errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, dest="set_id", help="dev, dev_de, t1, t2 or t3")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--variant", required=True, choices=(*prompts.VARIANTS, "auto"),
                        help="auto = the winner in prompt_selection.json")
    parser.add_argument("--ids-file", help="run only these pair ids (one per line)")
    parser.add_argument("--limit", type=int, help="run only the first N pairs")
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy, else working copy)")
    parser.add_argument("--out-dir", help="results directory (default: experiments/results)")
    parser.add_argument("--retry-errors", action="store_true", help="run ids again whose last call failed")
    args = parser.parse_args()
    variant = common.winner_variant() if args.variant == "auto" else args.variant
    ids = common.read_ids(args.ids_file) if args.ids_file else None
    run_set(args.set_id, args.condition, variant, pipeline_path=args.pipeline, ids=ids, limit=args.limit,
            out_dir=args.out_dir, retry_errors=args.retry_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
