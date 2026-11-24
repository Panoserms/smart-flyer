#!/usr/bin/env python3
"""Write the website data file (SCHEMA.md, "fixture": false) from the finished study.

  python experiments/export_site_data.py experiments/results/site_data.json

Needs everything finished (triage on t1-t3, robustness, copy runs, ratings, agreement.json). The data
is validated before anything is written: keys, enums, counts, and the condition metrics recomputed
from the per-pair records must equal the summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import common  # noqa: E402
import run_copy  # noqa: E402
import study_stats as st  # noqa: E402

CHECK_IDS = ("names", "number", "sentence", "length", "language", "address", "claim", "oov", "parse")
SOURCES = ("llm", "template", "none")
DATASET_INFO = {
    "t1": ("Online Retail II, 2009–10", "real baskets, same shop as tuning, new pairs", "en"),
    "t2": ("Groceries (arules)", "real baskets, other domain", "en"),
    "t3": ("Constructed German grocery pairs", "constructed German pairs, not real baskets", "de"),
}
METRIC_KEYS = ("precision", "recall", "f1", "f1_ci", "accuracy", "llm_calls", "llm_tokens", "llm_seconds",
               "kept", "tp", "fp", "fn", "tn")
CL_KEYS = ("stop_term", "same_head", "compound", "head_a", "head_b", "pos_a", "pos_b", "stripped_a", "stripped_b")
HYP_KEYS = ("id", "text", "estimate", "ci", "p_raw", "p_holm", "supported", "note")
SUMMARY_KEYS = ("publishable_share", "publishable_ci", "llm_share", "checks_pass_share", "calls", "seconds")
CELL_KEYS = ("title", "copy", "source", "tries", "failed_checks", "first_try_failed_checks", "oov_words", "ratings")


class ExportError(Exception):
    pass


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ExportError(message)


def datasets_block(summary: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads((common.testsets_dir() / "manifest.json").read_text(encoding="utf-8"))["sets"]
    agreement_path = common.gold_dir() / "agreement.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else {}
    out = []
    for set_id in common.TEST_SETS:
        name, kind, lang = DATASET_INFO[set_id]
        _check(set_id in info, f"set {set_id} is not analysed")
        _check(set_id in agreement, f"{agreement_path} has no agreement statistics for {set_id}")
        out.append({"id": set_id, "name": name, "kind": kind, "lang": lang,
                    "n_transactions": manifest.get(set_id, {}).get("baskets_after_cleaning"),
                    "n_pairs": info[set_id]["n_pairs"], "kappa_3class": agreement[set_id]["kappa_3class"],
                    "kappa_binary": agreement[set_id]["kappa_binary"], "gold_counts": info[set_id]["gold_counts"]})
    return out


def pairs_block(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({"id": r["pair_id"], "set": r["set"], "a": r["a"], "b": r["b"], "lift": r["lift"],
                    "gold": r["gold"], "cl": {k: r["cl"][k] for k in CL_KEYS},
                    "llm": {"plain": r["plain"]["verdict"], "informed": r["informed"]["verdict"]}})
    return out


def copy_block(summary: dict[str, Any], internal: dict[str, Any]) -> dict[str, Any]:
    cells = internal["copy_cells"]
    bundles = []
    for b in internal["bundles"]:
        conditions = {}
        for cond in run_copy.CONDITIONS:
            conditions[cond] = {}
            for lang in run_copy.LANGS:
                c = cells[(b["id"], cond, lang)]
                conditions[cond][lang] = {
                    "title": c["title"], "copy": c["copy"], "source": c["source"], "tries": c["tries"],
                    "failed_checks": c["failed_checks"], "first_try_failed_checks": c["first_try_failed_checks"],
                    "oov_words": c["oov_words"],
                    "ratings": {"publishable_both": c["publishable_both"], "grammatical_both": c["grammatical_both"],
                                "natural_mean": c["natural_mean"]}}
        bundles.append({"id": b["id"], "set": b["set"], "anchor": b["anchor"], "pull": b["pull"], "pct": b["pct"],
                        "conditions": conditions})
    sm = {g: {c: {k: summary["copy"]["summary"][g][c][k] for k in SUMMARY_KEYS} for c in run_copy.CONDITIONS}
          for g in analyze.GROUPS_COPY}
    return {"bundles": bundles, "summary": sm}


def build_export(summary: dict[str, Any], internal: dict[str, Any]) -> dict[str, Any]:
    _check(sorted(summary["meta"]["sets_analysed"]) == list(common.TEST_SETS),
           "triage is not finished on all of t1, t2, t3: " + "; ".join(summary["messages"]))
    _check(summary.get("robustness") is not None, "robustness runs are missing")
    _check(summary.get("copy") is not None and internal.get("copy_cells") is not None,
           "copy runs or ratings are missing: " + "; ".join(summary["messages"]))
    rob = summary["robustness"]
    return {
        "fixture": False,
        "meta": {"model": summary["meta"]["model"], "generated": date.today().isoformat(),
                 "frozen_sha256": summary["meta"]["pipeline_sha256"], "prompt_variant": summary["meta"]["prompt_variant"]},
        "datasets": datasets_block(summary, internal["set_info"]),
        "pairs": pairs_block(internal["rows"]),
        "triage": {g: {c: {k: v["conditions"][c][k] for k in METRIC_KEYS} for c in analyze.CONDITIONS}
                   for g, v in summary["triage"].items()},
        "hypotheses": [{k: h[k] for k in HYP_KEYS} for h in summary["hypotheses"]],
        "robustness": {"n": rob["n"], "swap_flip_rate": rob["swap_flip_rate"], "case_flip_rate": rob["case_flip_rate"]},
        "copy": copy_block(summary, internal),
    }


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def validate(data: dict[str, Any]) -> None:
    """Raise ExportError unless the data fits the TypeScript types and its own arithmetic."""
    _check(data.get("fixture") is False, "fixture must be false")
    _check(set(data) == {"fixture", "meta", "datasets", "pairs", "triage", "hypotheses", "robustness", "copy"},
           f"top-level keys are {sorted(data)}")
    _check(set(data["meta"]) == {"model", "generated", "frozen_sha256", "prompt_variant"}, "meta keys")
    _check(all(isinstance(v, str) and v for v in data["meta"].values()), "meta values must be non-empty strings")
    sets = list(common.TEST_SETS)
    _check([d["id"] for d in data["datasets"]] == sets, "datasets must be t1, t2, t3 in order")
    for d in data["datasets"]:
        _check(set(d) == {"id", "name", "kind", "lang", "n_transactions", "n_pairs", "kappa_3class", "kappa_binary",
                          "gold_counts"}, f"dataset keys of {d['id']}")
        _check(set(d["gold_counts"]) == set(common.GOLD_LABELS) and sum(d["gold_counts"].values()) == d["n_pairs"],
               f"gold counts of {d['id']} do not sum to n_pairs")
        _check(d["n_transactions"] is None or _is_num(d["n_transactions"]), "n_transactions")
        _check(_is_num(d["kappa_3class"]) and _is_num(d["kappa_binary"]), f"kappa of {d['id']}")
    ids = [p["id"] for p in data["pairs"]]
    _check(len(ids) == len(set(ids)), "duplicate pair ids")
    for p in data["pairs"]:
        _check(set(p) == {"id", "set", "a", "b", "lift", "gold", "cl", "llm"}, f"pair keys of {p['id']}")
        _check(p["set"] in sets and p["gold"] in common.GOLD_LABELS, f"pair {p['id']}: set or gold invalid")
        _check(p["lift"] is None or _is_num(p["lift"]), f"pair {p['id']}: lift")
        _check(set(p["cl"]) == set(CL_KEYS), f"pair {p['id']}: cl keys")
        cl = p["cl"]
        _check(isinstance(cl["same_head"], bool) and (cl["stop_term"] is None or isinstance(cl["stop_term"], str))
               and (cl["compound"] is None or isinstance(cl["compound"], str)), f"pair {p['id']}: cl types")
        _check(all(isinstance(cl[k], str) for k in ("head_a", "head_b", "pos_a", "pos_b")), f"pair {p['id']}: cl strings")
        _check(all(isinstance(cl[k], list) and all(isinstance(w, str) for w in cl[k]) for k in ("stripped_a", "stripped_b")),
               f"pair {p['id']}: stripped lists")
        _check(set(p["llm"]) == {"plain", "informed"} and all(v in common.VERDICTS for v in p["llm"].values()),
               f"pair {p['id']}: llm verdicts")
    for d in data["datasets"]:
        _check(sum(p["set"] == d["id"] for p in data["pairs"]) == d["n_pairs"], f"pair count of {d['id']}")
        golds = [p["gold"] for p in data["pairs"] if p["set"] == d["id"]]
        _check({k: golds.count(k) for k in common.GOLD_LABELS} == d["gold_counts"], f"gold counts of {d['id']} vs pairs")
    _check(set(data["triage"]) == {*sets, "pooled"}, "triage groups")
    for group, conds in data["triage"].items():
        _check(set(conds) == set(analyze.CONDITIONS), f"triage {group}: conditions")
        members = data["pairs"] if group == "pooled" else [p for p in data["pairs"] if p["set"] == group]
        gold = np.array([p["gold"] == "complement" for p in members], bool)
        reject = np.array([bool(p["cl"]["stop_term"]) or p["cl"]["same_head"] or bool(p["cl"]["compound"])
                           for p in members], bool)
        plain = np.array([p["llm"]["plain"] == "COMPLEMENT" for p in members], bool)
        informed = np.array([p["llm"]["informed"] == "COMPLEMENT" for p in members], bool)
        preds = {"L": plain, "C": ~reject, "CL": ~reject & plain, "Lf": informed, "CLf": ~reject & informed}
        for cond, m in conds.items():
            _check(set(m) == set(METRIC_KEYS), f"triage {group}/{cond}: keys")
            _check(all(_is_num(m[k]) for k in METRIC_KEYS if k != "f1_ci"), f"triage {group}/{cond}: numbers")
            _check(isinstance(m["f1_ci"], list) and len(m["f1_ci"]) == 2 and m["f1_ci"][0] <= m["f1_ci"][1],
                   f"triage {group}/{cond}: f1_ci")
            _check(m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(members), f"triage {group}/{cond}: counts do not sum to n")
            _check(m["kept"] == m["tp"] + m["fp"], f"triage {group}/{cond}: kept")
            again = st.metrics(preds[cond], gold)
            for k, v in again.items():
                _check(abs(m[k] - v) < 1e-9, f"triage {group}/{cond}: {k} {m[k]} differs from the per-pair records {v}")
        _check(conds["C"]["llm_calls"] == 0, f"triage {group}: C must have 0 LLM calls")
    for cond in analyze.CONDITIONS:
        for k in ("tp", "fp", "fn", "tn", "kept", "llm_calls"):
            _check(sum(data["triage"][s][cond][k] for s in sets) == data["triage"]["pooled"][cond][k],
                   f"triage pooled/{cond}: {k} is not the sum of the sets")
    ids_h = [h["id"] for h in data["hypotheses"]]
    _check(ids_h == [f"H{i}" for i in range(1, 8)], f"hypotheses are {ids_h}")
    for h in data["hypotheses"]:
        _check(set(h) == set(HYP_KEYS) and isinstance(h["supported"], bool), f"{h['id']}: keys")
        _check(h["estimate"] is None or _is_num(h["estimate"]), f"{h['id']}: estimate")
        _check(h["ci"] is None or (len(h["ci"]) == 2 and h["ci"][0] <= h["ci"][1]), f"{h['id']}: ci")
        _check(not h["note"].startswith("NOT EVALUATED"), f"{h['id']} was not evaluated")
    rob = data["robustness"]
    _check(set(rob) == {"n", "swap_flip_rate", "case_flip_rate"} and _is_num(rob["n"]), "robustness keys")
    _check(set(rob["swap_flip_rate"]) == {"L", "Lf", "C"} and set(rob["case_flip_rate"]) == {"L", "C"}, "flip rate keys")
    _check(all(0 <= v <= 1 for v in (*rob["swap_flip_rate"].values(), *rob["case_flip_rate"].values())), "flip rates in 0-1")
    copy = data["copy"]
    _check(set(copy) == {"bundles", "summary"}, "copy keys")
    langs, conds = list(run_copy.LANGS), list(run_copy.CONDITIONS)
    for b in copy["bundles"]:
        _check(set(b) == {"id", "set", "anchor", "pull", "pct", "conditions"} and b["set"] in sets, f"bundle {b.get('id')}: keys")
        _check(set(b["conditions"]) == set(conds), f"bundle {b['id']}: conditions")
        for cond in conds:
            _check(set(b["conditions"][cond]) == set(langs), f"bundle {b['id']}/{cond}: languages")
            for lang, cell in b["conditions"][cond].items():
                where = f"bundle {b['id']}/{cond}/{lang}"
                _check(set(cell) == set(CELL_KEYS), f"{where}: keys")
                _check(cell["source"] in SOURCES and isinstance(cell["tries"], int), f"{where}: source or tries")
                _check(all(c in CHECK_IDS for c in cell["failed_checks"] + cell["first_try_failed_checks"]), f"{where}: check ids")
                _check(isinstance(cell["title"], str) and isinstance(cell["copy"], str), f"{where}: text")
                _check(set(cell["ratings"]) == {"publishable_both", "grammatical_both", "natural_mean"}, f"{where}: ratings")
    _check(set(copy["summary"]) == {"pooled", *langs}, "copy summary groups")
    for g, per in copy["summary"].items():
        _check(set(per) == set(conds), f"copy summary {g}: conditions")
        for cond, cell in per.items():
            _check(set(cell) == set(SUMMARY_KEYS), f"copy summary {g}/{cond}: keys")
            share = [b["conditions"][cond][lg]["ratings"]["publishable_both"] for b in copy["bundles"]
                     for lg in (langs if g == "pooled" else [g])]
            _check(abs(cell["publishable_share"] - sum(share) / len(share)) < 1e-9,
                   f"copy summary {g}/{cond}: publishable_share differs from the cells")
            llm = [b["conditions"][cond][lg]["source"] == "llm" for b in copy["bundles"]
                   for lg in (langs if g == "pooled" else [g])]
            _check(abs(cell["llm_share"] - sum(llm) / len(llm)) < 1e-9, f"copy summary {g}/{cond}: llm_share")
            _check(cell["publishable_ci"][0] <= cell["publishable_ci"][1], f"copy summary {g}/{cond}: ci")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", help="path of the JSON file to write")
    parser.add_argument("--variant")
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy)")
    parser.add_argument("--root", help="project tree to read (default: this repository)")
    parser.add_argument("--resamples", type=int, default=st.DEFAULT_RESAMPLES)
    args = parser.parse_args()
    if args.root:
        common.ROOT = Path(args.root)
    summary, internal = analyze.build(variant=args.variant, pipeline_path=args.pipeline, resamples=args.resamples)
    try:
        data = build_export(summary, internal)
        validate(data)
    except ExportError as exc:
        print(f"ERROR: not written. {exc}", file=sys.stderr, flush=True)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {len(data['pairs'])} pairs, {len(data['copy']['bundles'])} bundles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
