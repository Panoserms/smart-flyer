#!/usr/bin/env python3
"""Analysis of the whole study: everything the pre-registration lists, from the JSONL results.

Task A (triage): L, C, CL, Lf, CLf per set and pooled, costs, paired bootstrap, H1-H5, Holm,
item-level sensitivity, confusion, flip rates, INVALID counts.
Task B (copy): H6, H7, check failures, retries, template fallback, cost, rater kappa.

Writes experiments/results/summary.json and REPORT.md. A partly finished study is analysed as
far as it goes and the missing parts are named; --triage-only skips Task B.

  python experiments/analyze.py                    # everything that is finished
  python experiments/analyze.py --triage-only      # no copy runs or ratings needed
  python experiments/analyze.py --root DIR --out-dir DIR2 --pipeline FILE   # scratch trees (tests)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl_features  # noqa: E402
import common  # noqa: E402
import make_rating_sheet  # noqa: E402
import run_copy  # noqa: E402
import run_robustness  # noqa: E402
import run_triage  # noqa: E402
import study_stats as st  # noqa: E402

CONDITIONS = ("L", "C", "CL", "Lf", "CLf")
COPY_CONDITIONS = run_copy.CONDITIONS
LANGS = run_copy.LANGS
GROUPS_COPY = ("pooled", *LANGS)
ALPHA = 0.05
NONINFERIORITY_MARGIN = -0.03
MIN_CALL_REDUCTION = 0.5
# name -> (condition x, condition y, metric): the difference x - y
DIFFS = {"CL-L:f1": ("CL", "L", "f1"), "CL-L:precision": ("CL", "L", "precision"),
         "CL-C:f1": ("CL", "C", "f1"), "Lf-L:f1": ("Lf", "L", "f1")}
HYPOTHESIS_TEXT = {
    "H1": "CL uses at least 50% fewer LLM calls than L and its F1 is non-inferior (lower CI bound of F1(CL) - F1(L) above -0.03).",
    "H2": "Precision of CL is higher than precision of L.",
    "H3": "F1 of CL is higher than F1 of C (the LLM adds value to the rules).",
    "H4": "F1 of Lf (LLM with the parser's analysis in the prompt) is higher than F1 of L.",
    "H5": "Swapping the product order flips the verdict of Lf less often than that of L.",
    "H6": "The publishable share of guarded copy is higher than that of raw copy.",
    "H7": "The publishable share of delex copy is higher than that of raw copy.",
}


class Incomplete(Exception):
    """Some part of the study is not finished; the message says what is missing."""


# --- Task A: loading ---------------------------------------------------------------

def load_triage(set_ids: tuple[str, ...], variant: str, pipeline: Any) -> tuple[list[dict[str, Any]], dict[str, str],
                                                                                dict[str, Any]]:
    """Per-pair rows of every finished set, the sets skipped (with reason) and set-level info."""
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    info: dict[str, Any] = {}
    for set_id in set_ids:
        try:
            pairs, gold = common.read_pairs(set_id), common.read_gold(set_id)
        except FileNotFoundError as exc:
            skipped[set_id] = f"missing file: {Path(str(exc).split(' does not')[0]).name}"
            continue
        plain = common.latest_by_pair(run_triage.results_path(set_id, "plain", variant))
        informed = common.latest_by_pair(run_triage.results_path(set_id, "informed", variant))
        ids = [p["pair_id"] for p in pairs]
        gaps = [("gold labels", [i for i in ids if i not in gold]), ("plain LLM runs", [i for i in ids if i not in plain]),
                ("informed LLM runs", [i for i in ids if i not in informed])]
        gaps = [(what, miss) for what, miss in gaps if miss]
        if gaps:
            skipped[set_id] = "; ".join(f"{len(miss)} of {len(ids)} pairs without {what}" for what, miss in gaps)
            continue
        analysis = cl_features.analyse_pairs(pairs, common.read_catalog(set_id), pipeline)
        for p in pairs:
            pid = p["pair_id"]
            cl = dict(analysis[pid])
            rows.append({"pair_id": pid, "set": set_id, "a": p["a"], "b": p["b"], "lift": p.get("lift"),
                         "gold": gold[pid], "reject": bool(cl.pop("cl_reject")), "cl": cl,
                         "plain": plain[pid], "informed": informed[pid]})
        info[set_id] = {"n_pairs": len(pairs), "gold_counts": {k: sum(g == k for g in (gold[i] for i in ids))
                                                               for k in common.GOLD_LABELS}}
    return rows, skipped, info


def predictions(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Positive = predicted complement. C keeps what the rules do not reject; an unparseable LLM answer is negative."""
    reject = np.array([r["reject"] for r in rows], bool)
    plain = np.array([r["plain"]["verdict"] == "COMPLEMENT" for r in rows], bool)
    informed = np.array([r["informed"]["verdict"] == "COMPLEMENT" for r in rows], bool)
    return {"L": plain, "C": ~reject, "CL": ~reject & plain, "Lf": informed, "CLf": ~reject & informed}


def _tokens(rec: dict[str, Any]) -> int:
    return int(rec.get("prompt_eval_count") or 0) + int(rec.get("eval_count") or 0)


def costs(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """LLM cost per condition; a cascade pays only for the pairs the rules let through."""
    def total(key: str, only_survivors: bool) -> dict[str, float]:
        used = [r[key] for r in rows if not (only_survivors and r["reject"])]
        return {"llm_calls": len(used), "llm_tokens": sum(_tokens(r) for r in used),
                "llm_seconds": round(sum((r.get("wall_ms") or 0) for r in used) / 1000, 3)}
    none = {"llm_calls": 0, "llm_tokens": 0, "llm_seconds": 0.0}
    return {"L": total("plain", False), "C": none, "CL": total("plain", True),
            "Lf": total("informed", False), "CLf": total("informed", True)}


def _diff_block(boot: dict[str, dict[str, np.ndarray]], point: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, (x, y, metric) in DIFFS.items():
        d = boot[x][metric] - boot[y][metric]
        out[name] = {"estimate": float(point[x][metric] - point[y][metric]), "ci": list(st.percentile_ci(d)),
                     "p": float(st.bootstrap_p(d))}
    return out


def confusion(rows: list[dict[str, Any]], preds: dict[str, np.ndarray]) -> dict[str, Any]:
    by_condition = {}
    for cond in CONDITIONS:
        by_condition[cond] = {g: {"n": 0, "predicted_complement": 0, "predicted_other": 0} for g in common.GOLD_LABELS}
        for r, positive in zip(rows, preds[cond]):
            cell = by_condition[cond][r["gold"]]
            cell["n"] += 1
            cell["predicted_complement" if positive else "predicted_other"] += 1
    verdicts = {}
    for key in ("plain", "informed"):
        verdicts[key] = {g: {v: 0 for v in common.VERDICTS} for g in common.GOLD_LABELS}
        for r in rows:
            verdicts[key][r["gold"]][r[key]["verdict"]] += 1
    return {"by_condition": by_condition, "llm_verdicts": verdicts}


def invalid_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in ("plain", "informed"):
        used = [r[key] for r in rows]
        survivors = [r[key] for r in rows if not r["reject"]]
        out[key] = {"n": len(used), "invalid": sum(u["verdict"] == "INVALID" for u in used),
                    "call_errors": sum(bool(u.get("error")) for u in used),
                    "invalid_among_cascade_survivors": sum(u["verdict"] == "INVALID" for u in survivors)}
    return out


def group_stats(rows: list[dict[str, Any]], resamples: int, with_items: bool = False) -> dict[str, Any]:
    gold = np.array([r["gold"] == "complement" for r in rows], bool)
    preds = predictions(rows)
    boot = st.pair_bootstrap(preds, gold, resamples)
    cost = costs(rows)
    conditions = {}
    for cond in CONDITIONS:
        m = st.metrics(preds[cond], gold)
        conditions[cond] = {**m, "f1_ci": list(st.percentile_ci(boot[cond]["f1"])), **cost[cond]}
    correct_l, correct_cl = preds["L"] == gold, preds["CL"] == gold
    b, c = int((correct_l & ~correct_cl).sum()), int((~correct_l & correct_cl).sum())
    out = {"n": len(rows), "conditions": conditions, "diffs": _diff_block(boot, conditions),
           "mcnemar_L_vs_CL": {"L_right_CL_wrong": b, "CL_right_L_wrong": c, "p": st.mcnemar_exact(b, c)},
           "confusion": confusion(rows, preds), "invalid": invalid_counts(rows)}
    if with_items:
        item = st.item_bootstrap(preds, gold, [r["set"] for r in rows], [r["a"] for r in rows],
                                 [r["b"] for r in rows], resamples)
        out["item_level"] = _diff_block(item, conditions)
        out["item_level"]["f1_ci"] = {c_: list(st.percentile_ci(item[c_]["f1"])) for c_ in CONDITIONS}
    return out


# --- Robustness ----------------------------------------------------------------------

def compute_robustness(rows: list[dict[str, Any]], variant: str, resamples: int) -> dict[str, Any]:
    ids_file, rules_file = run_robustness.ids_path(), run_robustness.rules_path()
    paths = {k: run_robustness.llm_path(k, variant) for k in run_robustness.KINDS}
    missing = [p.name for p in (ids_file, rules_file, *paths.values()) if not p.exists()]
    if missing:
        raise Incomplete("robustness runs missing: " + ", ".join(missing))
    base = {r["pair_id"]: r for r in rows}
    wanted = {i for ids in json.loads(ids_file.read_text(encoding="utf-8"))["ids"].values() for i in ids} & set(base)
    recs = {k: common.latest_by_pair(p) for k, p in paths.items()}
    rules = {r["pair_id"]: r for r in common.read_jsonl(rules_file)}
    gaps = {k: len(wanted - set(v)) for k, v in recs.items() if wanted - set(v)}
    gaps_rules = len(wanted - set(rules))
    if gaps or gaps_rules:
        raise Incomplete(f"robustness runs unfinished (pairs without a record: {gaps}, rules {gaps_rules})")
    ids = sorted(wanted)
    flips = {
        "L_swap": [recs["swap_plain"][i]["verdict"] != base[i]["plain"]["verdict"] for i in ids],
        "Lf_swap": [recs["swap_informed"][i]["verdict"] != base[i]["informed"]["verdict"] for i in ids],
        "L_case": [recs["case_plain"][i]["verdict"] != base[i]["plain"]["verdict"] for i in ids],
        "C_swap": [rules[i]["swap_reject"] != rules[i]["base_reject"] for i in ids],
        "C_case": [rules[i]["case_reject"] != rules[i]["base_reject"] for i in ids],
    }
    arr = {k: np.array(v, float) for k, v in flips.items()}
    b = int(((arr["L_swap"] == 1) & (arr["Lf_swap"] == 0)).sum())   # L flipped, Lf did not
    c = int(((arr["L_swap"] == 0) & (arr["Lf_swap"] == 1)).sum())   # Lf flipped, L did not
    diff = {"d": arr["Lf_swap"] - arr["L_swap"]}
    boot = st.mean_bootstrap(diff, resamples)["d"]
    rate = lambda k: float(arr[k].mean())  # noqa: E731
    return {"n": len(ids),
            "swap_flip_rate": {"L": rate("L_swap"), "Lf": rate("Lf_swap"), "C": rate("C_swap")},
            "case_flip_rate": {"L": rate("L_case"), "C": rate("C_case")},
            "mcnemar_swap_L_vs_Lf": {"L_flipped_Lf_not": b, "Lf_flipped_L_not": c, "p": st.mcnemar_exact(b, c)},
            "swap_diff_Lf_minus_L": {"estimate": float(diff["d"].mean()), "ci": list(st.percentile_ci(boot))}}


# --- Hypotheses A --------------------------------------------------------------------

def _sig(estimate: float, ci: list[float], p_holm: float | None, positive: bool = True) -> bool:
    excludes = ci[0] > 0 if positive else ci[1] < 0
    right_sign = estimate > 0 if positive else estimate < 0
    return bool(excludes and right_sign and p_holm is not None and p_holm < ALPHA)


def triage_hypotheses(pooled: dict[str, Any], robustness: dict[str, Any] | None,
                      robustness_msg: str) -> list[dict[str, Any]]:
    """H1-H5 from the pooled statistics. Holm runs over the tests that could be computed."""
    d, cond = pooled["diffs"], pooled["conditions"]
    raw_p = {"H2": d["CL-L:precision"]["p"], "H3": d["CL-C:f1"]["p"], "H4": d["Lf-L:f1"]["p"]}
    if robustness:
        raw_p["H5"] = robustness["mcnemar_swap_L_vs_Lf"]["p"]
    holm = st.holm(raw_p)
    item = pooled.get("item_level", {})
    out = []
    calls_l, calls_cl = cond["L"]["llm_calls"], cond["CL"]["llm_calls"]
    reduction = 1 - calls_cl / calls_l if calls_l else 0.0
    h1 = d["CL-L:f1"]
    h1_item = item.get("CL-L:f1")
    h1_supported = reduction >= MIN_CALL_REDUCTION and h1["ci"][0] > NONINFERIORITY_MARGIN
    note = (f"CL used {calls_cl} LLM calls, L used {calls_l}: {100 * reduction:.1f}% fewer. "
            f"F1 CL {cond['CL']['f1']:.3f}, L {cond['L']['f1']:.3f}.")
    if h1_item:
        note += f" Item-level CI of the difference: [{h1_item['ci'][0]:+.3f}, {h1_item['ci'][1]:+.3f}]."
    out.append({"id": "H1", "text": HYPOTHESIS_TEXT["H1"], "estimate": h1["estimate"], "ci": h1["ci"], "p_raw": None,
                "p_holm": None, "supported": bool(h1_supported), "note": note})
    for hid, key, label in (("H2", "CL-L:precision", "precision CL - L"), ("H3", "CL-C:f1", "F1 CL - C"),
                            ("H4", "Lf-L:f1", "F1 Lf - L")):
        res = d[key]
        note = f"Difference in {label}. Raw p {raw_p[hid]:.4f}, Holm-corrected {holm[hid]:.4f}."
        if hid in ("H2", "H3", "H4") and item:
            note += f" Item-level CI: [{item[key]['ci'][0]:+.3f}, {item[key]['ci'][1]:+.3f}]."
        raw_ok = res["estimate"] > 0 and res["ci"][0] > 0
        sup = _sig(res["estimate"], res["ci"], holm[hid])
        if raw_ok and not sup:
            note += " The CI excludes 0 but the Holm-corrected p is not below 0.05."
        out.append({"id": hid, "text": HYPOTHESIS_TEXT[hid], "estimate": res["estimate"], "ci": res["ci"],
                    "p_raw": raw_p[hid], "p_holm": holm[hid], "supported": sup, "note": note})
    if robustness:
        r = robustness["swap_diff_Lf_minus_L"]
        mc = robustness["mcnemar_swap_L_vs_Lf"]
        sup = bool(r["estimate"] < 0 and holm["H5"] < ALPHA)
        note = (f"Swap flip rate: L {robustness['swap_flip_rate']['L']:.3f}, Lf {robustness['swap_flip_rate']['Lf']:.3f} "
                f"on {robustness['n']} pairs. Discordant pairs: L flipped only {mc['L_flipped_Lf_not']}, Lf flipped only "
                f"{mc['Lf_flipped_L_not']}. Estimate = flip rate Lf - flip rate L. Raw p {mc['p']:.4f}, "
                f"Holm-corrected {holm['H5']:.4f}.")
        out.append({"id": "H5", "text": HYPOTHESIS_TEXT["H5"], "estimate": r["estimate"], "ci": r["ci"],
                    "p_raw": mc["p"], "p_holm": holm["H5"], "supported": sup, "note": note})
    else:
        out.append({"id": "H5", "text": HYPOTHESIS_TEXT["H5"], "estimate": None, "ci": None, "p_raw": None,
                    "p_holm": None, "supported": False, "note": "NOT EVALUATED: " + robustness_msg})
    return out


# --- Task B ----------------------------------------------------------------------------

def _flag(value: str) -> int:
    v = str(value).strip().lower()
    if v in ("1", "yes", "y", "true", "ja"):
        return 1
    if v in ("0", "no", "n", "false", "nein"):
        return 0
    raise ValueError(f"cannot read yes/no value {value!r}")


def read_ratings() -> dict[str, dict[str, dict[str, Any]]]:
    """rater -> rating_id -> {faithful, grammatical, natural, publishable}."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rater in ("R1", "R2"):
        path = make_rating_sheet.sheet_dir() / f"{rater}_scored.csv"
        if not path.exists():
            raise Incomplete(f"{path} does not exist yet (the rater's scored sheet)")
        rows: dict[str, dict[str, Any]] = {}
        for r in common.read_csv(path):
            natural = int(float(r["natural"]))
            if not 1 <= natural <= 5:
                raise ValueError(f"{path}: natural must be 1-5, got {r['natural']} in {r['rating_id']}")
            rows[r["rating_id"]] = {"faithful": _flag(r["faithful"]), "grammatical": _flag(r["grammatical"]),
                                    "natural": natural, "publishable": _flag(r["publishable"])}
        out[rater] = rows
    return out


def load_copy() -> dict[str, Any]:
    runs, bundles_file, key_file = run_copy.runs_path(), run_copy.bundles_path(), make_rating_sheet.key_path()
    for path in (runs, bundles_file, key_file):
        if not path.exists():
            raise Incomplete(f"{path} does not exist yet")
    bundles = json.loads(bundles_file.read_text(encoding="utf-8"))
    records = common.read_jsonl(runs)
    finals = make_rating_sheet.latest_finals(runs)
    expected = len(bundles) * len(COPY_CONDITIONS) * len(LANGS)
    if len(finals) != expected:
        raise Incomplete(f"copy runs unfinished: {len(finals)} of {expected} final texts")
    calls = {}
    for r in records:
        if r["kind"] == "call":
            calls[(r["bundle_id"], r["condition"], r["lang"], r["try"])] = r
    ratings = read_ratings()
    key = {(k["bundle_id"], k["condition"], k["language"]): k["rating_id"] for k in common.read_csv(key_file)}
    unrated = [rid for rid in key.values() if any(rid not in ratings[r] for r in ratings)]
    if unrated:
        raise Incomplete(f"{len(unrated)} texts in the rating key have no rating from both raters")
    return {"bundles": bundles, "finals": finals, "calls": calls, "ratings": ratings, "key": key}


def copy_cells(data: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """One record per (bundle, condition, lang): the final text, its checks, ratings and cost."""
    cells = {}
    for (bid, cond, lang), fin in data["finals"].items():
        rid = data["key"].get((bid, cond, lang))
        if rid is None:  # empty output: never sent to a rater, counts as not publishable
            rates = {"publishable_both": False, "grammatical_both": False, "faithful_both": False, "natural_mean": 0.0,
                     "pub": [0, 0], "rated": False}
        else:
            r1, r2 = data["ratings"]["R1"][rid], data["ratings"]["R2"][rid]
            rates = {"publishable_both": bool(r1["publishable"] and r2["publishable"]),
                     "grammatical_both": bool(r1["grammatical"] and r2["grammatical"]),
                     "faithful_both": bool(r1["faithful"] and r2["faithful"]),
                     "natural_mean": (r1["natural"] + r2["natural"]) / 2,
                     "pub": [r1["publishable"], r2["publishable"]], "gram": [r1["grammatical"], r2["grammatical"]],
                     "faith": [r1["faithful"], r2["faithful"]], "rated": True}
        first = "delex" if cond == "delex" else "raw"
        shared = data["calls"].get((bid, first, "all", 0))
        repairs = [c for (b, cd, lg, t), c in data["calls"].items() if b == bid and cd == cond and lg == lang and t >= 1]
        parts = ([shared] if shared else []) + repairs
        weights = ([1 / len(LANGS)] if shared else []) + [1.0] * len(repairs)
        cells[(bid, cond, lang)] = {
            **fin, **rates,
            "calls": float(sum(weights)),
            "tokens": float(sum(w * _tokens(c) for w, c in zip(weights, parts))),
            "seconds": float(sum(w * (c.get("wall_ms") or 0) / 1000 for w, c in zip(weights, parts))),
        }
    return cells


def compute_copy(data: dict[str, Any], resamples: int) -> dict[str, Any]:
    cells = copy_cells(data)
    bundle_ids = [b["id"] for b in data["bundles"]]
    summary: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    for group in GROUPS_COPY:
        langs = LANGS if group == "pooled" else (group,)
        per_bundle = {c: np.array([np.mean([cells[(b, c, lg)]["publishable_both"] for lg in langs]) for b in bundle_ids])
                      for c in COPY_CONDITIONS}
        boot = st.mean_bootstrap(per_bundle, resamples)
        summary[group] = {}
        for cond in COPY_CONDITIONS:
            sel = [cells[(b, cond, lg)] for b in bundle_ids for lg in langs]
            n = len(sel)
            llm = [c for c in sel if c["source"] == "llm"]
            checks_first: dict[str, int] = {}
            checks_final: dict[str, int] = {}
            for c in sel:
                for k in c["first_try_failed_checks"]:
                    checks_first[k] = checks_first.get(k, 0) + 1
                for k in c["failed_checks"]:
                    checks_final[k] = checks_final.get(k, 0) + 1
            summary[group][cond] = {
                "n_texts": n,
                "publishable_share": float(per_bundle[cond].mean()),
                "publishable_ci": list(st.percentile_ci(boot[cond])),
                "llm_share": len(llm) / n, "template_share": sum(c["source"] == "template" for c in sel) / n,
                "none_share": sum(c["source"] == "none" for c in sel) / n,
                "checks_pass_share": sum(not c["failed_checks"] for c in sel) / n,
                "first_try_pass_share": sum(not c["first_try_failed_checks"] for c in sel) / n,
                "n_llm_texts": len(llm),
                "publishable_share_among_llm_texts": (sum(c["publishable_both"] for c in llm) / len(llm)) if llm else None,
                "mean_natural": sum(c["natural_mean"] for c in sel) / n,
                "retries_total": sum(c["tries"] - 1 for c in sel),
                "texts_with_retry": sum(c["tries"] > 1 for c in sel),
                "calls": round(sum(c["calls"] for c in sel), 3), "tokens": round(sum(c["tokens"] for c in sel)),
                "seconds": round(sum(c["seconds"] for c in sel), 3),
                "first_try_failed_checks": checks_first, "final_failed_checks": checks_final,
            }
        tests[group] = {}
        for name, cond in (("H6", "guarded"), ("H7", "delex")):
            diff = boot[cond] - boot["raw"]
            tests[group][name] = {"estimate": float(per_bundle[cond].mean() - per_bundle["raw"].mean()),
                                  "ci": list(st.percentile_ci(diff)), "p": float(st.bootstrap_p(diff))}
    # Rater agreement over all rated texts
    rated = [c for c in cells.values() if c["rated"]]
    kappa = {}
    for dim, key in (("publishable", "pub"), ("grammatical", "gram"), ("faithful", "faith")):
        a, b = [c[key][0] for c in rated], [c[key][1] for c in rated]
        kappa[dim] = {"kappa": st.cohen_kappa(a, b) if a else None, "n": len(a),
                      "percent_agreement": 100 * sum(x == y for x, y in zip(a, b)) / len(a) if a else None}
    return {"n_bundles": len(bundle_ids), "summary": summary, "tests": tests, "rater_kappa": kappa,
            "n_empty_not_rated": len(cells) - len(rated)}, cells


def copy_hypotheses(copy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for hid, cond in (("H6", "guarded"), ("H7", "delex")):
        t = copy["tests"]["pooled"][hid]
        per_lang = ", ".join(f"{lg}: {copy['tests'][lg][hid]['estimate']:+.3f} [{copy['tests'][lg][hid]['ci'][0]:+.3f}, "
                             f"{copy['tests'][lg][hid]['ci'][1]:+.3f}] p={copy['tests'][lg][hid]['p']:.4f}" for lg in LANGS)
        s = copy["summary"]["pooled"]
        note = (f"Publishable share {cond} {s[cond]['publishable_share']:.3f} vs raw {s['raw']['publishable_share']:.3f} "
                f"(pooled over {copy['n_bundles']} bundles x 3 languages). Per language, difference [95% CI] and raw p: "
                f"{per_lang}. Not part of the Holm family; raw p only.")
        out.append({"id": hid, "text": HYPOTHESIS_TEXT[hid], "estimate": t["estimate"], "ci": t["ci"], "p_raw": t["p"],
                    "p_holm": None, "supported": bool(t["estimate"] > 0 and t["ci"][0] > 0), "note": note})
    return out


def skipped_copy_hypotheses(msg: str) -> list[dict[str, Any]]:
    return [{"id": h, "text": HYPOTHESIS_TEXT[h], "estimate": None, "ci": None, "p_raw": None, "p_holm": None,
             "supported": False, "note": "NOT EVALUATED: " + msg} for h in ("H6", "H7")]


# --- Orchestration -----------------------------------------------------------------------

# --- Human check (amendment A3) ------------------------------------------------------

def human_check_path() -> Path:
    return common.ROOT / "eval" / "human" / "human_labels_report.json"


def load_human_check(path: Path | None = None) -> dict[str, Any] | None:
    """The compact result of experiments/human_audit.py --score, or None if nobody has labelled yet.
    It validates the silver labels; it changes no result above it."""
    path = path or human_check_path()
    if not path.exists():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    patterns: dict[str, int] = {}
    for d in rep["disagreements"]:
        key = f"{d['gold']} -> {d['human']}"
        patterns[key] = patterns.get(key, 0) + 1
    keep = ("n", "kappa_3class", "kappa_binary", "percent_agreement_3class", "percent_agreement_binary")
    out = {"n": rep["labelled_rows"], "pooled": {k: rep["pooled"][k] for k in keep},
           "by_set": {s: {k: m[k] for k in keep} for s, m in rep["by_set"].items() if m["n"]},
           "confidence_counts": {c: m["n"] for c, m in rep.get("by_confidence", {}).items()},
           "n_disagreements": len(rep["disagreements"]),
           "disagreement_patterns": dict(sorted(patterns.items(), key=lambda kv: -kv[1]))}
    if "systems" in rep:
        sy = rep["systems"]
        out["systems"] = {lab: {"positives": sy[lab]["positives"],
                                "f1": {c: m["f1"] for c, m in sy[lab]["conditions"].items()},
                                "precision": {c: m["precision"] for c, m in sy[lab]["conditions"].items()},
                                "recall": {c: m["recall"] for c, m in sy[lab]["conditions"].items()},
                                "diffs": {k: sy[lab]["diffs"][k] for k in ("CL-L:f1", "CL-L:precision")}}
                          for lab in ("silver", "human")}
    return out


def render_human_check(hc: dict[str, Any]) -> list[str]:
    p = hc["pooled"]
    out = ["## Human check of the labels (amendment A3, after the results)", "",
           f"{hc['n']} test pairs, stratified by set and gold label, labelled blind by the author with the same guide. "
           "The author knows the study, so this is not an outside annotator. Agreement with the adjudicated silver gold; "
           "nothing above this section was changed by it.", "",
           _table(["slice", "n", "3-class kappa", "binary kappa", "3-class %", "binary %"],
                  [["pooled", p["n"], _f(p["kappa_3class"]), _f(p["kappa_binary"]), _f(p["percent_agreement_3class"], 1),
                    _f(p["percent_agreement_binary"], 1)]]
                  + [[s, m["n"], _f(m["kappa_3class"]), _f(m["kappa_binary"]), _f(m["percent_agreement_3class"], 1),
                      _f(m["percent_agreement_binary"], 1)] for s, m in hc["by_set"].items()]), "",
           f"{hc['n_disagreements']} disagreements (gold -> human): "
           + (", ".join(f"{k} {v}" for k, v in hc["disagreement_patterns"].items()) or "none") + ".", ""]
    if "systems" in hc:
        sy = hc["systems"]
        out += [f"Exploratory: the same predictions on these {hc['n']} pairs, scored against each label set "
                f"(complements: silver {sy['silver']['positives']}, human {sy['human']['positives']}).", "",
                _table(["condition", "F1 silver", "F1 human", "precision silver", "precision human"],
                       [[c, _f(sy["silver"]["f1"][c]), _f(sy["human"]["f1"][c]), _f(sy["silver"]["precision"][c]),
                         _f(sy["human"]["precision"][c])] for c in CONDITIONS]), ""]
        for key in ("CL-L:f1", "CL-L:precision"):
            out.append(f"- {key}: silver {sy['silver']['diffs'][key]['estimate']:+.3f} {_ci(sy['silver']['diffs'][key]['ci'])}, "
                       f"human {sy['human']['diffs'][key]['estimate']:+.3f} {_ci(sy['human']['diffs'][key]['ci'])}")
        out.append("")
    return out


def _repo_relative(path: Path | str) -> str:
    """The path inside the repository, so no local folder names end up in the results."""
    try:
        return str(Path(path).resolve().relative_to(common.ROOT.resolve()))
    except ValueError:
        return Path(path).name


def build(*, triage_only: bool = False, variant: str | None = None, pipeline_path: str | None = None,
          resamples: int = st.DEFAULT_RESAMPLES, set_ids: tuple[str, ...] = common.TEST_SETS
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The summary dict and the internal per-pair / per-text records the exporter needs."""
    variant = variant or common.winner_variant()
    path = common.resolve_pipeline_path(pipeline_path, set_ids)
    pipeline = cl_features.load_pipeline(path)
    rows, skipped, info = load_triage(set_ids, variant, pipeline)
    messages = [f"Task A: set {s} skipped, {why}" for s, why in skipped.items()]
    summary: dict[str, Any] = {"meta": {"model": common.MODEL, "prompt_variant": variant, "pipeline": _repo_relative(path),
                                        "pipeline_sha256": common.sha256_file(path), "resamples": resamples,
                                        "positive_class": "complement", "sets_analysed": sorted(info)},
                               "messages": messages}
    internal: dict[str, Any] = {"rows": rows, "set_info": info}
    hypotheses: list[dict[str, Any]] = []
    if rows:
        triage = {s: group_stats([r for r in rows if r["set"] == s], resamples) for s in sorted(info)}
        triage["pooled"] = group_stats(rows, resamples, with_items=True)
        summary["triage"] = triage
        try:
            robustness, rmsg = compute_robustness(rows, variant, resamples), ""
        except Incomplete as exc:
            robustness, rmsg = None, str(exc)
            messages.append("Task A: H5 and flip rates not evaluated, " + rmsg)
        summary["robustness"] = robustness
        hypotheses += triage_hypotheses(triage["pooled"], robustness, rmsg)
    else:
        messages.append("Task A: no finished set; nothing to analyse")
    internal["copy_cells"] = None
    if triage_only:
        messages.append("Task B skipped (--triage-only)")
        hypotheses += skipped_copy_hypotheses("--triage-only")
    else:
        try:
            data = load_copy()
            copy, cells = compute_copy(data, resamples)
            summary["copy"] = copy
            hypotheses += copy_hypotheses(copy)
            internal.update(copy_cells=cells, bundles=data["bundles"])
        except Incomplete as exc:
            messages.append(f"Task B not evaluated: {exc}")
            hypotheses += skipped_copy_hypotheses(str(exc))
    summary["hypotheses"] = hypotheses
    human = load_human_check()
    if human:
        summary["human_check"] = human
    return summary, internal


# --- Report ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _f(x: Any, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(ci: Any) -> str:
    return "-" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def render_report(s: dict[str, Any]) -> str:
    out = ["# Smart Flyer study: results", "",
           f"Model {s['meta']['model']}, triage prompt `{s['meta']['prompt_variant']}`, positive class = complement, "
           f"{s['meta']['resamples']} bootstrap resamples. Pipeline SHA-256 `{s['meta']['pipeline_sha256']}`.", ""]
    out += ["Limitations: the gold labels come from two LLM annotators and an LLM adjudicator, so this is a silver standard. "
            "t3 is constructed, not real shopping data. The results describe this small model only. Latency is from one "
            "busy laptop; call and token counts are the primary cost measures.", ""]
    if s["messages"]:
        out += ["## Status", ""] + [f"- {m}" for m in s["messages"]] + [""]
    tri = s.get("triage")
    if tri:
        p = tri["pooled"]["conditions"]
        calls = f"CL used {p['CL']['llm_calls']} LLM calls against {p['L']['llm_calls']} for L"
        lead = "L" if p["L"]["f1"] > p["CL"]["f1"] else "CL"
        out += ["## Summary", "",
                f"Pooled over {', '.join(s['meta']['sets_analysed'])}: F1 L {p['L']['f1']:.3f}, C {p['C']['f1']:.3f}, CL "
                f"{p['CL']['f1']:.3f}, Lf {p['Lf']['f1']:.3f}, CLf {p['CLf']['f1']:.3f}. Precision L {p['L']['precision']:.3f}, "
                f"CL {p['CL']['precision']:.3f}. {calls}. "
                + (f"L has the higher F1 than CL ({p['L']['f1']:.3f} vs {p['CL']['f1']:.3f})." if lead == "L" else
                   f"CL has the higher F1 than L ({p['CL']['f1']:.3f} vs {p['L']['f1']:.3f})."), ""]
    out += ["## Hypotheses", "",
            "A hypothesis in H2-H4 counts as supported when the estimate has the predicted sign, the paired 95% bootstrap "
            "CI excludes 0 and the Holm-corrected p (across H2-H5) is below 0.05. H5 needs the predicted sign and a "
            "Holm-corrected exact McNemar p below 0.05. H1 needs at least 50% fewer calls and a CI lower bound above "
            "-0.03. H6 and H7 need a positive difference with a CI that excludes 0 (raw p; not in the Holm family).", ""]
    for h in s["hypotheses"]:
        verdict = "supported" if h["supported"] else "not supported"
        if h["note"].startswith("NOT EVALUATED"):
            verdict = "not evaluated"
        est = "-" if h["estimate"] is None else f"{h['estimate']:+.4f}"
        out += [f"**{h['id']}: {verdict}.** {h['text']}", "",
                f"Estimate {est}, 95% CI {_ci(h['ci'])}, raw p {_f(h['p_raw'], 4)}, Holm p {_f(h['p_holm'], 4)}. {h['note']}", ""]
    if tri:
        out += ["## Task A: triage", ""]
        for group in [*sorted(k for k in tri if k != "pooled"), "pooled"]:
            g = tri[group]
            out += [f"### {group} (n = {g['n']})", "", _table(
                ["cond", "precision", "recall", "F1", "F1 95% CI", "accuracy", "kept", "tp", "fp", "fn", "tn",
                 "LLM calls", "tokens", "seconds"],
                [[c, _f(m["precision"]), _f(m["recall"]), _f(m["f1"]), f"[{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}]",
                  _f(m["accuracy"]), m["kept"], m["tp"], m["fp"], m["fn"], m["tn"], m["llm_calls"], m["llm_tokens"],
                  _f(m["llm_seconds"], 1)] for c, m in g["conditions"].items()]), ""]
            out += [_table(["difference", "estimate", "95% CI", "p (bootstrap)"],
                           [[k, f"{v['estimate']:+.4f}", _ci(v["ci"]), _f(v["p"], 4)] for k, v in g["diffs"].items()]), ""]
            mc = g["mcnemar_L_vs_CL"]
            out += [f"Exact McNemar, L vs CL (per-pair correctness): L right and CL wrong {mc['L_right_CL_wrong']}, "
                    f"CL right and L wrong {mc['CL_right_L_wrong']}, p = {mc['p']:.4f}.", ""]
        item = tri["pooled"].get("item_level")
        if item:
            out += ["### Item-level sensitivity (pooled; product names resampled instead of pairs)", "",
                    _table(["difference", "estimate", "95% CI"],
                           [[k, f"{item[k]['estimate']:+.4f}", _ci(item[k]["ci"])] for k in DIFFS]), ""]
        out += ["### Confusion by gold class (pooled): predicted complement / predicted other", "",
                _table(["gold class", "n", *CONDITIONS],
                       [[gc, tri["pooled"]["confusion"]["by_condition"]["L"][gc]["n"],
                         *[f"{tri['pooled']['confusion']['by_condition'][c][gc]['predicted_complement']} / "
                           f"{tri['pooled']['confusion']['by_condition'][c][gc]['predicted_other']}" for c in CONDITIONS]]
                        for gc in common.GOLD_LABELS]), ""]
        for key in ("plain", "informed"):
            v = tri["pooled"]["confusion"]["llm_verdicts"][key]
            out += [f"LLM verdicts by gold class, {key} prompt:", "",
                    _table(["gold class", *common.VERDICTS], [[gc, *[v[gc][x] for x in common.VERDICTS]]
                                                              for gc in common.GOLD_LABELS]), ""]
        inv_rows = []
        for group in [*sorted(k for k in tri if k != "pooled"), "pooled"]:
            for key in ("plain", "informed"):
                i = tri[group]["invalid"][key]
                inv_rows.append([group, key, i["n"], i["invalid"], i["call_errors"], i["invalid_among_cascade_survivors"]])
        out += ["### INVALID verdicts (unparseable answers and failed calls count as not positive)", "",
                _table(["set", "prompt", "pairs", "INVALID", "call errors", "INVALID among survivors"], inv_rows), ""]
    rob = s.get("robustness")
    if rob:
        out += ["### Robustness flips (verdict changed)", "",
                _table(["", "L", "Lf", "C"], [["swap of product order", _f(rob["swap_flip_rate"]["L"]),
                                               _f(rob["swap_flip_rate"]["Lf"]), _f(rob["swap_flip_rate"]["C"])],
                                              ["letter-case change", _f(rob["case_flip_rate"]["L"]), "not run",
                                               _f(rob["case_flip_rate"]["C"])]]),
                f"{rob['n']} seeded test pairs. A flip is any change of the LLM's verdict word (INVALID included) or of the rule decision.", ""]
    cp = s.get("copy")
    if cp:
        out += ["## Task B: flyer copy", "",
                f"{cp['n_bundles']} bundles x 3 languages x 3 conditions. Publishable = both raters said yes; an empty output "
                f"counts as not publishable ({cp['n_empty_not_rated']} empty texts were not rated).", ""]
        for group in GROUPS_COPY:
            g = cp["summary"][group]
            out += [f"### {group}", "", _table(
                ["cond", "publishable", "95% CI", "LLM-written", "template", "no output", "checks pass (final)",
                 "checks pass (1st try)", "publishable among LLM texts", "mean natural", "retries", "calls", "tokens", "seconds"],
                [[c, _f(m["publishable_share"]), f"[{m['publishable_ci'][0]:.3f}, {m['publishable_ci'][1]:.3f}]",
                  _f(m["llm_share"]), _f(m["template_share"]), _f(m["none_share"]), _f(m["checks_pass_share"]),
                  _f(m["first_try_pass_share"]), _f(m["publishable_share_among_llm_texts"]), _f(m["mean_natural"], 2),
                  m["retries_total"], _f(m["calls"], 1), m["tokens"], _f(m["seconds"], 1)] for c, m in g.items()]), ""]
        checks = sorted({k for c in cp["summary"]["pooled"].values() for k in (*c["first_try_failed_checks"], *c["final_failed_checks"])})
        if checks:
            out += ["### Check failures (texts failing each check, all languages)", "", _table(
                ["check", *[f"{c} 1st try" for c in COPY_CONDITIONS], *[f"{c} final" for c in COPY_CONDITIONS]],
                [[k, *[cp["summary"]["pooled"][c]["first_try_failed_checks"].get(k, 0) for c in COPY_CONDITIONS],
                  *[cp["summary"]["pooled"][c]["final_failed_checks"].get(k, 0) for c in COPY_CONDITIONS]] for k in checks]), ""]
        out += ["### Per-language tests (difference of publishable share to raw)", "", _table(
            ["language", "H6 guarded-raw", "95% CI", "p", "H7 delex-raw", "95% CI", "p"],
            [[g, f"{cp['tests'][g]['H6']['estimate']:+.3f}", _ci(cp["tests"][g]["H6"]["ci"]), _f(cp["tests"][g]["H6"]["p"], 4),
              f"{cp['tests'][g]['H7']['estimate']:+.3f}", _ci(cp["tests"][g]["H7"]["ci"]), _f(cp["tests"][g]["H7"]["p"], 4)]
             for g in GROUPS_COPY]), ""]
        out += ["### Rater agreement (Cohen's kappa)", "", _table(
            ["dimension", "kappa", "% agreement", "texts"],
            [[d, _f(k["kappa"]), _f(k["percent_agreement"], 1), k["n"]] for d, k in cp["rater_kappa"].items()]), "",
            "Calls for guarded include the shared first call it reuses from raw; each three-language first call is split "
            "evenly over the three languages in per-language rows.", ""]
    if s.get("human_check"):
        out += render_human_check(s["human_check"])
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--triage-only", action="store_true", help="skip Task B (copy and ratings)")
    parser.add_argument("--variant", help="triage prompt variant (default: the winner in prompt_selection.json)")
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy)")
    parser.add_argument("--root", help="project tree to read (default: this repository)")
    parser.add_argument("--out-dir", help="where summary.json and REPORT.md go (default: experiments/results)")
    parser.add_argument("--resamples", type=int, default=st.DEFAULT_RESAMPLES)
    args = parser.parse_args()
    if args.root:
        common.ROOT = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else common.results_dir()
    summary, _ = build(triage_only=args.triage_only, variant=args.variant, pipeline_path=args.pipeline,
                       resamples=args.resamples)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    for message in summary["messages"]:
        print(message, flush=True)
    print(f"Wrote {out_dir / 'summary.json'} and {out_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
