"""A small invented study tree (no real data) for the analyze and export tests."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import common
import make_rating_sheet as mrs
import run_copy
import run_robustness as rr
import run_triage as rt

VARIANT = "definitions"
EN_PAIRS = [("RED MUG", "BLUE MUG"), ("TEA POT", "TEA CUP"), ("POSTAGE", "RED MUG"), ("SALT SHAKER", "PEPPER MILL"),
            ("WOODEN SPOON", "WOODEN FORK"), ("CAKE TIN", "CAKE CASES"), ("CANDLE HOLDER", "MATCH BOX"),
            ("SOAP DISH", "SOAP BAR"), ("PAPER NAPKINS", "PAPER PLATES"), ("GARDEN GLOVES", "PLANT POT"),
            ("PHOTO FRAME", "ALBUM BOOK"), ("TOY CAR", "TOY TRAIN")]
DE_PAIRS = [("Bio Vollmilch 1L", "Bio Fettarme Milch 1L"), ("Spaghetti 500g", "Tomatensauce 400g"),
            ("Erdbeerjoghurt 500g", "Vanillejoghurt 500g"), ("Kaffee 250g", "Filtertüten"),
            ("Butter 250g", "Brötchen 6 Stück"), ("Pfand Einweg 0,25 €", "Cola 1L"), ("Käse 200g", "Wein 0,75L"),
            ("Brot 500g", "Marmelade 450g"), ("Eier 10er", "Mehl 1kg"), ("Apfelsaft 1L", "Orangensaft 1L"),
            ("Chips 150g", "Salsa 300g"), ("Tee 20 Beutel", "Honig 250g")]
LABELS = list(common.GOLD_LABELS)
VERDICT_BY_LABEL = common.LABEL_TO_VERDICT


def build(root: Path, pipeline_src: Path, seed: int = 7) -> None:
    """Write a complete study tree under root (every task finished)."""
    rng = random.Random(seed)
    common.ROOT = root
    exp = root / "experiments"
    (exp / "frozen").mkdir(parents=True)
    shutil.copy(pipeline_src, exp / "frozen" / "export_smart_flyer_v1.py")
    ts = common.testsets_dir()
    ts.mkdir(parents=True)
    manifest = {"sets": {"t1": {"baskets_after_cleaning": 1000}, "t2": {"baskets_after_cleaning": 500},
                         "t3": {"baskets_after_cleaning": None}}}
    (ts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    agreement = {}
    pair_header = ["pair_id", "set", "a", "b", "support", "confidence", "lift", "price_a", "price_b", "lang_a", "lang_b"]
    rows_by_set = {}
    for set_id, base in (("t1", EN_PAIRS), ("t2", EN_PAIRS[::-1]), ("t3", DE_PAIRS)):
        lang = "de" if set_id == "t3" else "en"
        rows = [{"pair_id": f"{set_id}-{i:04d}", "set": set_id, "a": a, "b": b, "support": 0.02, "confidence": 0.3,
                 "lift": "" if set_id == "t3" else round(2 + i / 3, 2), "price_a": 2.5, "price_b": 1.5 + i / 10,
                 "lang_a": lang, "lang_b": lang} for i, (a, b) in enumerate(base, start=1)]
        rows_by_set[set_id] = rows
        common.write_csv(common.pairs_path(set_id), pair_header, rows)
        gold = [{"pair_id": r["pair_id"], "label": LABELS[(i * 7 + len(set_id)) % 3] if i % 4 else "complement"}
                for i, r in enumerate(rows, start=1)]
        common.write_csv(common.final_gold_path(set_id), ["pair_id", "label"], gold)
        agreement[set_id] = {"kappa_3class": 0.7, "kappa_binary": 0.75}
    (common.gold_dir() / "agreement.json").write_text(json.dumps(agreement), encoding="utf-8")
    res = common.results_dir()
    res.mkdir(parents=True, exist_ok=True)
    (res / "prompt_selection.json").write_text(json.dumps({"winner": VARIANT}), encoding="utf-8")

    def verdict(label: str, noise: float) -> str:
        return VERDICT_BY_LABEL[label] if rng.random() > noise else rng.choice(common.VERDICTS)

    def record(pid, set_id, cond, v):
        return {"pair_id": pid, "set": set_id, "condition": cond, "variant": VARIANT, "raw": v, "verdict": v,
                "prompt_eval_count": 300, "eval_count": 3, "wall_ms": 900.0, "attempts": 1, "prompt_sha": "x"}

    all_gold = {}
    for set_id, rows in rows_by_set.items():
        gold = common.read_gold(set_id)
        all_gold.update(gold)
        for cond, noise in (("plain", 0.5), ("informed", 0.3)):
            for r in rows:
                common.append_jsonl(rt.results_path(set_id, cond, VARIANT),
                                    record(r["pair_id"], set_id, cond, verdict(gold[r["pair_id"]], noise)))
    ids = {s: [r["pair_id"] for r in rows[:8]] for s, rows in rows_by_set.items()}
    (res / "robustness_ids.json").write_text(json.dumps({"seed": 1, "ids": ids}), encoding="utf-8")
    for kind in rr.KINDS:
        for s, pids in ids.items():
            for pid in pids:
                common.append_jsonl(rr.llm_path(kind, VARIANT), record(pid, s, "x", rng.choice(common.VERDICTS)))
    for s, pids in ids.items():
        for pid in pids:
            base = rng.random() < 0.5
            common.append_jsonl(rr.rules_path(), {"pair_id": pid, "set": s, "base_reject": base,
                                                  "swap_reject": base, "case_reject": base if rng.random() < 0.8 else not base})
    build_copy(root, rng)


def build_copy(root: Path, rng: random.Random, n_bundles: int = 6) -> None:
    res = common.results_dir()
    bundles = [{"id": f"b{i:02d}", "set": ("t1", "t2", "t3")[i % 3], "pair_id": f"x-{i}", "anchor": f"RED MUG {i}",
                "pull": f"BLUE MUG {i}", "pct": 20, "lift": 3.0} for i in range(1, n_bundles + 1)]
    (res / "copy_bundles.json").write_text(json.dumps(bundles), encoding="utf-8")
    runs = run_copy.runs_path()
    for b in bundles:
        for cond in ("raw", "delex"):
            common.append_jsonl(runs, {"kind": "call", "bundle_id": b["id"], "condition": cond, "lang": "all", "try": 0,
                                       "prompt": "p", "raw": "{}", "prompt_eval_count": 200, "eval_count": 300,
                                       "wall_ms": 30000.0, "attempts": 1})
        for cond in ("raw", "guarded", "delex"):
            for lang in run_copy.LANGS:
                retried = cond != "raw" and rng.random() < 0.4
                if retried:
                    common.append_jsonl(runs, {"kind": "call", "bundle_id": b["id"], "condition": cond, "lang": lang,
                                               "try": 1, "prompt": "q", "raw": "{}", "prompt_eval_count": 100,
                                               "eval_count": 60, "wall_ms": 5000.0, "attempts": 1})
                empty = cond == "raw" and rng.random() < 0.1
                template = retried and rng.random() < 0.5
                failed = ["names"] if (cond == "raw" and not empty and rng.random() < 0.5) else []
                common.append_jsonl(runs, {
                    "kind": "final", "bundle_id": b["id"], "set": b["set"], "condition": cond, "lang": lang,
                    "title": "" if empty else f"Title {lang}", "copy": "" if empty else f"Copy {lang} {b['id']}",
                    "source": "none" if empty else ("template" if template else "llm"), "tries": 2 if retried else 1,
                    "failed_checks": failed, "first_try_failed_checks": failed or (["number"] if retried else []),
                    "oov_words": []})
    finals = mrs.latest_finals(runs)
    sheet, key = mrs.build_rows(finals, bundles)
    common.write_csv(mrs.key_path(), mrs.KEY_HEADER, key)
    for rater in ("R1", "R2"):
        rows = [{"rating_id": s["rating_id"], "faithful": rng.randint(0, 1), "grammatical": rng.randint(0, 1),
                 "natural": rng.randint(1, 5), "publishable": int(rng.random() < 0.6)} for s in sheet]
        common.write_csv(mrs.sheet_dir() / f"{rater}_scored.csv",
                         ["rating_id", "faithful", "grammatical", "natural", "publishable"], rows)
