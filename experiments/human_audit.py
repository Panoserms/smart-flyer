#!/usr/bin/env python3
"""Human spot-check audit for the silver-standard test labels (eval/gold/final_{t1,t2,t3}.csv).

Creates a blind sample sheet without gold labels, optionally runs an interactive terminal
labelling session, and scores a completed human sheet against the adjudicated LLM gold labels
(3-class and binary Cohen's kappa, percent agreement, confusion matrix, and disagreements).

  python experiments/human_audit.py --sample
  python experiments/human_audit.py --interactive
  python experiments/human_audit.py --score eval/gold/human_audit_blind.csv

Two batches of 50 for the browser app (eval/human/label_pairs.html):

  python experiments/human_audit.py --batches
  python experiments/human_audit.py --score eval/human/human_labels_batch_1.csv eval/human/human_labels_batch_2.csv \
      --report-out eval/human/human_labels_report.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import study_stats  # noqa: E402

AUDIT_SEED = 20261111
BLIND_COLUMNS = ("pair_id", "set", "a", "b", "label", "note")
SHORTCUTS = {
    "1": "complement",
    "c": "complement",
    "complement": "complement",
    "2": "substitute",
    "s": "substitute",
    "substitute": "substitute",
    "3": "not_a_bundle",
    "n": "not_a_bundle",
    "not_a_bundle": "not_a_bundle",
}


BATCH_COLUMNS = ("pair_id", "set", "a", "b", "label", "confidence", "note")
APP_FIELDS = ("pair_id", "set", "a", "b")  # all the app gets: never the gold or AI labels
TEMPLATE_PATH = Path(__file__).resolve().parent / "label_app_template.html"
DATA_SLOT = "/*__DATA__*/[]"


def human_dir() -> Path:
    return common.ROOT / "eval" / "human"


def default_blind_path() -> Path:
    return common.ROOT / "eval" / "gold" / "human_audit_blind.csv"


def default_report_path() -> Path:
    return common.ROOT / "eval" / "gold" / "human_audit_report.json"


def sample_blind_rows(n_per_set: int = 10, seed: int = AUDIT_SEED,
                      sets: tuple[str, ...] = common.TEST_SETS) -> list[dict[str, str]]:
    """Sample n_per_set pairs per test set and shuffle into a blind annotation sheet."""
    rows: list[dict[str, str]] = []
    for set_id in sets:
        pairs = sorted(common.read_pairs(set_id), key=lambda p: p["pair_id"])
        rng = random.Random(f"{seed}:{set_id}")
        picked = rng.sample(pairs, min(n_per_set, len(pairs)))
        for p in picked:
            rows.append({
                "pair_id": p["pair_id"],
                "set": set_id,
                "a": p["a"],
                "b": p["b"],
                "label": "",
                "note": "",
            })
    random.Random(seed).shuffle(rows)
    return rows


def write_blind_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    common.write_csv(path, BLIND_COLUMNS, rows)


def _allocate(sizes: dict[Any, int], total: int) -> dict[Any, int]:
    """Largest-remainder allocation of `total` across strata in proportion to their sizes."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    quota = {k: int(x) for k, x in exact.items()}
    left = total - sum(quota.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:left]:
        quota[k] += 1
    return quota


def sample_batches(n_batches: int = 2, batch_size: int = 50, seed: int = AUDIT_SEED,
                   sets: tuple[str, ...] = common.TEST_SETS) -> list[list[dict[str, str]]]:
    """A stratified sample of test pairs (set x gold label), split into equal batches.

    The gold label is used only to stratify, so each batch covers every set and every label in proportion;
    the rows carry no label. Each stratum is split evenly between the batches, and the odd items are placed
    so that sets and labels stay balanced (exhaustive search over the few odd strata).
    """
    if n_batches != 2:
        raise ValueError("balanced splitting is implemented for two batches")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for set_id in sets:
        gold = common.read_gold(set_id)
        for p in sorted(common.read_pairs(set_id), key=lambda p: p["pair_id"]):
            strata.setdefault((set_id, gold[p["pair_id"]]), []).append(p)
    quota = _allocate({k: len(v) for k, v in strata.items()}, n_batches * batch_size)
    rng = random.Random(f"{seed}:batches")
    picked = {k: rng.sample(strata[k], quota[k]) for k in sorted(strata)}

    halves: list[list[tuple[tuple[str, str], dict[str, Any]]]] = [[], []]
    odd: list[tuple[tuple[str, str], dict[str, Any]]] = []
    for k, items in picked.items():
        for i in range(len(items) // 2 * 2):
            halves[i % 2].append((k, items[i]))
        if len(items) % 2:
            odd.append((k, items[-1]))

    def imbalance(choice: tuple[int, ...]) -> tuple[int, int]:
        trial = [list(h) for h in halves]
        for (k, item), side in zip(odd, choice):
            trial[side].append((k, item))
        size_gap = abs(len(trial[0]) - len(trial[1]))
        worst = 0
        for dim in (0, 1):
            for v in {k[dim] for k in picked}:
                worst = max(worst, abs(sum(k[dim] == v for k, _ in trial[0]) - sum(k[dim] == v for k, _ in trial[1])))
        return size_gap, worst

    choices = list(itertools.product((0, 1), repeat=len(odd)))
    best = min(imbalance(c) for c in choices)
    best_choices = [c for c in choices if imbalance(c) == best]
    for (k, item), side in zip(odd, rng.choice(best_choices)):
        halves[side].append((k, item))

    batches: list[list[dict[str, str]]] = []
    for half in halves:
        rows = [{"pair_id": p["pair_id"], "set": k[0], "a": p["a"], "b": p["b"], "label": "", "confidence": "", "note": ""}
                for k, p in half]
        rng.shuffle(rows)
        batches.append(rows)
    return batches


def write_batch_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    common.write_csv(path, list(BATCH_COLUMNS), rows)


def write_label_app(path: Path, batches: list[list[dict[str, str]]]) -> None:
    """The self-contained labelling app: the template with the batches (ids and product names only) inlined."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(DATA_SLOT) != 1:
        raise ValueError(f"{TEMPLATE_PATH}: expected exactly one {DATA_SLOT}")
    data = [[{f: r[f] for f in APP_FIELDS} for r in batch] for batch in batches]
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace(DATA_SLOT, payload), encoding="utf-8")


def _metrics_for_slice(items: list[dict[str, str]]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {"n": 0, "kappa_3class": 0.0, "kappa_binary": 0.0,
                "percent_agreement_3class": 0.0, "percent_agreement_binary": 0.0}
    human_3 = [r["human"] for r in items]
    gold_3 = [r["gold"] for r in items]
    human_bin = ["complement" if x == "complement" else "other" for x in human_3]
    gold_bin = ["complement" if x == "complement" else "other" for x in gold_3]
    agree_3 = sum(h == g for h, g in zip(human_3, gold_3)) / n
    agree_bin = sum(h == g for h, g in zip(human_bin, gold_bin)) / n
    matrix = {
        g_cls: {h_cls: sum(r["gold"] == g_cls and r["human"] == h_cls for r in items)
                for h_cls in common.GOLD_LABELS}
        for g_cls in common.GOLD_LABELS
    }
    return {
        "n": n,
        "kappa_3class": round(study_stats.cohen_kappa(human_3, gold_3), 4),
        "kappa_binary": round(study_stats.cohen_kappa(human_bin, gold_bin), 4),
        "percent_agreement_3class": round(100.0 * agree_3, 2),
        "percent_agreement_binary": round(100.0 * agree_bin, 2),
        "confusion_gold_rows_human_cols": matrix,
    }


def score_human_sheet(path: Path, sets: tuple[str, ...] = common.TEST_SETS) -> dict[str, Any]:
    """Compare a human-labelled CSV against eval/gold/final_<set>.csv."""
    return score_human_sheets([path], sets)


def score_human_sheets(paths: list[Path], sets: tuple[str, ...] = common.TEST_SETS) -> dict[str, Any]:
    """Compare one or more human-labelled CSVs (for example both batches) against eval/gold/final_<set>.csv."""
    raw_rows: list[dict[str, str]] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for row in common.read_csv(path):
            pid = (row.get("pair_id") or "").strip()
            if pid in seen and (row.get("label") or "").strip():
                raise ValueError(f"{path}: pair {pid} is also in {seen[pid]}")
            if (row.get("label") or "").strip():
                seen[pid] = path
            raw_rows.append(row)
    path = paths[0] if len(paths) == 1 else ", ".join(str(p) for p in paths)
    gold_by_set = {s: common.read_gold(s) for s in sets}
    gold_all = {pid: label for g in gold_by_set.values() for pid, label in g.items()}

    scored: list[dict[str, str]] = []
    for row in raw_rows:
        pid = (row.get("pair_id") or "").strip()
        raw_label = (row.get("label") or "").strip().lower()
        if not raw_label:
            continue
        label = SHORTCUTS.get(raw_label, raw_label)
        if label not in common.GOLD_LABELS:
            raise ValueError(f"{path}: invalid label {raw_label!r} for {pid}")
        if pid not in gold_all:
            raise ValueError(f"{path}: unknown pair_id {pid!r}")
        set_id = (row.get("set") or pid.split("-")[0]).strip()
        scored.append({
            "pair_id": pid,
            "set": set_id,
            "a": row.get("a", ""),
            "b": row.get("b", ""),
            "human": label,
            "gold": gold_all[pid],
            "confidence": (row.get("confidence") or "").strip(),
            "note": row.get("note", ""),
        })

    by_set = {s: _metrics_for_slice([r for r in scored if r["set"] == s]) for s in sets}
    disagreements = [r for r in scored if r["human"] != r["gold"]]
    return {
        "source_csv": str(path),
        "total_rows_in_sheet": len(raw_rows),
        "labelled_rows": len(scored),
        "pooled": _metrics_for_slice(scored),
        "by_set": by_set,
        "by_confidence": {c: _metrics_for_slice([r for r in scored if r["confidence"] == c])
                          for c in ("3", "2", "1") if any(r["confidence"] == c for r in scored)},
        "disagreements": disagreements,
        "human_labels": {r["pair_id"]: r["human"] for r in scored},
    }


def systems_on_labels(human: dict[str, str], resamples: int = study_stats.DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Exploratory (amendment A3): the five triage conditions on the human-labelled pairs, scored once against the
    silver gold and once against the human labels. Same predictions, same metrics as analyze.py; only the labels differ."""
    import analyze  # noqa: PLC0415  (heavy: loads the frozen pipeline)
    import cl_features  # noqa: PLC0415

    path = common.resolve_pipeline_path(None, common.TEST_SETS)
    rows, _, _ = analyze.load_triage(common.TEST_SETS, common.winner_variant(), cl_features.load_pipeline(path))
    rows = [r for r in rows if r["pair_id"] in human]
    preds = analyze.predictions(rows)
    out: dict[str, Any] = {"n": len(rows), "positive_class": "complement"}
    for name, labels in (("silver", [r["gold"] for r in rows]), ("human", [human[r["pair_id"]] for r in rows])):
        y = np.array([lab == "complement" for lab in labels], bool)
        boot = study_stats.pair_bootstrap(preds, y, resamples)
        conditions = {c: {**study_stats.metrics(preds[c], y), "f1_ci": list(study_stats.percentile_ci(boot[c]["f1"]))}
                      for c in analyze.CONDITIONS}
        out[name] = {"positives": int(y.sum()), "conditions": conditions,
                     "diffs": analyze._diff_block(boot, conditions)}
    return out


def format_systems_summary(systems: dict[str, Any]) -> str:
    lines = ["", f"## Triage conditions on these {systems['n']} pairs (exploratory)", "",
             "| Condition | P silver | R silver | F1 silver | P human | R human | F1 human |", "|---|--:|--:|--:|--:|--:|--:|"]
    for c, s in systems["silver"]["conditions"].items():
        h = systems["human"]["conditions"][c]
        lines.append(f"| {c} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} "
                     f"| {h['precision']:.3f} | {h['recall']:.3f} | {h['f1']:.3f} |")
    lines.append("")
    for name in ("silver", "human"):
        for key in ("CL-L:f1", "CL-L:precision"):
            d = systems[name]["diffs"][key]
            lines.append(f"- {name} {key}: {d['estimate']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    return "\n".join(lines)


def format_markdown_summary(report: dict[str, Any]) -> str:
    p = report["pooled"]
    lines = [
        "# Human Audit vs. Adjudicated Silver-Standard Gold",
        "",
        f"Labelled pairs: **{report['labelled_rows']} / {report['total_rows_in_sheet']}**",
        "",
        "| Slice | n | 3-class κ | Binary κ (complement) | 3-class agreement | Binary agreement |",
        "|---|--:|--:|--:|--:|--:|",
        f"| **Pooled** | {p['n']} | {p['kappa_3class']:.3f} | {p['kappa_binary']:.3f} "
        f"| {p['percent_agreement_3class']:.1f}% | {p['percent_agreement_binary']:.1f}% |",
    ]
    for set_id, m in report["by_set"].items():
        if m["n"] > 0:
            lines.append(
                f"| `{set_id}` | {m['n']} | {m['kappa_3class']:.3f} | {m['kappa_binary']:.3f} "
                f"| {m['percent_agreement_3class']:.1f}% | {m['percent_agreement_binary']:.1f}% |"
            )
    return "\n".join(lines)


def interactive_session(path: Path) -> int:
    if not path.exists():
        rows = sample_blind_rows()
        write_blind_sheet(path, rows)
        print(f"Created blind audit sheet at {path} ({len(rows)} pairs).")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pending = [i for i, r in enumerate(rows) if not (r.get("label") or "").strip()]
    print(f"{len(rows) - len(pending)}/{len(rows)} already labelled. Shortcuts: [1/c] complement, [2/s] substitute, [3/n] not_a_bundle, [q] quit.")
    for idx in pending:
        r = rows[idx]
        print(f"\n[{idx + 1}/{len(rows)}] ({r['set']}) {r['a']}  +  {r['b']}")
        while True:
            ans = input("Label (1=complement, 2=substitute, 3=not_a_bundle, q=quit): ").strip().lower()
            if ans == "q":
                write_blind_sheet(path, rows)
                print(f"Saved progress to {path}.")
                return 0
            if ans in SHORTCUTS:
                r["label"] = SHORTCUTS[ans]
                write_blind_sheet(path, rows)
                break
    report = score_human_sheet(path)
    print("\n" + format_markdown_summary(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", action="store_true", help="write a new blind audit CSV")
    parser.add_argument("--n-per-set", type=int, default=10, help="pairs per test set (default: 10)")
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--out", type=Path, default=default_blind_path(), help="blind CSV path")
    parser.add_argument("--interactive", action="store_true", help="label the blind CSV in the terminal")
    parser.add_argument("--score", type=Path, nargs="+", help="score one or more human CSVs against final gold")
    parser.add_argument("--batches", action="store_true",
                        help="write eval/human/batch_{1,2}.csv (50 pairs each) and the app eval/human/label_pairs.html")
    parser.add_argument("--report-out", type=Path, default=default_report_path(), help="JSON report output path")
    parser.add_argument("--resamples", type=int, default=study_stats.DEFAULT_RESAMPLES,
                        help="bootstrap resamples for the system comparison")
    parser.add_argument("--no-systems", action="store_true",
                        help="only agreement; skip scoring the triage conditions on the human labels")
    args = parser.parse_args(argv)

    if args.sample:
        rows = sample_blind_rows(n_per_set=args.n_per_set, seed=args.seed)
        write_blind_sheet(args.out, rows)
        print(f"Wrote {len(rows)} blind pairs to {args.out}")
        return 0
    if args.batches:
        batches = sample_batches(seed=args.seed)
        out = human_dir()
        for i, batch in enumerate(batches, start=1):
            write_batch_sheet(out / f"batch_{i}.csv", batch)
        write_label_app(out / "label_pairs.html", batches)
        print(f"Wrote {' + '.join(str(len(b)) for b in batches)} blind pairs to {out}/batch_*.csv and {out / 'label_pairs.html'}")
        return 0
    if args.interactive:
        return interactive_session(args.out)
    if args.score:
        report = score_human_sheets(args.score)
        if not args.no_systems:
            report["systems"] = systems_on_labels(report["human_labels"], args.resamples)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(format_markdown_summary(report))
        if "systems" in report:
            print(format_systems_summary(report["systems"]))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
