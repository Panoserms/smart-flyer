"""Amendment A1 of the pre-registration: drop t1 pairs whose stock codes match a dev pair.

Those pairs are the same products as a tuning pair under a re-worded description, so they are not new pairs.
Run after build_testsets.py (which writes all 132 pairs and data/testsets/t1_dev_audit.csv). Safe to re-run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def filter_csv(path: Path, drop: set[str]) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["pair_id"] not in drop]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    testsets = ROOT / "data" / "testsets"
    with (testsets / "t1_dev_audit.csv").open(encoding="utf-8", newline="") as handle:
        drop = {row["pair_id"] for row in csv.DictReader(handle) if row["same_stock_codes"] == "1"}
    kept = filter_csv(testsets / "pairs_t1.csv", drop)
    for who in ("A", "B"):
        assert filter_csv(ROOT / "eval" / "gold" / "blind" / f"{who}_t1.csv", drop) == kept
    manifest_path = testsets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sets"]["t1"]["removed_same_stock_codes_amendment_a1"] = len(drop)
    manifest["sets"]["t1"]["final_n"] = kept
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"t1: removed {len(drop)} pairs, {kept} remain")


if __name__ == "__main__":
    main()
