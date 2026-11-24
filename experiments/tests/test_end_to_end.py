"""analyze.py and export_site_data.py on a small invented study in a tmp dir."""
import json
from pathlib import Path

import numpy as np
import pytest

import analyze
import common
import export_site_data as export
import synthetic

WORKING = Path(__file__).resolve().parents[2] / "pipeline" / "export_smart_flyer.py"
RESAMPLES = 300


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    real_root = common.ROOT
    root = tmp_path_factory.mktemp("study")
    synthetic.build(root, WORKING)
    summary, internal = analyze.build(resamples=RESAMPLES)
    yield root, summary, internal
    common.ROOT = real_root


def test_triage_counts_and_recomputed_metrics(study):
    _, summary, internal = study
    rows = internal["rows"]
    assert len(rows) == 36
    for group, g in summary["triage"].items():
        members = [r for r in rows if group == "pooled" or r["set"] == group]
        assert g["n"] == len(members)
        for cond, m in g["conditions"].items():
            assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(members)
    pooled = summary["triage"]["pooled"]["conditions"]
    # C is rules only; a cascade pays only for the pairs the rules let through
    assert pooled["C"]["llm_calls"] == 0 and pooled["L"]["llm_calls"] == 36
    survivors = sum(not r["reject"] for r in rows)
    assert pooled["CL"]["llm_calls"] == survivors < 36
    assert pooled["CL"]["llm_tokens"] == survivors * 303
    assert pooled["L"]["llm_seconds"] == pytest.approx(36 * 0.9)
    # hand check of CL against the raw rows
    gold = np.array([r["gold"] == "complement" for r in rows])
    cl = np.array([not r["reject"] and r["plain"]["verdict"] == "COMPLEMENT" for r in rows])
    assert pooled["CL"]["tp"] == int((cl & gold).sum()) and pooled["CL"]["fp"] == int((cl & ~gold).sum())


def test_hypotheses_and_holm(study):
    _, summary, _ = study
    hyp = {h["id"]: h for h in summary["hypotheses"]}
    assert list(hyp) == [f"H{i}" for i in range(1, 8)]
    assert all("NOT EVALUATED" not in h["note"] for h in hyp.values())
    from study_stats import holm
    raw = {k: hyp[k]["p_raw"] for k in ("H2", "H3", "H4", "H5")}
    corrected = holm(raw)
    for k in raw:
        assert hyp[k]["p_holm"] == pytest.approx(corrected[k]) and hyp[k]["p_holm"] >= hyp[k]["p_raw"]
    assert hyp["H1"]["p_raw"] is None and hyp["H6"]["p_holm"] is None
    assert "item_level" in summary["triage"]["pooled"]


def test_copy_summary(study):
    _, summary, internal = study
    cp = summary["copy"]
    assert cp["n_bundles"] == 6
    assert cp["n_empty_not_rated"] == sum(c["source"] == "none" for c in internal["copy_cells"].values())
    raw = cp["summary"]["pooled"]["raw"]
    assert raw["n_texts"] == 18 and raw["calls"] == pytest.approx(6.0)     # one 3-language call per bundle
    assert raw["retries_total"] == 0
    guarded = cp["summary"]["pooled"]["guarded"]
    assert guarded["calls"] >= 6.0                                           # shared first call plus repairs
    cells = [c for (b, cond, lg), c in internal["copy_cells"].items() if cond == "raw"]
    assert raw["publishable_share"] == pytest.approx(sum(c["publishable_both"] for c in cells) / len(cells))
    assert all(not c["publishable_both"] for c in cells if c["source"] == "none")
    assert -1 <= cp["rater_kappa"]["publishable"]["kappa"] <= 1


def test_main_writes_files_and_triage_only_needs_no_ratings(study, tmp_path):
    root, _, _ = study
    (root / "eval" / "gold" / "blind_rating" / "R1_scored.csv").rename(root / "R1_hidden.csv")
    out = tmp_path / "out"
    summary, _ = analyze.build(triage_only=True, resamples=100)
    assert "copy" not in summary and any("Task B skipped" in m for m in summary["messages"])
    summary, _ = analyze.build(resamples=100)
    assert "copy" not in summary and any("Task B not evaluated" in m for m in summary["messages"])
    report = analyze.render_report(summary)
    assert "H1" in report and "not evaluated" in report
    (root / "R1_hidden.csv").rename(root / "eval" / "gold" / "blind_rating" / "R1_scored.csv")
    import sys
    sys.argv = ["analyze.py", "--out-dir", str(out), "--resamples", "100", "--root", str(root)]
    assert analyze.main() == 0
    assert json.loads((out / "summary.json").read_text())["hypotheses"][0]["id"] == "H1"
    text = (out / "REPORT.md").read_text()
    assert "supported" in text and "Task B: flyer copy" in text


def test_export_validates_and_matches_schema(study):
    _, summary, internal = study
    data = export.build_export(summary, internal)
    export.validate(data)
    assert data["fixture"] is False and len(data["pairs"]) == 36
    assert [d["id"] for d in data["datasets"]] == ["t1", "t2", "t3"] and data["datasets"][2]["n_transactions"] is None
    json.dumps(data)
    # a corrupted summary must be caught
    data["triage"]["pooled"]["L"]["tp"] += 1
    with pytest.raises(export.ExportError):
        export.validate(data)


def test_export_rejects_bad_enum_and_missing_key(study):
    _, summary, internal = study
    data = export.build_export(summary, internal)
    data["pairs"][0]["llm"]["plain"] = "MAYBE"
    with pytest.raises(export.ExportError):
        export.validate(data)
    data = export.build_export(summary, internal)
    del data["copy"]["summary"]["pooled"]["raw"]["calls"]
    with pytest.raises(export.ExportError):
        export.validate(data)
    data = export.build_export(summary, internal)
    data["copy"]["bundles"][0]["conditions"]["raw"]["de"]["failed_checks"] = ["spelling"]
    with pytest.raises(export.ExportError):
        export.validate(data)


def test_export_main_writes_file(study, tmp_path):
    import sys
    root, _, _ = study
    out = tmp_path / "site" / "smart_flyer_study.json"
    sys.argv = ["export", str(out), "--root", str(root), "--resamples", "100"]
    assert export.main() == 0
    assert json.loads(out.read_text())["fixture"] is False
