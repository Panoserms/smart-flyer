"""Hand-computed values for the statistics and the triage verdict parser."""
import numpy as np
import pytest

import run_triage
import study_stats as st


def test_metrics_hand_computed():
    pred = np.array([1, 1, 0, 0, 1], bool)
    gold = np.array([1, 0, 1, 0, 1], bool)
    m = st.metrics(pred, gold)
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["kept"]) == (2, 1, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_metrics_empty_denominators_are_zero():
    m = st.metrics(np.zeros(3, bool), np.zeros(3, bool))
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert m["accuracy"] == 1.0


def test_kappa_hand_computed():
    # observed 3/4, chance 0.5 * 0.25 + 0.5 * 0.75 = 0.5, kappa = 0.25 / 0.5
    assert st.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert st.cohen_kappa(list("abab"), list("abab")) == pytest.approx(1.0)
    assert st.cohen_kappa(list("abab"), list("baba")) == pytest.approx(-1.0)


def test_mcnemar_exact_hand_computed():
    assert st.mcnemar_exact(1, 5) == pytest.approx(14 / 64)      # 2 * (1 + 6) / 64
    assert st.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert st.mcnemar_exact(3, 3) == 1.0
    assert st.mcnemar_exact(0, 0) == 1.0


def test_holm_hand_computed():
    out = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert out["a"] == pytest.approx(0.04)     # 0.01 * 4
    assert out["c"] == pytest.approx(0.09)     # 0.03 * 3
    assert out["b"] == pytest.approx(0.09)     # max(0.04 * 2, 0.09)
    assert out["d"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_seed_dependent():
    gold = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], bool)
    preds = {"x": np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0], bool), "y": gold.copy()}
    a = st.pair_bootstrap(preds, gold, 300, seed=5)
    b = st.pair_bootstrap(preds, gold, 300, seed=5)
    c = st.pair_bootstrap(preds, gold, 300, seed=6)
    assert np.array_equal(a["x"]["f1"], b["x"]["f1"])
    assert not np.array_equal(a["x"]["f1"], c["x"]["f1"])
    assert np.all(a["y"]["f1"] == 1.0)           # a perfect predictor stays perfect under resampling
    low, high = st.percentile_ci(a["x"]["f1"])
    assert low <= float(st.metrics(preds["x"], gold)["f1"]) <= high


def test_bootstrap_is_paired_across_conditions():
    gold = np.array([1, 0] * 10, bool)
    preds = {"x": gold.copy(), "y": gold.copy()}
    boot = st.pair_bootstrap(preds, gold, 200, seed=1)
    assert np.array_equal(boot["x"]["f1"], boot["y"]["f1"])


def test_mean_bootstrap_and_p_value():
    values = {"a": np.ones(30), "b": np.zeros(30)}
    boot = st.mean_bootstrap(values, 400, seed=3)
    assert np.all(boot["a"] == 1.0) and np.all(boot["b"] == 0.0)
    assert st.bootstrap_p(boot["a"] - boot["b"]) == pytest.approx(2 / 401)


@pytest.mark.parametrize("variant,text,expected", [
    ("definitions", "COMPLEMENT", "COMPLEMENT"),
    ("definitions", "  Substitute.\n", "SUBSTITUTE"),
    ("definitions", "The answer is neither", "NEITHER"),
    ("definitions", "I am not sure.", "INVALID"),
    ("definitions", "", "INVALID"),
    ("examples", "NEITHER\n", "NEITHER"),
    ("examples", "complementary", "INVALID"),      # not the word, only a prefix
    ("reasoning", "They are used together. Answer: COMPLEMENT", "COMPLEMENT"),
    ("reasoning", "Not a substitute at all.\nAnswer: **NEITHER**", "NEITHER"),
    ("reasoning", "Answer: substitute\nAnswer: complement", "COMPLEMENT"),   # last marker counts
    ("reasoning", "They go together, so COMPLEMENT.", "INVALID"),              # no "Answer:" line
    ("reasoning", "Answer: maybe", "INVALID"),
])
def test_parse_verdict(variant, text, expected):
    assert run_triage.parse_verdict(text, variant) == expected


def test_human_audit_sample_and_score(tmp_path):
    import common
    import human_audit

    rows = human_audit.sample_blind_rows(n_per_set=3, seed=42)
    assert len(rows) == 9
    assert all(r["label"] == "" for r in rows)
    gold = {pid: label for s in common.TEST_SETS for pid, label in common.read_gold(s).items()}
    for i, r in enumerate(rows):
        # 8 of 9 match gold; 1 disagrees so kappa < 1.0 is exercised.
        r["label"] = gold[r["pair_id"]] if i > 0 else (
            "substitute" if gold[r["pair_id"]] == "complement" else "complement"
        )
    csv_path = tmp_path / "audit.csv"
    human_audit.write_blind_sheet(csv_path, rows)
    report = human_audit.score_human_sheet(csv_path)
    assert report["labelled_rows"] == 9
    assert report["pooled"]["n"] == 9
    assert len(report["disagreements"]) == 1
    assert 0.5 < report["pooled"]["kappa_3class"] < 1.0
    md = human_audit.format_markdown_summary(report)
    assert "Pooled" in md and "9 / 9" in md


# --- Human label batches (two batches of 50 + HTML app) ----------------------------------------

def _all_test_pairs():
    import common
    gold = {pid: label for s in common.TEST_SETS for pid, label in common.read_gold(s).items()}
    sets = {p["pair_id"]: s for s in common.TEST_SETS for p in common.read_pairs(s)}
    return gold, sets


def test_human_batches_are_two_disjoint_balanced_batches_of_50():
    import human_audit
    gold, sets = _all_test_pairs()
    batches = human_audit.sample_batches()
    assert [len(b) for b in batches] == [50, 50]
    ids = [r["pair_id"] for b in batches for r in b]
    assert len(set(ids)) == 100
    assert all(pid in gold for pid in ids)
    assert all(r["label"] == "" for b in batches for r in b)
    # Stratified: each set and each gold label is split evenly (within one) between the batches.
    for key in ("set", "gold"):
        def count(batch, value):
            return sum((sets if key == "set" else gold)[r["pair_id"]] == value for r in batch)
        values = set(sets.values()) if key == "set" else set(gold.values())
        for v in values:
            assert abs(count(batches[0], v) - count(batches[1], v)) <= 1, (key, v)
            assert count(batches[0], v) + count(batches[1], v) >= 10, (key, v)


def test_human_batches_are_deterministic_and_seeded():
    import human_audit
    first = [[r["pair_id"] for r in b] for b in human_audit.sample_batches()]
    again = [[r["pair_id"] for r in b] for b in human_audit.sample_batches()]
    other = [[r["pair_id"] for r in b] for b in human_audit.sample_batches(seed=1)]
    assert first == again
    assert first != other


def test_label_app_contains_every_pair_and_no_gold_labels(tmp_path):
    import json
    import re
    import human_audit
    gold, _ = _all_test_pairs()
    batches = human_audit.sample_batches()
    out = tmp_path / "label_pairs.html"
    human_audit.write_label_app(out, batches)
    html = out.read_text(encoding="utf-8")
    assert "__DATA__" not in html
    m = re.search(r"const BATCHES = (\[.*?\]);\n", html, re.S)
    assert m, "embedded batch data not found"
    data = json.loads(m.group(1))
    assert [len(b) for b in data] == [50, 50]
    for batch in data:
        for row in batch:
            assert set(row) == {"pair_id", "set", "a", "b"}  # nothing that could leak the gold label
    assert {r["pair_id"] for b in data for r in b} == {r["pair_id"] for b in batches for r in b}
    # The app exports exactly the columns score_human_sheet reads.
    assert "pair_id,set,a,b,label,confidence,note" in html
    assert "complement" in html and "substitute" in html and "not_a_bundle" in html
    assert all(label not in ("", None) for label in gold.values())


def test_score_accepts_both_batch_files(tmp_path):
    import human_audit
    gold, _ = _all_test_pairs()
    paths = []
    for i, batch in enumerate(human_audit.sample_batches(), start=1):
        for r in batch:
            r["label"] = gold[r["pair_id"]]
            r["confidence"] = "3"
        p = tmp_path / f"human_labels_batch_{i}.csv"
        human_audit.write_batch_sheet(p, batch)
        paths.append(p)
    report = human_audit.score_human_sheets(paths)
    assert report["labelled_rows"] == 100
    assert report["pooled"]["kappa_3class"] == 1.0
    one = human_audit.score_human_sheets(paths[:1])
    assert one["labelled_rows"] == 50


def test_systems_scored_on_human_labels_match_silver_when_labels_agree():
    import human_audit
    gold, _ = _all_test_pairs()
    ids = [r["pair_id"] for b in human_audit.sample_batches() for r in b]
    out = human_audit.systems_on_labels({pid: gold[pid] for pid in ids}, resamples=200)
    assert out["n"] == 100
    assert set(out["silver"]["conditions"]) == {"L", "C", "CL", "Lf", "CLf"}
    for cond in ("L", "C", "CL", "Lf", "CLf"):
        assert out["human"]["conditions"][cond]["f1"] == out["silver"]["conditions"][cond]["f1"]
    diff = out["human"]["diffs"]["CL-L:f1"]
    assert diff["ci"][0] <= diff["estimate"] <= diff["ci"][1]


def test_systems_scored_on_human_labels_follow_the_human_labels():
    import human_audit
    gold, _ = _all_test_pairs()
    ids = [r["pair_id"] for b in human_audit.sample_batches() for r in b]
    out = human_audit.systems_on_labels({pid: "complement" for pid in ids}, resamples=200)
    # Every pair is a complement for the human, so whatever a system keeps is right.
    for cond in ("L", "C", "CL", "Lf", "CLf"):
        m = out["human"]["conditions"][cond]
        assert m["precision"] == 1.0 or m["kept"] == 0
    assert out["human"]["positives"] == 100 and out["silver"]["positives"] < 100


def test_score_command_line_adds_system_comparison(tmp_path):
    import json

    import human_audit
    gold, _ = _all_test_pairs()
    paths = []
    for i, batch in enumerate(human_audit.sample_batches(), start=1):
        for r in batch:
            r["label"], r["confidence"] = gold[r["pair_id"]], "3"
        p = tmp_path / f"human_labels_batch_{i}.csv"
        human_audit.write_batch_sheet(p, batch)
        paths.append(p)
    out = tmp_path / "report.json"
    assert human_audit.main(["--score", *map(str, paths), "--report-out", str(out), "--resamples", "200"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["systems"]["n"] == 100
    assert report["systems"]["human"]["conditions"]["CL"]["f1"] == report["systems"]["silver"]["conditions"]["CL"]["f1"]


def test_report_gets_human_check_section_from_the_audit_report(tmp_path):
    import json

    import analyze
    import human_audit
    gold, _ = _all_test_pairs()
    paths = []
    for i, batch in enumerate(human_audit.sample_batches(), start=1):
        for r in batch:
            r["label"], r["confidence"] = gold[r["pair_id"]], "3"
        p = tmp_path / f"human_labels_batch_{i}.csv"
        human_audit.write_batch_sheet(p, batch)
        paths.append(p)
    rep = tmp_path / "human_labels_report.json"
    human_audit.main(["--score", *map(str, paths), "--report-out", str(rep), "--resamples", "200"])
    hc = analyze.load_human_check(rep)
    assert hc["n"] == 100 and hc["pooled"]["kappa_3class"] == 1.0
    assert set(hc["by_set"]) == {"t1", "t2", "t3"}
    assert "human_labels" not in hc and "disagreements" not in hc and hc["n_disagreements"] == 0
    text = "\n".join(analyze.render_human_check(hc))
    assert "## Human check of the labels (amendment A3" in text and "1.000" in text
    assert analyze.load_human_check(tmp_path / "missing.json") is None


def test_human_audit_command_line_writes_batches_and_app(tmp_path, monkeypatch):
    import human_audit
    monkeypatch.setattr(human_audit, "human_dir", lambda: tmp_path)
    assert human_audit.main(["--batches"]) == 0
    assert (tmp_path / "batch_1.csv").exists() and (tmp_path / "batch_2.csv").exists()
    assert (tmp_path / "label_pairs.html").exists()
