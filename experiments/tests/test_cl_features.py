"""cl_features' cl_reject must equal the pipeline's own run_pruner decision (stop or shared)."""
import pytest

import cl_features
import common


@pytest.mark.parametrize("set_id", ["dev", "dev_de"])
def test_cl_reject_equals_run_pruner(set_id):
    pipeline = cl_features.load_pipeline(common.resolve_pipeline_path(None, [set_id]))
    pairs = common.read_pairs(set_id)
    catalog = common.read_catalog(set_id)
    mine = cl_features.analyse_pairs(pairs, catalog, pipeline)

    learned = pipeline.learn_variant_attributes(catalog) if catalog else frozenset()
    langs = {p["lang_a"] for p in pairs} | {p["lang_b"] for p in pairs}
    extractor = pipeline.HeadNounExtractor(learned)
    extractor.load(langs)
    cands = [pipeline.Candidate(p["a"], p["b"], p["price_a"] or 0.0, p["price_b"] or 0.0, p["support"] or 0.0,
                                p["confidence"] or 0.0, p["lift"] or 0.0, "uci_retail", p["lang_a"], p["lang_b"])
             for p in pairs]
    records, _ = pipeline.run_pruner(cands, extractor, 0.0, bag_guard=True)
    expected = {p["pair_id"]: r["status"] in (pipeline.STATUS_PACKAGING, pipeline.STATUS_HEAD)
                for p, r in zip(pairs, records)}
    wrong = [pid for pid in expected if mine[pid]["cl_reject"] != expected[pid]]
    assert not wrong, f"{len(wrong)} of {len(pairs)} pairs differ, first {wrong[:5]}"
    assert any(expected.values()) and not all(expected.values())
