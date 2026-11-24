"""Tests for pipeline/export_smart_flyer.py (run: .venv/bin/python -m pytest -q)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import copy_checks as checks  # noqa: E402
import export_smart_flyer as sf  # noqa: E402

CONTRACT_RULE_KEYS = {"id", "domain", "antecedent", "consequent", "metrics",
                      "linguistic_checks", "promo_mechanics", "flyer_copy"}
CONTRACT_TELEMETRY_KEYS = {"transactions_analyzed", "raw_association_rules", "rejected_packaging_noise",
                           "rejected_same_head_noun", "rejected_low_margin", "rejected_llm_judge",
                           "surviving_complementary_bundles", "llm_calls_saved_pct", "cpu_pruning_time_ms"}


@pytest.fixture(scope="module")
def extractor() -> sf.HeadNounExtractor:
    ex = sf.HeadNounExtractor()
    ex.load({"en", "de"})
    return ex


# --- Filter 1: regex ---------------------------------------------------------

@pytest.mark.parametrize("name", ["POSTAGE", "DOTCOM POSTAGE", "Plastic Carrier Bag", "Pfand Einweg 0,25 €",
                                  "Manual", "Adjust bad debt", "Gutschein 10 €", "Gift Card 25"])
def test_regex_hits_operational_items(name):
    assert sf.stop_item_match(name)


def test_spec_regex_misses_plurals():
    # Known gap in the spec regex: \bbag\b does not match "bags".
    assert sf.stop_item_match("PAPER BAGS") is None
    assert sf.stop_item_match("CARRIER BAGS") == "CARRIER"  # caught via "carrier"


@pytest.mark.parametrize("name", ["Bananen", "Pfandflasche-Rückgabe", "Baguette", "WHITE HANGING HEART T-LIGHT HOLDER"])
def test_regex_word_boundaries(name):
    # \b keeps "Baguette" and compounds like "Pfandflasche" from matching.
    assert sf.stop_item_match(name) is None


@pytest.mark.parametrize("name", ["JUMBO BAG RED RETROSPOT", "LUNCH BAG SUKI DESIGN", "WOODLAND CHARLOTTE BAG"])
def test_bag_guard_keeps_sold_bags(name):
    assert sf.stop_item_match(name) is None
    assert sf.stop_item_match(name, bag_guard=False) == "BAG"  # the spec regex alone removes them


@pytest.mark.parametrize("name", ["Plastic Carrier Bag", "PAPER BAG", "Bag", "Shopping bag"])
def test_bag_guard_still_removes_packaging_bags(name):
    assert sf.stop_item_match(name)


# --- Filter 2: head nouns ----------------------------------------------------

def test_all_caps_is_title_cased_before_parsing():
    assert sf.normalise_for_nlp("WHITE HANGING HEART T-LIGHT HOLDER") == "Hanging Heart T-light Holder"
    assert sf.smart_title("CHILDREN'S CUP") == "Children's Cup"


def test_units_counts_and_variants_are_stripped():
    assert sf.normalise_for_nlp("PACK OF 72 RETROSPOT CAKE CASES") == "Cake Cases"
    assert sf.normalise_for_nlp("Bio Vollmilch 3,8% 1L") == "Vollmilch"
    assert sf.normalise_for_nlp("REGENCY CAKESTAND 3 TIER") == "Cakestand"
    assert sf.normalise_for_nlp("Spaghetti No. 5 500g") == "Spaghetti"


@pytest.mark.parametrize("name,lang,head", [
    ("WHITE HANGING HEART T-LIGHT HOLDER", "en", "holder"),
    ("ALARM CLOCK BAKELIKE GREEN", "en", "clock"),
    ("ROSES REGENCY TEACUP AND SAUCER", "en", "teacup"),
    ("PACK OF 72 RETROSPOT CAKE CASES", "en", "case"),
    ("LUNCH BOX WITH CUTLERY RETROSPOT", "en", "box"),
    ("Bio Espressobohnen Ganze Bohne 1kg", "de", "espressobohne"),
    ("Barista Haferdrink Ungesüßt 1L", "de", "haferdrink"),
    ("Bio Fettarme Milch 1,5% 1L", "de", "milch"),
    ("Grillkohle Buchenholz 3kg", "de", "grillkohle"),
    ("Thüringer Rostbratwurst 6 Stück", "de", "rostbratwurst"),
    ("Cherrytomaten 250g", "de", "cherrytomate"),
])
def test_head_noun(extractor, name, lang, head):
    assert extractor.get(name, lang).head_noun == head


def test_head_noun_capitalised_participle(extractor):
    assert extractor.get("Passierte Tomaten 500g", "de").head_noun == "tomate"
    assert extractor.get("Gehackte Mandeln 200g", "de").head_noun == "mandel"


@pytest.mark.parametrize("a,b,german,expected", [
    ("holder", "holder", False, True),
    ("vollmilch", "milch", True, True),
    ("vollmilch", "hafermilch", True, True),        # 5-char compound suffix
    ("erdbeerjoghurt", "vanillejoghurt", True, True),
    ("apfelsaft", "orangensaft", True, True),        # short base from the lexicon
    ("holder", "folder", False, False),              # 5-char rule not applied to English
    ("espressobohne", "haferdrink", True, False),
    ("grillkohle", "rostbratwurst", True, False),
])
def test_head_collision(a, b, german, expected):
    assert sf.head_collision(a, b, german) is expected


def test_short_compound_bases_mus_eis_salz():
    assert sf.head_collision("apfelmus", "pflaumenmus", True) is True
    assert sf.head_collision("vanilleeis", "schokoeis", True) is True
    assert sf.head_collision("meersalz", "jodsalz", True) is True


def test_learn_variant_attributes_from_minimal_pairs():
    catalog = ["LUNCH BAG ALPHABET DESIGN", "LUNCH BAG APPLE DESIGN", "JUMBO BAG ALPHABET DESIGN",
               "JUMBO BAG APPLE DESIGN", "PINK CRYSTAL NECKLACE", "PINK CRYSTAL BRACELET",
               "BLUE CRYSTAL NECKLACE", "BLUE CRYSTAL BRACELET"]
    learned = sf.learn_variant_attributes(catalog)
    assert {"alphabet", "apple"} <= learned
    assert not {"necklace", "bracelet", "bag"} & learned   # head slot: product types, kept


@pytest.mark.parametrize("a,b,expected", [
    ("HAND WARMER BIRD DESIGN", "HAND WARMER OWL DESIGN", "hand warmer"),
    ("COOK WITH WINE METAL SIGN", "GIN + TONIC DIET METAL SIGN", "metal sign"),
    ("CHARLOTTE BAG DOLLY GIRL DESIGN", "RED RETROSPOT CHARLOTTE BAG", "charlotte bag"),
    ("RED RETROSPOT CHARLOTTE BAG", "RED RETROSPOT CAKE CASES", None),   # attribute compound
    ("REGENCY CAKESTAND 3 TIER", "ROSES REGENCY TEACUP AND SAUCER", None),
    ("JUMBO BAG PEARS", "JUMBO BAG VINTAGE LEAF", "jumbo bag"),
    ("CHOCOLATE HOT WATER BOTTLE", "HOT WATER BOTTLE KEEP CALM", "hot water bottle"),
    ("SET/6 RED SPOTTY PAPER PLATES", "SET/6 RED SPOTTY PAPER CUPS", None),  # complements
])
def test_open_compound_match(a, b, expected):
    assert sf.open_compound_match(a, b) == expected


def test_dolly_girl_normalises_to_product_head(extractor):
    assert sf.normalise_for_nlp("CHARLOTTE BAG DOLLY GIRL DESIGN") == "Charlotte Bag"
    assert extractor.get("CHARLOTTE BAG DOLLY GIRL DESIGN", "en").head_noun == "bag"


# --- Filter 3: promo maths + gate ----------------------------------------------

def test_promo_math_matches_contract_example():
    m = sf.promo_mechanics("Bio Espressobohnen Ganze Bohne 1kg", 12.99, "Barista Haferdrink Ungesüßt 1L", 2.19)
    assert m["anchor_item"].startswith("Bio Espresso")
    assert (m["regular_bundle_price"], m["promo_bundle_price"], m["net_basket_lift_eur"]) == (15.18, 14.74, 1.75)
    assert sf.margin_gate(m, lift=4.15, min_lift=2.0) == (True, None)


def test_anchor_is_higher_priced_item_regardless_of_rule_direction():
    m = sf.promo_mechanics("Mozzarella", 0.99, "Cherrytomaten", 1.49)
    assert m["anchor_item"] == "Cherrytomaten" and m["discounted_item"] == "Mozzarella"


def test_gate_rejects_tiny_pull_through_item():
    m = sf.promo_mechanics("Cake tins", 5.74, "Cake cases", 0.64)
    ok, reason = sf.margin_gate(m, lift=3.2, min_lift=2.0)
    assert not ok and "basket expansion" in reason


def test_gate_rejects_low_lift():
    m = sf.promo_mechanics("Butter", 2.29, "Toastbrot", 1.19)
    ok, reason = sf.margin_gate(m, lift=1.6, min_lift=2.0)
    assert not ok and "lift" in reason.lower()


# --- Copy ------------------------------------------------------------------------

def test_copy_checks_on_display_names():
    names = ("Set/6 Red Spotty Paper Plates", "Set/6 Red Spotty Paper Cups")
    ok = "Buy Set/6 Red Spotty Paper Plates and get 20% off Set/6 Red Spotty Paper Cups!"
    assert not checks.failed_ids(checks.run_checks("en", "Party time", ok, names, 20))
    renamed = "Buy the paper plates and get 20% off the cups!"
    assert "names" in checks.failed_ids(checks.run_checks("en", "Party time", renamed, names, 20))
    assert "number" in checks.failed_ids(checks.run_checks("en", "Party", ok.replace("20", "30"), names, 20))


def test_copy_checks_detect_reversed_offer_direction():
    names = ("Regency Cakestand 3 Tier", "Roses Regency Teacup And Saucer")
    swapped = "Buy the Roses Regency Teacup and Saucer and get 20% off the Regency Cakestand 3 Tier."
    res = checks.run_checks("en", "Bundle and Save Big", swapped, names, 20)
    assert "names" in checks.failed_ids(res) and res["names"].get("reversed_offer") is True
    # Valid pull-first phrasing with a conditional buy clause on the anchor must still pass.
    pull_first_ok = "Get 20% off the Roses Regency Teacup And Saucer when you buy the Regency Cakestand 3 Tier."
    assert not checks.failed_ids(checks.run_checks("en", "Bundle and Save Big", pull_first_ok, names, 20))
    prompt = sf.repair_prompt("en", ("Bundle", swapped), res, names, 20)
    assert "You swapped the two products" in prompt


OFFER = ("Grillbriketts 2,5kg", "Grillanzünder 32 Stück")  # anchor at full price, pull item at 20 % off


@pytest.mark.parametrize("copy", [
    "Buy Grillbriketts 2,5kg and get 20% off Grillanzünder 32 Stück!",
    "Get 20% off Grillanzünder 32 Stück when you buy Grillbriketts 2,5kg!",
    "Get Grillanzünder 32 Stück with 20% off when you buy Grillbriketts 2,5kg.",
    "Kaufe Grillbriketts 2,5kg und erhalte Grillanzünder 32 Stück mit 20% Rabatt!",
    # "hol dir" means "get yourself": it names what you take, not what you pay full price for
    "Hol dir Grillanzünder 32 Stück mit 20% Rabatt beim Kauf von Grillbriketts 2,5kg!",
    "Hol dir Grillanzünder 32 Stück mit 20% Rabatt, wenn du Grillbriketts 2,5kg kaufst.",
])
def test_offer_direction_accepts_correct_copy(copy):
    assert checks._offer_reversed(copy, OFFER) is False


@pytest.mark.parametrize("copy", [
    "Buy Grillanzünder 32 Stück and get 20% off Grillbriketts 2,5kg!",
    "Get Grillbriketts 2,5kg with 20% off when you buy Grillanzünder 32 Stück.",
    "Grillbriketts 2,5kg 20% off when you buy Grillanzünder 32 Stück!",
    "Kaufe Grillanzünder 32 Stück und erhalte Grillbriketts 2,5kg mit 20% Rabatt!",
    "Hol dir Grillbriketts 2,5kg mit 20% Rabatt beim Kauf von Grillanzünder 32 Stück!",
    "Hol dir Grillbriketts 2,5kg mit 20% Rabatt, wenn du Grillanzünder 32 Stück kaufst.",
])
def test_offer_direction_catches_reversed_copy(copy):
    assert checks._offer_reversed(copy, OFFER) is True


MUG = ("Red Mug", "Small Plate")  # final review 2026-11-12: correct copy that the first version flagged


@pytest.mark.parametrize("copy", [
    "Love your Red Mug? Buy Red Mug and get 20% off Small Plate!",            # the anchor named twice
    "Your Red Mug purchase gets you 20% off Small Plate.",                    # the buy noun follows its name
    "Buy Small Plate at 20% off with any Red Mug.",                           # the buy cue names the discounted item
    "Kauf Small Plate 20 % günstiger zu jeder Red Mug.",
    "Αγοράστε Small Plate με 20% έκπτωση μαζί με Red Mug.",
    "Beim Red Mug-Kauf gibt es Small Plate 20 % günstiger.",
])
def test_offer_direction_accepts_more_correct_phrasings(copy):
    assert checks._offer_reversed(copy, MUG) is False


@pytest.mark.parametrize("copy", [
    "Wer Small Plate kauft, spart 20 % auf Red Mug.",
    "Get 20% off Red Mug with every Small Plate purchase.",
    "Buying Small Plate? Save 20% on Red Mug.",
    "ΑΓΟΡΑΣΤΕ Small Plate και πάρτε 20% έκπτωση στο Red Mug.",                # all caps Greek, no accents
    "Αγοράζοντας Small Plate κερδίζεις 20% έκπτωση στο Red Mug.",
])
def test_offer_direction_catches_more_reversed_phrasings(copy):
    assert checks._offer_reversed(copy, MUG) is True


def test_copy_checks_reject_german_full_price_calque():
    names = ("Poppy's Playhouse Livingroom", "Poppy's Playhouse Kitchen")
    calque = "Kaufe das volle Preis-Produkt Poppy's Playhouse Livingroom und erhalte Poppy's Playhouse Kitchen mit 20% Rabatt!"
    assert "claim" in checks.failed_ids(checks.run_checks("de", "Super Rabatt", calque, names, 20))


def test_exported_public_json_copy_passes_all_checks():
    root = Path(__file__).resolve().parent.parent / "public"
    for filename in ("smart_flyer_sample.json", "smart_flyer_data.json"):
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        for rule in payload["rules"]:
            fc = rule["flyer_copy"]
            if fc is None:
                continue
            pm = rule["promo_mechanics"]
            names = (sf.display_name(pm["anchor_item"]), sf.display_name(pm["discounted_item"]))
            for lang in ("de", "en", "el"):
                entry = fc[lang]
                res = checks.run_checks(lang, entry["title"], entry["Copy"], names, pm["discount_pct"])
                assert not checks.failed_ids(res), f"{filename} {rule['id']} {lang}: {res}"


def test_template_copy_is_trilingual():
    c = sf.template_copy("REGENCY CAKESTAND 3 TIER", "ROSES REGENCY TEACUP AND SAUCER", 20, "k")
    assert set(c) == {"de", "en", "el"} and all(set(v) == {"title", "Copy"} for v in c.values())
    assert "Regency Cakestand 3 Tier" in c["en"]["Copy"]


def test_ollama_down_falls_back_to_templates():
    w = sf.GemmaCopywriter("http://127.0.0.1:9", None, enabled=True, probe_timeout=0.5)
    assert w.model is None
    assert w.write("A", "B", 20, "k")["en"]["Copy"] == "Buy A and get 20% off B."


# --- End to end (built-in sample, no LLM) -------------------------------------------

def test_builtin_end_to_end(tmp_path):
    out = tmp_path / "out.json"
    assert sf.main(["--no-llm", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    t = data["telemetry"]
    assert set(t) == CONTRACT_TELEMETRY_KEYS
    assert t["raw_association_rules"] == len(sf.BUILTIN_RULES) == len(data["rules"]) == 16
    assert t["rejected_llm_judge"] == 0   # --no-llm: judge stage skipped
    assert (t["rejected_packaging_noise"] + t["rejected_same_head_noun"] + t["rejected_low_margin"]
            + t["rejected_llm_judge"] + t["surviving_complementary_bundles"]) == t["raw_association_rules"]
    statuses = {}
    for rule in data["rules"]:
        assert set(rule) == CONTRACT_RULE_KEYS
        checks = rule["linguistic_checks"]
        statuses[(rule["antecedent"]["name"], rule["consequent"]["name"])] = checks["status"]
        approved = checks["status"] == "approved_bundle"
        assert (rule["flyer_copy"] is not None) == approved
        assert (checks["rejection_explanation"] is None) == approved
        if approved:
            assert set(rule["flyer_copy"]) == {"badge_text", "de", "en", "el"}
    assert statuses[("ALARM CLOCK BAKELIKE GREEN", "ALARM CLOCK BAKELIKE RED")] == "rejected_same_head_noun"
    assert statuses[("Bio Vollmilch 3,8% 1L", "Bio Fettarme Milch 1,5% 1L")] == "rejected_same_head_noun"
    assert statuses[("Plastic Carrier Bag", "Bio Bananen 1kg")] == "rejected_packaging_noise"
    assert statuses[("Bio Espressobohnen Ganze Bohne 1kg", "Barista Haferdrink Ungesüßt 1L")] == "approved_bundle"


# --- Tuning round on dev_de (2026-11-06) -------------------------------------------

def test_german_stop_word_inside_compound():
    assert sf.stop_item_match("Flaschenpfand 0,25 €") == "pfand"
    assert sf.stop_item_match("Pfandflasche-Rückgabe") is None  # the stop word must end the compound


@pytest.mark.parametrize("a,b,expected", [
    ("Hofgut Gouda jung 400g", "Alpenkrone Gouda mittelalt Scheiben 250g", "gouda"),
    ("Energy Drink Zero 0,5L", "Energy Drink Classic 0,25L", "energy drink"),
    ("Salami Scheiben 100g", "Gouda Scheiben 100g", None),   # pack word
    ("Spaghetti No. 5 500g", "Tomatensauce Basilikum 400g", None),
])
def test_shared_name_words(a, b, expected):
    assert sf.shared_name_words(a, b) == expected


def test_german_lemma_keeps_final_e_of_unknown_compounds():
    assert sf.german_lemma("Schuhcreme", "Schuhcrem") == "schuhcreme"
    assert sf.german_lemma("Tomaten", "Tomate") == "tomate"


# --- LLM judge stage -----------------------------------------------------------------

def _records(extractor, rules):
    recs, _ = sf.run_pruner([sf.Candidate(*r) for r in rules], extractor, 2.0)
    return recs


GOOD = ("Grillkohle Buchenholz 3kg", "Thüringer Rostbratwurst 6 Stück", 5.99, 3.49, 0.02, 0.5, 6.8,
        "supermarket_grocery", "de", "de")
GOOD2 = ("Mozzarella 125g", "Cherrytomaten 250g", 0.99, 1.49, 0.02, 0.4, 3.4, "supermarket_grocery", "de", "de")
SUBST = ("Erdbeerjoghurt 500g", "Vanillejoghurt 500g", 0.89, 0.89, 0.02, 0.4, 4.6, "supermarket_grocery", "de", "de")
POST = ("POSTAGE", "ROUND SNACK BOXES SET OF4 WOODLAND", 20.88, 3.42, .02, .3, 5.1, "uci_retail")


def test_judge_only_sees_survivors(extractor, monkeypatch):
    recs = _records(extractor, [GOOD, SUBST, POST])
    prompts = []
    monkeypatch.setattr(sf, "ollama_generate", lambda *a, **k: prompts.append(a[2]) or "COMPLEMENT")
    stats = sf.judge_survivors(recs, "http://x", "m", 5)
    assert stats["calls"] == 1 == len(prompts)
    assert "Grillkohle" in prompts[0] and "Erdbeerjoghurt" not in prompts[0] and "POSTAGE" not in prompts[0]
    assert "A parser reports" in prompts[0] and "Same head noun: no" in prompts[0]
    assert [r["status"] for r in recs] == [sf.STATUS_APPROVED, sf.STATUS_HEAD, sf.STATUS_PACKAGING]


@pytest.mark.parametrize("answer,verdict", [("COMPLEMENT", "COMPLEMENT"), ("  substitute.", "SUBSTITUTE"),
                                            ("NEITHER", "NEITHER"), ("I think they go together", "INVALID"),
                                            ("", "INVALID")])
def test_judge_verdict_parsing(answer, verdict):
    assert sf.parse_judge_verdict(answer) == verdict


@pytest.mark.parametrize("answer,expected,text", [
    ("COMPLEMENT", sf.STATUS_APPROVED, None),
    ("SUBSTITUTE", sf.STATUS_JUDGE, "LLM judge: SUBSTITUTE"),
    ("NEITHER", sf.STATUS_JUDGE, "LLM judge: NEITHER"),
    ("hmm", sf.STATUS_JUDGE, "LLM judge: INVALID")])
def test_only_complement_stays_approved(extractor, monkeypatch, answer, expected, text):
    recs = _records(extractor, [GOOD])
    monkeypatch.setattr(sf, "ollama_generate", lambda *a, **k: answer)
    sf.judge_survivors(recs, "http://x", "m", 5)
    assert recs[0]["status"] == expected and recs[0]["explanation"] == text


def test_judge_stops_when_ollama_dies(extractor, monkeypatch):
    import requests
    recs = _records(extractor, [GOOD, GOOD2, GOOD, GOOD2, GOOD])
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(sf, "ollama_generate", boom)
    stats = sf.judge_survivors(recs, "http://x", "m", 5)
    # A failed call says nothing about the pair: every pair keeps the rules' decision.
    assert stats["calls"] == 3 and stats["unjudged"] == 5 and stats["rejected"] == 0
    assert all(r["status"] == sf.STATUS_APPROVED for r in recs)


def test_judge_rejections_are_counted_in_telemetry(extractor, monkeypatch):
    recs = _records(extractor, [GOOD, GOOD2, SUBST])
    answers = iter(["COMPLEMENT", "SUBSTITUTE"])
    monkeypatch.setattr(sf, "ollama_generate", lambda *a, **k: next(answers))
    stats = sf.judge_survivors(recs, "http://x", "m", 5)
    writer = sf.GemmaCopywriter("http://x", None, enabled=False)
    t = sf.build_payload(recs, 1.0, 0, writer, 5, None, stats)["telemetry"]
    assert t["rejected_llm_judge"] == 1 and t["surviving_complementary_bundles"] == 1
    assert t["rejected_same_head_noun"] == 1
    assert t["llm_calls_saved_pct"] == round(100 * (1 - 2 / 3), 1)   # 2 judge calls, 0 copy calls, 3 raw


def test_rules_only_when_ollama_is_down(tmp_path, monkeypatch):
    def no_call(*a, **k):
        raise AssertionError("no LLM call expected")
    monkeypatch.setattr(sf, "ollama_generate", no_call)
    out = tmp_path / "o.json"
    assert sf.main(["--ollama-url", "http://127.0.0.1:9", "--out", str(out)]) == 0
    t = json.loads(out.read_text(encoding="utf-8"))["telemetry"]
    assert t["rejected_llm_judge"] == 0 and t["surviving_complementary_bundles"] == 6
    assert t["llm_calls_saved_pct"] == 62.5


# --- Guarded copy loop -----------------------------------------------------------------

def _writer(monkeypatch, answers, languages=("de", "en")):
    calls = []
    def fake(base, model, prompt, **k):
        calls.append(prompt)
        return answers[min(len(calls), len(answers)) - 1]
    monkeypatch.setattr(sf, "ollama_generate", fake)
    w = sf.GemmaCopywriter("http://x", None, enabled=False, languages=languages)
    w.model = "fake"
    return w, calls


A, B = "Mozzarella 125g", "Cherrytomaten 250g"
EN_OK = {"title": "Tomato and cheese", "copy": f"Buy {B} and get 20% off {A}!"}
EN_BAD = {"title": "Tomato and cheese", "copy": "Buy the tomatoes and get 20% off the cheese!"}
DE_OK = {"title": "Besser zusammen", "copy": f"Kaufe {B} und spare 20% auf {A}!"}


def test_repair_after_failed_names_check(monkeypatch):
    first = json.dumps({"de": DE_OK, "en": EN_BAD})
    w, calls = _writer(monkeypatch, [first, json.dumps(EN_OK)])
    copy = w.write(B, A, 20, "k")
    assert len(calls) == 2 and "exactly as written" in calls[1] and A in calls[1]
    assert copy["en"]["Copy"] == EN_OK["copy"] and copy["de"]["Copy"] == DE_OK["copy"]
    assert set(copy["en"]) == {"title", "Copy"}
    assert w.stats["repairs"] == 1 and w.stats["llm_ok"] == 2


def test_three_failures_give_the_template(monkeypatch):
    first = json.dumps({"de": DE_OK, "en": EN_BAD})
    w, calls = _writer(monkeypatch, [first, json.dumps(EN_BAD), json.dumps(EN_BAD), json.dumps(EN_OK)])
    copy = w.write(B, A, 20, "k")
    assert len(calls) == 3                       # first call + exactly 2 retries
    assert copy["en"] == sf.template_copy(B, A, 20, "k")["en"]
    assert copy["de"]["Copy"] == DE_OK["copy"]


def test_greek_uses_template_by_default_and_is_prompted_only_on_request(monkeypatch):
    w, calls = _writer(monkeypatch, [json.dumps({"de": DE_OK, "en": EN_OK})])
    copy = w.write(B, A, 20, "k")
    assert copy["el"] == sf.template_copy(B, A, 20, "k")["el"] and '"el"' not in calls[0]
    w2, calls2 = _writer(monkeypatch, [json.dumps({"de": DE_OK, "en": EN_OK, "el": {"title": "x", "copy": "y"}})],
                         languages=("de", "en", "el"))
    w2.write(B, A, 20, "k")
    assert '"el"' in calls2[0]


def test_copy_checks_skip_missing_models(monkeypatch):
    monkeypatch.setattr(checks, "_nlp", lambda lang: None)
    names = (B, A)
    assert not checks.failed_ids(checks.run_checks("en", "Deal", EN_OK["copy"], names, 20))


def test_sample_run_does_not_overwrite_the_real_data_file():
    # The README's first command runs the built-in sample; it must not replace the UCI run the showcase labels
    # "UCI Online Retail" (public/smart_flyer_data.json).
    sample = sf.parse_args(["--no-llm"])
    assert sample.out != sf.DEFAULT_OUT and "public" not in sample.out.parts
    real = sf.parse_args(["--csv", "data/raw/OnlineRetail.xlsx"])
    assert real.out == sf.DEFAULT_OUT
    explicit = sf.parse_args(["--no-llm", "--out", "x.json"])
    assert str(explicit.out) == "x.json"


def test_model_help_names_the_preferred_models():
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        sf.parse_args(["--help"])
    assert sf.PREFERRED_MODELS[0] in " ".join(buf.getvalue().split())
