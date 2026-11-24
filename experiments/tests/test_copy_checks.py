"""copy_checks: texts that must pass and texts that must fail."""
import pytest

import cl_features
import common
import copy_checks as cc


@pytest.fixture(scope="module")
def pipeline():
    return cl_features.load_pipeline(common.resolve_pipeline_path(None, ["dev"]))


A, B = "SET/6 RED SPOTTY PAPER CUPS", "SET/6 RED SPOTTY PAPER PLATES"
GREEK_NAMES = (A, B)


def run(lang, title, copy, names=GREEK_NAMES, pct=20):
    return cc.run_checks(lang, title, copy, names, pct)


@pytest.mark.parametrize("lang", ["de", "en", "el"])
def test_pipeline_template_passes_without_false_alarm(pipeline, lang):
    anchor, pull = "WHITE HANGING HEART T-LIGHT HOLDER", "RED HANGING HEART T-LIGHT HOLDER"
    names = (pipeline.display_name(anchor), pipeline.display_name(pull))
    for key in ("k1", "k2", "k3", "k4"):
        text = pipeline.template_copy(anchor, pull, 20, key)[lang]
        results = cc.run_checks(lang, text["title"], text["Copy"], names, 20)
        assert cc.failed_ids(results) == [], (lang, text, results)


def test_correct_greek_sentences_pass():
    for title, copy in [
        ("Καλύτερα μαζί", f"Αγοράστε το προϊόν «{A}» και κερδίστε 20% έκπτωση στο προϊόν «{B}»."),
        ("Το τέλειο δίδυμο", f"Πάρτε {A} και απολαύστε 20% έκπτωση στο {B}!"),
        ("Προσφορά της εβδομάδας", f"Με την αγορά του {A} παίρνετε 20% έκπτωση στο {B}."),
    ]:
        assert cc.failed_ids(run("el", title, copy)) == [], (title, copy)


def test_invented_greek_word_fails():
    copy = "Αγοράστε το Σετ/6 Κόκκινα Χαρτοπετρώτες και λάβετε το Σετ/6 Κόκκινα Χαρτοπετρώτες με 20% έκπτωση!"
    failed = cc.failed_ids(run("el", "Καλύτερα μαζί", copy))
    assert "oov" in failed
    assert "Χαρτοπετρώτες" in cc.oov_words(run("el", "Καλύτερα μαζί", copy))


def test_thai_letters_inside_greek_fail():
    failed = cc.failed_ids(run("el", "Καλύτερα μαζί", f"Αγοράστε το {A} και κερδίστε 20% ส่วนลด στο {B}."))
    assert "language" in failed


def test_invented_title_word_fails():
    results = run("el", "Εξοικονόμηση Σουκχάρου", f"Αγοράστε το {A} και κερδίστε 20% έκπτωση στο {B}.")
    assert "oov" in cc.failed_ids(results)
    assert "Σουκχάρου" in cc.oov_words(results)


def test_english_good_and_bad():
    good = run("en", "Better Together", f"Buy {A} and get 20% off {B}.")
    assert cc.failed_ids(good) == []
    assert "names" in cc.failed_ids(run("en", "Better Together", "Buy the cups and get 20% off the plates."))
    assert "number" in cc.failed_ids(run("en", "Better Together", f"Buy {A} and get 25% off {B}."))
    assert "number" in cc.failed_ids(run("en", "Better Together", f"Buy {A} and get 20% off {B} for 5 euros."))
    assert "sentence" in cc.failed_ids(run("en", "Better Together", f"Buy {A}. Get 20% off {B}."))
    assert "claim" in cc.failed_ids(run("en", "Better Together", f"Buy {A} and get 20% off {B} today only."))
    assert "length" in cc.failed_ids(run("en", "One Two Three Four Five Six Seven", f"Buy {A} and get 20% off {B}."))


def test_german_address_and_language():
    assert cc.failed_ids(run("de", "Besser zusammen", f"Kaufe „{A}“ und sichere dir 20 % Rabatt auf „{B}“.")) == []
    assert "address" in cc.failed_ids(run("de", "Besser zusammen", f"Kaufen Sie {A} und sichern Sie sich 20% Rabatt auf {B}."))
    assert "language" in cc.failed_ids(run("de", "Besser zusammen", f"Buy {A} and get the 20% off {B} for you."))


def test_empty_output_is_a_parse_failure():
    assert cc.failed_ids(run("en", "", "")) == ["parse"]


def test_delex_context_prompts_and_fill():
    import run_copy
    run_copy.checks = cc
    names = ("Grillkohle Buchenholz 3kg", "Thüringer Rostbratwurst 6 Stück")
    prompt = run_copy.delex_context_prompt(names[0], names[1], 20)
    assert names[0] in prompt and names[1] in prompt and "[A]" in prompt and "[B]" in prompt
    entry = ("Grillzeit am Wochenende", "Kaufe [A] und sichere dir 20% Rabatt auf [B]!")
    results, title, copy = run_copy.assess("de", entry, names, 20, delex=True)
    assert not cc.failed_ids(results)
    assert names[0] in copy and names[1] in copy and "[A]" not in copy
    bad_results, _, _ = run_copy.assess("de", ("Grill", "Kaufe die Kohlen!"), names, 20, delex=True)
    rep = run_copy.repair_prompt({"pct": 20}, "de", ("Grill", "Kaufe die Kohlen!"), bad_results, names,
                                 delex=True, delex_context=True)
    assert names[0] in rep and "[A]" in rep
