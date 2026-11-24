# Study data contract: `smart_flyer_study.json`

Written by `experiments/export_site_data.py`; read by the website page (a copy is in `experiments/results/site_data.json`).
A fixture has `"fixture": true` and the page must then show a visible "placeholder data" banner.

```jsonc
{
  "fixture": false,
  "meta": { "model": "gemma4:e2b-it-qat", "generated": "2026-11-06", "frozen_sha256": "…", "prompt_variant": "definitions" },
  "datasets": [
    { "id": "t1", "name": "Online Retail II, 2009–10", "kind": "real baskets, same shop as tuning, new pairs", "lang": "en",
      "n_transactions": 20000, "n_pairs": 200,
      "kappa_3class": 0.71, "kappa_binary": 0.78, "gold_counts": { "complement": 40, "substitute": 120, "not_a_bundle": 40 } }
    // t2, t3 (t3 has n_transactions null and kind "constructed German pairs, not real baskets")
  ],
  "pairs": [
    {
      "id": "t1-0001", "set": "t1", "a": "JUMBO BAG RED RETROSPOT", "b": "JUMBO BAG PINK POLKADOT",
      "lift": 7.1,                 // null when the set has no baskets
      "gold": "substitute",        // complement | substitute | not_a_bundle
      "cl": {
        "stop_term": null,         // string when module "regex" fires
        "same_head": true,         // module "head": same head noun, or one head ends in the other
        "compound": null,          // module "compound": shared stem/compound as a string, e.g. "-milch" or "jumbo bag"; null otherwise
        "head_a": "Bag", "head_b": "Bag",
        "pos_a": "PROPN", "pos_b": "PROPN",
        "stripped_a": ["jumbo", "red", "retrospot"], "stripped_b": ["jumbo", "pink", "polkadot"]
      },
      "llm": { "plain": "SUBSTITUTE", "informed": "SUBSTITUTE" }   // COMPLEMENT | SUBSTITUTE | NEITHER | INVALID
    }
  ],
  // Positive = complement. Conditions: L, C, CL, Lf, CLf. Sets: t1, t2, t3, pooled.
  "triage": {
    "pooled": {
      "L":  { "precision": 0.5, "recall": 0.8, "f1": 0.62, "f1_ci": [0.55, 0.68], "accuracy": 0.7, "llm_calls": 500, "llm_tokens": 90000, "llm_seconds": 1500,
              "kept": 120, "tp": 60, "fp": 60, "fn": 15, "tn": 365 }
      // C, CL, Lf, CLf; C has llm_calls 0
    }
    // t1, t2, t3: same shape
  },
  "hypotheses": [
    { "id": "H1", "text": "…", "estimate": 0.01, "ci": [-0.02, 0.04], "p_raw": null, "p_holm": null, "supported": true, "note": "…" }
    // H2 … H7
  ],
  "robustness": { "n": 100, "swap_flip_rate": { "L": 0.12, "Lf": 0.05, "C": 0.0 }, "case_flip_rate": { "L": 0.1, "C": 0.02 } },
  "copy": {
    "bundles": [
      {
        "id": "b01", "set": "t1", "anchor": "…", "pull": "…", "pct": 20,
        "conditions": {
          "raw": {
            "de": {
              "title": "…", "copy": "…",     // final text (empty strings when the output could not be parsed)
              "source": "llm",               // llm | template | none
              "tries": 1,
              "failed_checks": ["names"],    // checks the FINAL text fails; ids: names, number, sentence, length, language, address, claim, oov, parse
              "first_try_failed_checks": ["names"],
              "oov_words": [],
              "ratings": { "publishable_both": false, "grammatical_both": true, "natural_mean": 3.5 }
            },
            "en": { }, "el": { }
          },
          "guarded": { }, "delex": { }
        }
      }
    ],
    "summary": {
      // Keys: pooled, de, en, el. Conditions: raw, guarded, delex.
      "pooled": { "raw": { "publishable_share": 0.4, "publishable_ci": [0.3, 0.5], "llm_share": 1.0, "checks_pass_share": 0.3, "calls": 30, "seconds": 300 } }
    }
  }
}
```
