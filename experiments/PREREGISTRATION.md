# Pre-registration: does computational linguistics improve a small LLM's results?

Written 2026-11-06, **before** any test set was mined or labelled. Changes after this date go in the
"Amendments" section at the end, with the reason. All outcomes are reported, including those against the hypotheses.

## Question

A small local LLM (`gemma4:e2b-it-qat`, 4.6B parameters, on an 8 GB laptop) has two jobs in a promotion pipeline:
(A) decide which co-purchased product pairs make a good cross-sell bundle, and (B) write the flyer copy.
Does a computational-linguistics layer (regex, parser-based head nouns, compound analysis, output checks) make the
LLM's results **better**, not only cheaper? And where does it not help?

## Data

| Set | What | Role |
|---|---|---|
| `dev` | UCI Online Retail 2010–11: 300 distinct 1→1 pairs (`min_support` 0.01, lift ≥ 1.5) | tuning; rules were already fitted here, so results here are in-sample |
| `dev_de` | 30 constructed German grocery pairs | tuning for German |
| `t1` | UCI Online Retail II, 2009–10 sheet, same mining, pairs also in `dev` removed, at most 250 (seeded sample) | temporal hold-out: same shop and vocabulary, new pairs |
| `t2` | `Groceries` (arules; 9,835 real receipts; 169 English category names), same mining, at most 250 | cross-domain hold-out |
| `t3` | 120 constructed German grocery pairs, written without access to the rules | cross-language hold-out; **not real baskets** |

Rule-based attribute learning uses only the catalog of the set being scored. No test row may be inspected before the freeze.

## Gold labels

Three classes (`eval/ANNOTATION_GUIDE.md`): complement, substitute, not_a_bundle. Positive class for all metrics = complement.
Two independent annotators, blind to rules, model output and each other, using different row orders.
Disagreements go to a third blind adjudicator. Cohen's kappa is reported per set.
**Limitation, to be stated wherever results appear:** the annotators are large LLMs (two models from the same family), not people, so this is
a silver standard. The system's author does not resolve any label.

## Task A: triage conditions

| Id | Condition | LLM calls |
|---|---|---|
| `L` | **LLM only**: raw names + the best prompt | all pairs |
| `C` | **Linguistics only**: stop-item regex + head noun + compound stages | none |
| `CL` | **Cascade**: `C` rejects first, `L` judges the survivors | survivors only |
| `Lf` | **Informed LLM**: as `L`, with the parser's analysis added to the prompt | all pairs |
| `CLf` | **Informed cascade**: `C` rejects first, `Lf` judges the survivors | survivors only |

The retail margin & lift gate is a business rule and is excluded from these comparisons.

Fairness to the LLM: three prompt variants (definitions only; definitions + one invented example per class; short reasoning then answer)
are compared on a dev sample (50 `dev` + 30 `dev_de` pairs). The one with the best macro-F1 becomes the prompt for both `L` and `Lf`.
Temperature 0, fixed seed, thinking off. An unparseable answer counts as not positive and is logged.

## Rule freeze

Rules may be tuned on `dev` and `dev_de` only. Before any test scoring, the pipeline is copied to
`experiments/frozen/export_smart_flyer_v1.py` and its SHA-256 is recorded below. All test results use that file.

## Hypotheses (Task A)

Pooled over `t1`, `t2`, `t3`, with per-set results always shown beside the pooled ones.

- **H1 (cheaper, not worse):** `CL` uses at least 50 % fewer LLM calls than `L`, and its F1 is non-inferior
  (lower bound of the paired 95 % bootstrap CI of F1(`CL`) − F1(`L`) above −0.03).
- **H2 (better precision):** precision(`CL`) > precision(`L`); paired 95 % CI excludes 0.
- **H3 (the LLM adds value):** F1(`CL`) > F1(`C`); paired 95 % CI excludes 0.
- **H4 (information helps):** F1(`Lf`) > F1(`L`); paired 95 % CI excludes 0.
- **H5 (robustness):** on 100 seeded test pairs, swapping product order flips `Lf`'s verdict less often than `L`'s (exact McNemar on flipped / not flipped).

Paired bootstrap: 10,000 resamples of pairs, fixed seed. Holm correction across the H2, H3, H4 and H5 tests; both raw and corrected results are reported.
Known weakness: pairs share products, so pair-level resampling understates uncertainty. An item-level sensitivity check is reported for H1–H4 if feasible.

Also reported: exact McNemar for `L` vs `CL`; confusion by gold class; LLM tokens and wall-clock; the verdict flip rate of the rule stage under the same swap and under a change of letter case.

## Task B: flyer copy

Bundles: 30 gold-complement pairs (10 highest-lift from `t1`, 10 from `t2`, 10 seeded from `t3`). Discount 20 %. Languages DE, EN, EL.

| Id | Condition |
|---|---|
| `raw` | LLM prompt with product names; output used as it comes |
| `guarded` | same first output as `raw`; linguistic checks; targeted repair prompt (max 2 retries per language); then a fixed template |
| `delex` | delexicalised: the LLM writes with placeholders `[A]`, `[B]`; the same checks and repair; exact names filled in by code; then a fixed template |

Checks: both product names verbatim; discount number present and no other invented number; one sentence (spaCy, per language); length limit;
right language and script; German `du` address; no unsupported claim from a small lexicon (free, today only ...); Greek out-of-vocabulary words (frequency list).

**Blind rating:** two raters (the same two LLMs), who see the texts shuffled with condition hidden, score each text: faithful, grammatical, natural (1–5), publishable.
Primary outcome: share of texts both raters call publishable.

- **H6:** publishable share of `guarded` > `raw`, paired bootstrap over bundles, per language and pooled.
- **H7:** publishable share of `delex` > `raw`, same test.

Also reported: which checks fail most, retries, template-fallback rate per condition, calls, tokens, wall-clock.
A text served by a template counts as served but is flagged, and the LLM-only share is reported separately so that templates cannot hide a failing model.

## Interpretation rules

- Results describe this small model. A larger model could need rules less for accuracy, though not for cost. No claim about large LLMs is made.
- If `L` beats `CL` on quality, the page and the README say so in the first paragraph.
- `t3` is constructed and is never presented as real shopping data.
- Latency is measured on one busy laptop and is indicative only; call and token counts are the primary cost measures.

## Freeze record

Frozen 2026-11-06 after one tuning round on `dev` and `dev_de` gold, before any test scoring.
`experiments/frozen/export_smart_flyer_v1.py`, SHA-256 `5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748`.

Tuning round, all from errors visible in `dev_de`: stop terms inside German compounds (Flaschenpfand); a content word shared by both German names
(Hofgut Gouda / Alpenkrone Gouda, Energy Drink Zero / Classic); keeping the final e of unknown compounds in lemmas (Schuhcreme).
Before tuning, rules alone on the tuning sets: `dev` P 0.42, R 0.93, F1 0.58; `dev_de` P 0.55, R 0.92, F1 0.69. After: `dev` unchanged; `dev_de` P 0.69, R 0.92, F1 0.79.

## Amendments

**A1 (2026-11-06, before any labels or scoring of t1).** The data audit found that 34 of the 132 `t1` pairs have the same stock codes
as a `dev` pair: the same products with a re-worded description (6 differ only in punctuation). The rules were tuned on `dev`, so these are
not new pairs. They are removed by `experiments/apply_t1_audit.py`, leaving 98 pairs. `t1` and `t2` came out below the cap of 250, so no sampling happened.
Decision made from ids and stock codes only, without reading test rows.

**A2 (2026-11-06, after results, no change to any test).** The `delex` condition as implemented shows the model only the placeholders, with one seed, so its prompt was identical for all 30 bundles and it produced one text per language. H7 is reported as run (not supported), but its bootstrap CI treats one text rated 30 times as 30 observations and should be read as a single case. A version that shows the model the product names was not tested.

**A3 (2026-11-12, after results, no change to any test or result).** A human check of the silver-standard labels: 100 of the
324 test pairs, stratified by set and gold label (seed 20261111), in two batches of 50, labelled blind by the author with
the same guide (`eval/ANNOTATION_GUIDE.md`) in `eval/human/label_pairs.html`. It is scored as agreement with the adjudicated
gold (`experiments/human_audit.py --score`). It validates the labels; it does not replace them, and the reported results stay as run.
The author knows the study, so this is not a blind outside annotator.
Result (2026-11-12): kappa 0.687 three-class, 0.659 binary (t1 0.600, t2 0.370, t3 0.958). Exploratory, not tested as a
hypothesis: on these 100 pairs the CL-L precision difference is +0.138 [+0.039, +0.250] against the human labels and +0.125
against the silver labels. Reported in the last section of `experiments/results/REPORT.md`.

**A4 (2026-11-24, wording only).** For publication, the annotators and raters are described as two large LLMs from the same family instead of
by model name. No condition, test, label or result changed.
