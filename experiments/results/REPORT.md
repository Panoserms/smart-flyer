# Smart Flyer study: results

Model gemma4:e2b-it-qat, triage prompt `examples`, positive class = complement, 10000 bootstrap resamples. Pipeline SHA-256 `5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748`.

Limitations: the gold labels come from two LLM annotators and an LLM adjudicator, so this is a silver standard. t3 is constructed, not real shopping data. The results describe this small model only. Latency is from one busy laptop; call and token counts are the primary cost measures.

## Summary

Pooled over t1, t2, t3: F1 L 0.507, C 0.489, CL 0.586, Lf 0.554, CLf 0.621. Precision L 0.357, CL 0.458. CL used 227 LLM calls against 324 for L. CL has the higher F1 than L (0.586 vs 0.507).

## Hypotheses

A hypothesis in H2-H4 counts as supported when the estimate has the predicted sign, the paired 95% bootstrap CI excludes 0 and the Holm-corrected p (across H2-H5) is below 0.05. H5 needs the predicted sign and a Holm-corrected exact McNemar p below 0.05. H1 needs at least 50% fewer calls and a CI lower bound above -0.03. H6 and H7 need a positive difference with a CI that excludes 0 (raw p; not in the Holm family).

**H1: not supported.** CL uses at least 50% fewer LLM calls than L and its F1 is non-inferior (lower CI bound of F1(CL) - F1(L) above -0.03).

Estimate +0.0783, 95% CI [+0.038, +0.118], raw p -, Holm p -. CL used 227 LLM calls, L used 324: 29.9% fewer. F1 CL 0.586, L 0.507. Item-level CI of the difference: [+0.004, +0.158].

**H2: supported.** Precision of CL is higher than precision of L.

Estimate +0.1006, 95% CI [+0.064, +0.141], raw p 0.0002, Holm p 0.0008. Difference in precision CL - L. Raw p 0.0002, Holm-corrected 0.0008. Item-level CI: [+0.034, +0.200].

**H3: supported.** F1 of CL is higher than F1 of C (the LLM adds value to the rules).

Estimate +0.0970, 95% CI [+0.046, +0.146], raw p 0.0004, Holm p 0.0012. Difference in F1 CL - C. Raw p 0.0004, Holm-corrected 0.0012. Item-level CI: [-0.013, +0.193].

**H4: supported.** F1 of Lf (LLM with the parser's analysis in the prompt) is higher than F1 of L.

Estimate +0.0470, 95% CI [+0.014, +0.080], raw p 0.0054, Holm p 0.0108. Difference in F1 Lf - L. Raw p 0.0054, Holm-corrected 0.0108. Item-level CI: [-0.016, +0.106].

**H5: supported.** Swapping the product order flips the verdict of Lf less often than that of L.

Estimate -0.1200, 95% CI [-0.210, -0.030], raw p 0.0169, Holm p 0.0169. Swap flip rate: L 0.260, Lf 0.140 on 100 pairs. Discordant pairs: L flipped only 17, Lf flipped only 5. Estimate = flip rate Lf - flip rate L. Raw p 0.0169, Holm-corrected 0.0169.

**H6: supported.** The publishable share of guarded copy is higher than that of raw copy.

Estimate +0.2333, 95% CI [+0.122, +0.344], raw p 0.0002, Holm p -. Publishable share guarded 0.500 vs raw 0.267 (pooled over 30 bundles x 3 languages). Per language, difference [95% CI] and raw p: de: +0.200 [+0.067, +0.333] p=0.0042, en: +0.333 [+0.167, +0.500] p=0.0002, el: +0.167 [+0.033, +0.300] p=0.0114. Not part of the Holm family; raw p only.

**H7: not supported.** The publishable share of delex copy is higher than that of raw copy.

Estimate -0.0667, 95% CI [-0.178, +0.044], raw p 0.2442, Holm p -. Publishable share delex 0.200 vs raw 0.267 (pooled over 30 bundles x 3 languages). Per language, difference [95% CI] and raw p: de: -0.200 [-0.333, -0.067] p=0.0040, en: +0.100 [-0.133, +0.333] p=0.4806, el: -0.100 [-0.233, +0.000] p=0.0828. Not part of the Holm family; raw p only.

## Task A: triage

### t1 (n = 98)

| cond | precision | recall | F1 | F1 95% CI | accuracy | kept | tp | fp | fn | tn | LLM calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 0.302 | 1.000 | 0.464 | [0.310, 0.603] | 0.622 | 53 | 16 | 37 | 0 | 45 | 98 | 54831 | 44.1 |
| C | 0.311 | 0.875 | 0.459 | [0.292, 0.604] | 0.663 | 45 | 14 | 31 | 2 | 51 | 0 | 0 | 0.0 |
| CL | 0.538 | 0.875 | 0.667 | [0.478, 0.813] | 0.857 | 26 | 14 | 12 | 2 | 70 | 45 | 25206 | 20.4 |
| Lf | 0.348 | 1.000 | 0.516 | [0.351, 0.657] | 0.694 | 46 | 16 | 30 | 0 | 52 | 98 | 63054 | 83.9 |
| CLf | 0.583 | 0.875 | 0.700 | [0.514, 0.844] | 0.878 | 24 | 14 | 10 | 2 | 72 | 45 | 28989 | 38.1 |

| difference | estimate | 95% CI | p (bootstrap) |
|---|---|---|---|
| CL-L:f1 | +0.2029 | [+0.075, +0.332] | 0.0028 |
| CL-L:precision | +0.2366 | [+0.120, +0.376] | 0.0002 |
| CL-C:f1 | +0.2077 | [+0.125, +0.306] | 0.0002 |
| Lf-L:f1 | +0.0524 | [+0.012, +0.100] | 0.0162 |

Exact McNemar, L vs CL (per-pair correctness): L right and CL wrong 2, CL right and L wrong 25, p = 0.0000.

### t2 (n = 106)

| cond | precision | recall | F1 | F1 95% CI | accuracy | kept | tp | fp | fn | tn | LLM calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 0.203 | 0.867 | 0.329 | [0.187, 0.458] | 0.500 | 64 | 13 | 51 | 2 | 40 | 106 | 57854 | 48.2 |
| C | 0.152 | 1.000 | 0.263 | [0.152, 0.367] | 0.208 | 99 | 15 | 84 | 0 | 7 | 0 | 0 | 0.0 |
| CL | 0.228 | 0.867 | 0.361 | [0.209, 0.494] | 0.566 | 57 | 13 | 44 | 2 | 47 | 99 | 54032 | 45.1 |
| Lf | 0.240 | 0.800 | 0.369 | [0.208, 0.515] | 0.613 | 50 | 12 | 38 | 3 | 53 | 106 | 66242 | 68.6 |
| CLf | 0.273 | 0.800 | 0.407 | [0.235, 0.560] | 0.670 | 44 | 12 | 32 | 3 | 59 | 99 | 61866 | 64.0 |

| difference | estimate | 95% CI | p (bootstrap) |
|---|---|---|---|
| CL-L:f1 | +0.0320 | [+0.010, +0.061] | 0.0010 |
| CL-L:precision | +0.0249 | [+0.007, +0.050] | 0.0010 |
| CL-C:f1 | +0.0980 | [+0.017, +0.171] | 0.0206 |
| Lf-L:f1 | +0.0401 | [-0.026, +0.098] | 0.2058 |

Exact McNemar, L vs CL (per-pair correctness): L right and CL wrong 0, CL right and L wrong 7, p = 0.0156.

### t3 (n = 120)

| cond | precision | recall | F1 | F1 95% CI | accuracy | kept | tp | fp | fn | tn | LLM calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 0.519 | 0.837 | 0.641 | [0.539, 0.730] | 0.617 | 79 | 41 | 38 | 8 | 33 | 120 | 66994 | 63.2 |
| C | 0.554 | 0.939 | 0.697 | [0.600, 0.781] | 0.667 | 83 | 46 | 37 | 3 | 34 | 0 | 0 | 0.0 |
| CL | 0.644 | 0.776 | 0.704 | [0.596, 0.793] | 0.733 | 59 | 38 | 21 | 11 | 50 | 83 | 46305 | 43.8 |
| Lf | 0.562 | 0.837 | 0.672 | [0.571, 0.762] | 0.667 | 73 | 41 | 32 | 8 | 39 | 120 | 77017 | 86.1 |
| CLf | 0.655 | 0.776 | 0.710 | [0.604, 0.800] | 0.742 | 58 | 38 | 20 | 11 | 51 | 83 | 53213 | 59.3 |

| difference | estimate | 95% CI | p (bootstrap) |
|---|---|---|---|
| CL-L:f1 | +0.0631 | [+0.001, +0.123] | 0.0488 |
| CL-L:precision | +0.1251 | [+0.059, +0.199] | 0.0004 |
| CL-C:f1 | +0.0067 | [-0.069, +0.080] | 0.8607 |
| Lf-L:f1 | +0.0315 | [-0.021, +0.083] | 0.2360 |

Exact McNemar, L vs CL (per-pair correctness): L right and CL wrong 3, CL right and L wrong 17, p = 0.0026.

### pooled (n = 324)

| cond | precision | recall | F1 | F1 95% CI | accuracy | kept | tp | fp | fn | tn | LLM calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 0.357 | 0.875 | 0.507 | [0.433, 0.577] | 0.580 | 196 | 70 | 126 | 10 | 118 | 324 | 179679 | 155.5 |
| C | 0.330 | 0.938 | 0.489 | [0.417, 0.556] | 0.515 | 227 | 75 | 152 | 5 | 92 | 0 | 0 | 0.0 |
| CL | 0.458 | 0.812 | 0.586 | [0.505, 0.661] | 0.716 | 142 | 65 | 77 | 15 | 167 | 227 | 125543 | 109.2 |
| Lf | 0.408 | 0.863 | 0.554 | [0.478, 0.625] | 0.657 | 169 | 69 | 100 | 11 | 144 | 324 | 206313 | 238.6 |
| CLf | 0.508 | 0.800 | 0.621 | [0.540, 0.696] | 0.759 | 126 | 64 | 62 | 16 | 182 | 227 | 144068 | 161.5 |

| difference | estimate | 95% CI | p (bootstrap) |
|---|---|---|---|
| CL-L:f1 | +0.0783 | [+0.038, +0.118] | 0.0004 |
| CL-L:precision | +0.1006 | [+0.064, +0.141] | 0.0002 |
| CL-C:f1 | +0.0970 | [+0.046, +0.146] | 0.0004 |
| Lf-L:f1 | +0.0470 | [+0.014, +0.080] | 0.0054 |

Exact McNemar, L vs CL (per-pair correctness): L right and CL wrong 5, CL right and L wrong 49, p = 0.0000.

### Item-level sensitivity (pooled; product names resampled instead of pairs)

| difference | estimate | 95% CI |
|---|---|---|
| CL-L:f1 | +0.0783 | [+0.004, +0.158] |
| CL-L:precision | +0.1006 | [+0.034, +0.200] |
| CL-C:f1 | +0.0970 | [-0.013, +0.193] |
| Lf-L:f1 | +0.0470 | [-0.016, +0.106] |

### Confusion by gold class (pooled): predicted complement / predicted other

| gold class | n | L | C | CL | Lf | CLf |
|---|---|---|---|---|---|---|
| complement | 80 | 70 / 10 | 75 / 5 | 65 / 15 | 69 / 11 | 64 / 16 |
| substitute | 127 | 80 / 47 | 43 / 84 | 33 / 94 | 68 / 59 | 32 / 95 |
| not_a_bundle | 117 | 46 / 71 | 109 / 8 | 44 / 73 | 32 / 85 | 30 / 87 |

LLM verdicts by gold class, plain prompt:

| gold class | COMPLEMENT | SUBSTITUTE | NEITHER | INVALID |
|---|---|---|---|---|
| complement | 70 | 1 | 9 | 0 |
| substitute | 80 | 40 | 7 | 0 |
| not_a_bundle | 46 | 0 | 71 | 0 |

LLM verdicts by gold class, informed prompt:

| gold class | COMPLEMENT | SUBSTITUTE | NEITHER | INVALID |
|---|---|---|---|---|
| complement | 69 | 0 | 11 | 0 |
| substitute | 68 | 47 | 12 | 0 |
| not_a_bundle | 32 | 0 | 85 | 0 |

### INVALID verdicts (unparseable answers and failed calls count as not positive)

| set | prompt | pairs | INVALID | call errors | INVALID among survivors |
|---|---|---|---|---|---|
| t1 | plain | 98 | 0 | 0 | 0 |
| t1 | informed | 98 | 0 | 0 | 0 |
| t2 | plain | 106 | 0 | 0 | 0 |
| t2 | informed | 106 | 0 | 0 | 0 |
| t3 | plain | 120 | 0 | 0 | 0 |
| t3 | informed | 120 | 0 | 0 | 0 |
| pooled | plain | 324 | 0 | 0 | 0 |
| pooled | informed | 324 | 0 | 0 | 0 |

### Robustness flips (verdict changed)

|  | L | Lf | C |
|---|---|---|---|
| swap of product order | 0.260 | 0.140 | 0.000 |
| letter-case change | 0.180 | not run | 0.000 |
100 seeded test pairs. A flip is any change of the LLM's verdict word (INVALID included) or of the rule decision.

## Task B: flyer copy

30 bundles x 3 languages x 3 conditions. Publishable = both raters said yes; an empty output counts as not publishable (0 empty texts were not rated).

### pooled

| cond | publishable | 95% CI | LLM-written | template | no output | checks pass (final) | checks pass (1st try) | publishable among LLM texts | mean natural | retries | calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.267 | [0.156, 0.389] | 1.000 | 0.000 | 0.000 | 0.389 | 0.389 | 0.267 | 3.17 | 0 | 30.0 | 10173 | 408.7 |
| guarded | 0.500 | [0.389, 0.611] | 0.978 | 0.022 | 0.000 | 1.000 | 0.389 | 0.511 | 3.43 | 60 | 90.0 | 27487 | 724.3 |
| delex | 0.200 | [0.144, 0.256] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.200 | 2.39 | 0 | 30.0 | 9570 | 334.7 |

### de

| cond | publishable | 95% CI | LLM-written | template | no output | checks pass (final) | checks pass (1st try) | publishable among LLM texts | mean natural | retries | calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.200 | [0.067, 0.333] | 1.000 | 0.000 | 0.000 | 0.400 | 0.400 | 0.200 | 3.25 | 0 | 10.0 | 3391 | 136.2 |
| guarded | 0.400 | [0.233, 0.567] | 1.000 | 0.000 | 0.000 | 1.000 | 0.400 | 0.400 | 3.27 | 18 | 28.0 | 8171 | 219.1 |
| delex | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.83 | 0 | 10.0 | 3190 | 111.6 |

### en

| cond | publishable | 95% CI | LLM-written | template | no output | checks pass (final) | checks pass (1st try) | publishable among LLM texts | mean natural | retries | calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.500 | [0.333, 0.667] | 1.000 | 0.000 | 0.000 | 0.533 | 0.533 | 0.500 | 3.83 | 0 | 10.0 | 3391 | 136.2 |
| guarded | 0.833 | [0.700, 0.967] | 1.000 | 0.000 | 0.000 | 1.000 | 0.533 | 0.833 | 3.88 | 15 | 25.0 | 7423 | 208.2 |
| delex | 0.600 | [0.433, 0.767] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.600 | 3.52 | 0 | 10.0 | 3190 | 111.6 |

### el

| cond | publishable | 95% CI | LLM-written | template | no output | checks pass (final) | checks pass (1st try) | publishable among LLM texts | mean natural | retries | calls | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.100 | [0.000, 0.233] | 1.000 | 0.000 | 0.000 | 0.233 | 0.233 | 0.100 | 2.43 | 0 | 10.0 | 3391 | 136.2 |
| guarded | 0.267 | [0.100, 0.433] | 0.933 | 0.067 | 0.000 | 1.000 | 0.233 | 0.286 | 3.15 | 27 | 37.0 | 11893 | 297.0 |
| delex | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.83 | 0 | 10.0 | 3190 | 111.6 |

### Check failures (texts failing each check, all languages)

| check | raw 1st try | guarded 1st try | delex 1st try | raw final | guarded final | delex final |
|---|---|---|---|---|---|---|
| claim | 37 | 37 | 0 | 37 | 0 | 0 |
| language | 1 | 1 | 0 | 1 | 0 | 0 |
| names | 35 | 35 | 0 | 35 | 0 | 0 |
| number | 13 | 13 | 0 | 13 | 0 | 0 |
| oov | 13 | 13 | 0 | 13 | 0 | 0 |

### Per-language tests (difference of publishable share to raw)

| language | H6 guarded-raw | 95% CI | p | H7 delex-raw | 95% CI | p |
|---|---|---|---|---|---|---|
| pooled | +0.233 | [+0.122, +0.344] | 0.0002 | -0.067 | [-0.178, +0.044] | 0.2442 |
| de | +0.200 | [+0.067, +0.333] | 0.0042 | -0.200 | [-0.333, -0.067] | 0.0040 |
| en | +0.333 | [+0.167, +0.500] | 0.0002 | +0.100 | [-0.133, +0.333] | 0.4806 |
| el | +0.167 | [+0.033, +0.300] | 0.0114 | -0.100 | [-0.233, +0.000] | 0.0828 |

### Rater agreement (Cohen's kappa)

| dimension | kappa | % agreement | texts |
|---|---|---|---|
| publishable | 0.756 | 88.5 | 270 |
| grammatical | 0.831 | 91.9 | 270 |
| faithful | 0.971 | 98.9 | 270 |

Calls for guarded include the shared first call it reuses from raw; each three-language first call is split evenly over the three languages in per-language rows.

## Human check of the labels (amendment A3, after the results)

100 test pairs, stratified by set and gold label, labelled blind by the author with the same guide. The author knows the study, so this is not an outside annotator. Agreement with the adjudicated silver gold; nothing above this section was changed by it.

| slice | n | 3-class kappa | binary kappa | 3-class % | binary % |
|---|---|---|---|---|---|
| pooled | 100 | 0.687 | 0.659 | 79.0 | 85.0 |
| t1 | 30 | 0.600 | 0.710 | 76.7 | 90.0 |
| t2 | 33 | 0.370 | 0.319 | 60.6 | 66.7 |
| t3 | 37 | 0.958 | 0.943 | 97.3 | 97.3 |

21 disagreements (gold -> human): not_a_bundle -> complement 13, substitute -> not_a_bundle 5, not_a_bundle -> substitute 1, substitute -> complement 1, complement -> substitute 1.

Exploratory: the same predictions on these 100 pairs, scored against each label set (complements: silver 25, human 38).

| condition | F1 silver | F1 human | precision silver | precision human |
|---|---|---|---|---|
| L | 0.575 | 0.581 | 0.418 | 0.491 |
| C | 0.472 | 0.647 | 0.328 | 0.516 |
| CL | 0.633 | 0.603 | 0.543 | 0.629 |
| Lf | 0.605 | 0.562 | 0.451 | 0.490 |
| CLf | 0.644 | 0.556 | 0.559 | 0.588 |

- CL-L:f1: silver +0.058 [-0.052, +0.161], human +0.022 [-0.076, +0.112]
- CL-L:precision: silver +0.125 [+0.031, +0.233], human +0.138 [+0.039, +0.250]

