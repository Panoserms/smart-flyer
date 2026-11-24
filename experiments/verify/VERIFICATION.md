# Verification of the smart-flyer study numbers

**ALL CHECKS PASS**: 628 recomputed values compared with summary.json and the site JSON (exact for counts, 3 decimals for rates, +-0.015 for bootstrap CIs; own seed, 10,000 resamples).

Recomputed by `experiments/verify/verify_numbers.py` from the raw gold, JSONL, rating and pair files; rule outcomes by running the frozen pipeline through `cl_features.analyse_pairs`. The analysis scripts were not read.

| check | recomputed | published | result |
|---|---|---|---|
| frozen file SHA-256 vs pre-registration | 5e58efdeceb8ec1d... | 5e58efdeceb8ec1d... | PASS |
| frozen SHA vs summary.json meta | 5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748 | 5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748 | PASS |
| frozen SHA vs site meta | 5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748 | 5e58efdeceb8ec1dba07f32d8c60d973c1f5f46d2749c4e5dabd56e0acf04748 | PASS |
| t1 pair count | 98 | 98 | PASS |
| t1 ids overlapping audit list | 0 | 0 | PASS |
| t1 gold ids == pair ids | True | True | PASS |
| t2 gold ids == pair ids | True | True | PASS |
| t3 gold ids == pair ids | True | True | PASS |
| t1 n | 98 | 98 | PASS |
| t1 L tp (summary) | 16 | 16 | PASS |
| t1 L tp (site) | 16 | 16 | PASS |
| t1 L fp (summary) | 37 | 37 | PASS |
| t1 L fp (site) | 37 | 37 | PASS |
| t1 L fn (summary) | 0 | 0 | PASS |
| t1 L fn (site) | 0 | 0 | PASS |
| t1 L tn (summary) | 45 | 45 | PASS |
| t1 L tn (site) | 45 | 45 | PASS |
| t1 L llm_calls (summary) | 98 | 98 | PASS |
| t1 L llm_calls (site) | 98 | 98 | PASS |
| t1 L precision (summary) | 0.3019 | 0.3019 | PASS |
| t1 L precision (site) | 0.3019 | 0.3019 | PASS |
| t1 L recall (summary) | 1.0000 | 1.0000 | PASS |
| t1 L recall (site) | 1.0000 | 1.0000 | PASS |
| t1 L f1 (summary) | 0.4638 | 0.4638 | PASS |
| t1 L f1 (site) | 0.4638 | 0.4638 | PASS |
| t1 L accuracy (summary) | 0.6224 | 0.6224 | PASS |
| t1 L accuracy (site) | 0.6224 | 0.6224 | PASS |
| t1 C tp (summary) | 14 | 14 | PASS |
| t1 C tp (site) | 14 | 14 | PASS |
| t1 C fp (summary) | 31 | 31 | PASS |
| t1 C fp (site) | 31 | 31 | PASS |
| t1 C fn (summary) | 2 | 2 | PASS |
| t1 C fn (site) | 2 | 2 | PASS |
| t1 C tn (summary) | 51 | 51 | PASS |
| t1 C tn (site) | 51 | 51 | PASS |
| t1 C llm_calls (summary) | 0 | 0 | PASS |
| t1 C llm_calls (site) | 0 | 0 | PASS |
| t1 C precision (summary) | 0.3111 | 0.3111 | PASS |
| t1 C precision (site) | 0.3111 | 0.3111 | PASS |
| t1 C recall (summary) | 0.8750 | 0.8750 | PASS |
| t1 C recall (site) | 0.8750 | 0.8750 | PASS |
| t1 C f1 (summary) | 0.4590 | 0.4590 | PASS |
| t1 C f1 (site) | 0.4590 | 0.4590 | PASS |
| t1 C accuracy (summary) | 0.6633 | 0.6633 | PASS |
| t1 C accuracy (site) | 0.6633 | 0.6633 | PASS |
| t1 CL tp (summary) | 14 | 14 | PASS |
| t1 CL tp (site) | 14 | 14 | PASS |
| t1 CL fp (summary) | 12 | 12 | PASS |
| t1 CL fp (site) | 12 | 12 | PASS |
| t1 CL fn (summary) | 2 | 2 | PASS |
| t1 CL fn (site) | 2 | 2 | PASS |
| t1 CL tn (summary) | 70 | 70 | PASS |
| t1 CL tn (site) | 70 | 70 | PASS |
| t1 CL llm_calls (summary) | 45 | 45 | PASS |
| t1 CL llm_calls (site) | 45 | 45 | PASS |
| t1 CL precision (summary) | 0.5385 | 0.5385 | PASS |
| t1 CL precision (site) | 0.5385 | 0.5385 | PASS |
| t1 CL recall (summary) | 0.8750 | 0.8750 | PASS |
| t1 CL recall (site) | 0.8750 | 0.8750 | PASS |
| t1 CL f1 (summary) | 0.6667 | 0.6667 | PASS |
| t1 CL f1 (site) | 0.6667 | 0.6667 | PASS |
| t1 CL accuracy (summary) | 0.8571 | 0.8571 | PASS |
| t1 CL accuracy (site) | 0.8571 | 0.8571 | PASS |
| t1 Lf tp (summary) | 16 | 16 | PASS |
| t1 Lf tp (site) | 16 | 16 | PASS |
| t1 Lf fp (summary) | 30 | 30 | PASS |
| t1 Lf fp (site) | 30 | 30 | PASS |
| t1 Lf fn (summary) | 0 | 0 | PASS |
| t1 Lf fn (site) | 0 | 0 | PASS |
| t1 Lf tn (summary) | 52 | 52 | PASS |
| t1 Lf tn (site) | 52 | 52 | PASS |
| t1 Lf llm_calls (summary) | 98 | 98 | PASS |
| t1 Lf llm_calls (site) | 98 | 98 | PASS |
| t1 Lf precision (summary) | 0.3478 | 0.3478 | PASS |
| t1 Lf precision (site) | 0.3478 | 0.3478 | PASS |
| t1 Lf recall (summary) | 1.0000 | 1.0000 | PASS |
| t1 Lf recall (site) | 1.0000 | 1.0000 | PASS |
| t1 Lf f1 (summary) | 0.5161 | 0.5161 | PASS |
| t1 Lf f1 (site) | 0.5161 | 0.5161 | PASS |
| t1 Lf accuracy (summary) | 0.6939 | 0.6939 | PASS |
| t1 Lf accuracy (site) | 0.6939 | 0.6939 | PASS |
| t1 CLf tp (summary) | 14 | 14 | PASS |
| t1 CLf tp (site) | 14 | 14 | PASS |
| t1 CLf fp (summary) | 10 | 10 | PASS |
| t1 CLf fp (site) | 10 | 10 | PASS |
| t1 CLf fn (summary) | 2 | 2 | PASS |
| t1 CLf fn (site) | 2 | 2 | PASS |
| t1 CLf tn (summary) | 72 | 72 | PASS |
| t1 CLf tn (site) | 72 | 72 | PASS |
| t1 CLf llm_calls (summary) | 45 | 45 | PASS |
| t1 CLf llm_calls (site) | 45 | 45 | PASS |
| t1 CLf precision (summary) | 0.5833 | 0.5833 | PASS |
| t1 CLf precision (site) | 0.5833 | 0.5833 | PASS |
| t1 CLf recall (summary) | 0.8750 | 0.8750 | PASS |
| t1 CLf recall (site) | 0.8750 | 0.8750 | PASS |
| t1 CLf f1 (summary) | 0.7000 | 0.7000 | PASS |
| t1 CLf f1 (site) | 0.7000 | 0.7000 | PASS |
| t1 CLf accuracy (summary) | 0.8776 | 0.8776 | PASS |
| t1 CLf accuracy (site) | 0.8776 | 0.8776 | PASS |
| t2 n | 106 | 106 | PASS |
| t2 L tp (summary) | 13 | 13 | PASS |
| t2 L tp (site) | 13 | 13 | PASS |
| t2 L fp (summary) | 51 | 51 | PASS |
| t2 L fp (site) | 51 | 51 | PASS |
| t2 L fn (summary) | 2 | 2 | PASS |
| t2 L fn (site) | 2 | 2 | PASS |
| t2 L tn (summary) | 40 | 40 | PASS |
| t2 L tn (site) | 40 | 40 | PASS |
| t2 L llm_calls (summary) | 106 | 106 | PASS |
| t2 L llm_calls (site) | 106 | 106 | PASS |
| t2 L precision (summary) | 0.2031 | 0.2031 | PASS |
| t2 L precision (site) | 0.2031 | 0.2031 | PASS |
| t2 L recall (summary) | 0.8667 | 0.8667 | PASS |
| t2 L recall (site) | 0.8667 | 0.8667 | PASS |
| t2 L f1 (summary) | 0.3291 | 0.3291 | PASS |
| t2 L f1 (site) | 0.3291 | 0.3291 | PASS |
| t2 L accuracy (summary) | 0.5000 | 0.5000 | PASS |
| t2 L accuracy (site) | 0.5000 | 0.5000 | PASS |
| t2 C tp (summary) | 15 | 15 | PASS |
| t2 C tp (site) | 15 | 15 | PASS |
| t2 C fp (summary) | 84 | 84 | PASS |
| t2 C fp (site) | 84 | 84 | PASS |
| t2 C fn (summary) | 0 | 0 | PASS |
| t2 C fn (site) | 0 | 0 | PASS |
| t2 C tn (summary) | 7 | 7 | PASS |
| t2 C tn (site) | 7 | 7 | PASS |
| t2 C llm_calls (summary) | 0 | 0 | PASS |
| t2 C llm_calls (site) | 0 | 0 | PASS |
| t2 C precision (summary) | 0.1515 | 0.1515 | PASS |
| t2 C precision (site) | 0.1515 | 0.1515 | PASS |
| t2 C recall (summary) | 1.0000 | 1.0000 | PASS |
| t2 C recall (site) | 1.0000 | 1.0000 | PASS |
| t2 C f1 (summary) | 0.2632 | 0.2632 | PASS |
| t2 C f1 (site) | 0.2632 | 0.2632 | PASS |
| t2 C accuracy (summary) | 0.2075 | 0.2075 | PASS |
| t2 C accuracy (site) | 0.2075 | 0.2075 | PASS |
| t2 CL tp (summary) | 13 | 13 | PASS |
| t2 CL tp (site) | 13 | 13 | PASS |
| t2 CL fp (summary) | 44 | 44 | PASS |
| t2 CL fp (site) | 44 | 44 | PASS |
| t2 CL fn (summary) | 2 | 2 | PASS |
| t2 CL fn (site) | 2 | 2 | PASS |
| t2 CL tn (summary) | 47 | 47 | PASS |
| t2 CL tn (site) | 47 | 47 | PASS |
| t2 CL llm_calls (summary) | 99 | 99 | PASS |
| t2 CL llm_calls (site) | 99 | 99 | PASS |
| t2 CL precision (summary) | 0.2281 | 0.2281 | PASS |
| t2 CL precision (site) | 0.2281 | 0.2281 | PASS |
| t2 CL recall (summary) | 0.8667 | 0.8667 | PASS |
| t2 CL recall (site) | 0.8667 | 0.8667 | PASS |
| t2 CL f1 (summary) | 0.3611 | 0.3611 | PASS |
| t2 CL f1 (site) | 0.3611 | 0.3611 | PASS |
| t2 CL accuracy (summary) | 0.5660 | 0.5660 | PASS |
| t2 CL accuracy (site) | 0.5660 | 0.5660 | PASS |
| t2 Lf tp (summary) | 12 | 12 | PASS |
| t2 Lf tp (site) | 12 | 12 | PASS |
| t2 Lf fp (summary) | 38 | 38 | PASS |
| t2 Lf fp (site) | 38 | 38 | PASS |
| t2 Lf fn (summary) | 3 | 3 | PASS |
| t2 Lf fn (site) | 3 | 3 | PASS |
| t2 Lf tn (summary) | 53 | 53 | PASS |
| t2 Lf tn (site) | 53 | 53 | PASS |
| t2 Lf llm_calls (summary) | 106 | 106 | PASS |
| t2 Lf llm_calls (site) | 106 | 106 | PASS |
| t2 Lf precision (summary) | 0.2400 | 0.2400 | PASS |
| t2 Lf precision (site) | 0.2400 | 0.2400 | PASS |
| t2 Lf recall (summary) | 0.8000 | 0.8000 | PASS |
| t2 Lf recall (site) | 0.8000 | 0.8000 | PASS |
| t2 Lf f1 (summary) | 0.3692 | 0.3692 | PASS |
| t2 Lf f1 (site) | 0.3692 | 0.3692 | PASS |
| t2 Lf accuracy (summary) | 0.6132 | 0.6132 | PASS |
| t2 Lf accuracy (site) | 0.6132 | 0.6132 | PASS |
| t2 CLf tp (summary) | 12 | 12 | PASS |
| t2 CLf tp (site) | 12 | 12 | PASS |
| t2 CLf fp (summary) | 32 | 32 | PASS |
| t2 CLf fp (site) | 32 | 32 | PASS |
| t2 CLf fn (summary) | 3 | 3 | PASS |
| t2 CLf fn (site) | 3 | 3 | PASS |
| t2 CLf tn (summary) | 59 | 59 | PASS |
| t2 CLf tn (site) | 59 | 59 | PASS |
| t2 CLf llm_calls (summary) | 99 | 99 | PASS |
| t2 CLf llm_calls (site) | 99 | 99 | PASS |
| t2 CLf precision (summary) | 0.2727 | 0.2727 | PASS |
| t2 CLf precision (site) | 0.2727 | 0.2727 | PASS |
| t2 CLf recall (summary) | 0.8000 | 0.8000 | PASS |
| t2 CLf recall (site) | 0.8000 | 0.8000 | PASS |
| t2 CLf f1 (summary) | 0.4068 | 0.4068 | PASS |
| t2 CLf f1 (site) | 0.4068 | 0.4068 | PASS |
| t2 CLf accuracy (summary) | 0.6698 | 0.6698 | PASS |
| t2 CLf accuracy (site) | 0.6698 | 0.6698 | PASS |
| t3 n | 120 | 120 | PASS |
| t3 L tp (summary) | 41 | 41 | PASS |
| t3 L tp (site) | 41 | 41 | PASS |
| t3 L fp (summary) | 38 | 38 | PASS |
| t3 L fp (site) | 38 | 38 | PASS |
| t3 L fn (summary) | 8 | 8 | PASS |
| t3 L fn (site) | 8 | 8 | PASS |
| t3 L tn (summary) | 33 | 33 | PASS |
| t3 L tn (site) | 33 | 33 | PASS |
| t3 L llm_calls (summary) | 120 | 120 | PASS |
| t3 L llm_calls (site) | 120 | 120 | PASS |
| t3 L precision (summary) | 0.5190 | 0.5190 | PASS |
| t3 L precision (site) | 0.5190 | 0.5190 | PASS |
| t3 L recall (summary) | 0.8367 | 0.8367 | PASS |
| t3 L recall (site) | 0.8367 | 0.8367 | PASS |
| t3 L f1 (summary) | 0.6406 | 0.6406 | PASS |
| t3 L f1 (site) | 0.6406 | 0.6406 | PASS |
| t3 L accuracy (summary) | 0.6167 | 0.6167 | PASS |
| t3 L accuracy (site) | 0.6167 | 0.6167 | PASS |
| t3 C tp (summary) | 46 | 46 | PASS |
| t3 C tp (site) | 46 | 46 | PASS |
| t3 C fp (summary) | 37 | 37 | PASS |
| t3 C fp (site) | 37 | 37 | PASS |
| t3 C fn (summary) | 3 | 3 | PASS |
| t3 C fn (site) | 3 | 3 | PASS |
| t3 C tn (summary) | 34 | 34 | PASS |
| t3 C tn (site) | 34 | 34 | PASS |
| t3 C llm_calls (summary) | 0 | 0 | PASS |
| t3 C llm_calls (site) | 0 | 0 | PASS |
| t3 C precision (summary) | 0.5542 | 0.5542 | PASS |
| t3 C precision (site) | 0.5542 | 0.5542 | PASS |
| t3 C recall (summary) | 0.9388 | 0.9388 | PASS |
| t3 C recall (site) | 0.9388 | 0.9388 | PASS |
| t3 C f1 (summary) | 0.6970 | 0.6970 | PASS |
| t3 C f1 (site) | 0.6970 | 0.6970 | PASS |
| t3 C accuracy (summary) | 0.6667 | 0.6667 | PASS |
| t3 C accuracy (site) | 0.6667 | 0.6667 | PASS |
| t3 CL tp (summary) | 38 | 38 | PASS |
| t3 CL tp (site) | 38 | 38 | PASS |
| t3 CL fp (summary) | 21 | 21 | PASS |
| t3 CL fp (site) | 21 | 21 | PASS |
| t3 CL fn (summary) | 11 | 11 | PASS |
| t3 CL fn (site) | 11 | 11 | PASS |
| t3 CL tn (summary) | 50 | 50 | PASS |
| t3 CL tn (site) | 50 | 50 | PASS |
| t3 CL llm_calls (summary) | 83 | 83 | PASS |
| t3 CL llm_calls (site) | 83 | 83 | PASS |
| t3 CL precision (summary) | 0.6441 | 0.6441 | PASS |
| t3 CL precision (site) | 0.6441 | 0.6441 | PASS |
| t3 CL recall (summary) | 0.7755 | 0.7755 | PASS |
| t3 CL recall (site) | 0.7755 | 0.7755 | PASS |
| t3 CL f1 (summary) | 0.7037 | 0.7037 | PASS |
| t3 CL f1 (site) | 0.7037 | 0.7037 | PASS |
| t3 CL accuracy (summary) | 0.7333 | 0.7333 | PASS |
| t3 CL accuracy (site) | 0.7333 | 0.7333 | PASS |
| t3 Lf tp (summary) | 41 | 41 | PASS |
| t3 Lf tp (site) | 41 | 41 | PASS |
| t3 Lf fp (summary) | 32 | 32 | PASS |
| t3 Lf fp (site) | 32 | 32 | PASS |
| t3 Lf fn (summary) | 8 | 8 | PASS |
| t3 Lf fn (site) | 8 | 8 | PASS |
| t3 Lf tn (summary) | 39 | 39 | PASS |
| t3 Lf tn (site) | 39 | 39 | PASS |
| t3 Lf llm_calls (summary) | 120 | 120 | PASS |
| t3 Lf llm_calls (site) | 120 | 120 | PASS |
| t3 Lf precision (summary) | 0.5616 | 0.5616 | PASS |
| t3 Lf precision (site) | 0.5616 | 0.5616 | PASS |
| t3 Lf recall (summary) | 0.8367 | 0.8367 | PASS |
| t3 Lf recall (site) | 0.8367 | 0.8367 | PASS |
| t3 Lf f1 (summary) | 0.6721 | 0.6721 | PASS |
| t3 Lf f1 (site) | 0.6721 | 0.6721 | PASS |
| t3 Lf accuracy (summary) | 0.6667 | 0.6667 | PASS |
| t3 Lf accuracy (site) | 0.6667 | 0.6667 | PASS |
| t3 CLf tp (summary) | 38 | 38 | PASS |
| t3 CLf tp (site) | 38 | 38 | PASS |
| t3 CLf fp (summary) | 20 | 20 | PASS |
| t3 CLf fp (site) | 20 | 20 | PASS |
| t3 CLf fn (summary) | 11 | 11 | PASS |
| t3 CLf fn (site) | 11 | 11 | PASS |
| t3 CLf tn (summary) | 51 | 51 | PASS |
| t3 CLf tn (site) | 51 | 51 | PASS |
| t3 CLf llm_calls (summary) | 83 | 83 | PASS |
| t3 CLf llm_calls (site) | 83 | 83 | PASS |
| t3 CLf precision (summary) | 0.6552 | 0.6552 | PASS |
| t3 CLf precision (site) | 0.6552 | 0.6552 | PASS |
| t3 CLf recall (summary) | 0.7755 | 0.7755 | PASS |
| t3 CLf recall (site) | 0.7755 | 0.7755 | PASS |
| t3 CLf f1 (summary) | 0.7103 | 0.7103 | PASS |
| t3 CLf f1 (site) | 0.7103 | 0.7103 | PASS |
| t3 CLf accuracy (summary) | 0.7417 | 0.7417 | PASS |
| t3 CLf accuracy (site) | 0.7417 | 0.7417 | PASS |
| pooled n | 324 | 324 | PASS |
| pooled L tp (summary) | 70 | 70 | PASS |
| pooled L tp (site) | 70 | 70 | PASS |
| pooled L fp (summary) | 126 | 126 | PASS |
| pooled L fp (site) | 126 | 126 | PASS |
| pooled L fn (summary) | 10 | 10 | PASS |
| pooled L fn (site) | 10 | 10 | PASS |
| pooled L tn (summary) | 118 | 118 | PASS |
| pooled L tn (site) | 118 | 118 | PASS |
| pooled L llm_calls (summary) | 324 | 324 | PASS |
| pooled L llm_calls (site) | 324 | 324 | PASS |
| pooled L precision (summary) | 0.3571 | 0.3571 | PASS |
| pooled L precision (site) | 0.3571 | 0.3571 | PASS |
| pooled L recall (summary) | 0.8750 | 0.8750 | PASS |
| pooled L recall (site) | 0.8750 | 0.8750 | PASS |
| pooled L f1 (summary) | 0.5072 | 0.5072 | PASS |
| pooled L f1 (site) | 0.5072 | 0.5072 | PASS |
| pooled L accuracy (summary) | 0.5802 | 0.5802 | PASS |
| pooled L accuracy (site) | 0.5802 | 0.5802 | PASS |
| pooled C tp (summary) | 75 | 75 | PASS |
| pooled C tp (site) | 75 | 75 | PASS |
| pooled C fp (summary) | 152 | 152 | PASS |
| pooled C fp (site) | 152 | 152 | PASS |
| pooled C fn (summary) | 5 | 5 | PASS |
| pooled C fn (site) | 5 | 5 | PASS |
| pooled C tn (summary) | 92 | 92 | PASS |
| pooled C tn (site) | 92 | 92 | PASS |
| pooled C llm_calls (summary) | 0 | 0 | PASS |
| pooled C llm_calls (site) | 0 | 0 | PASS |
| pooled C precision (summary) | 0.3304 | 0.3304 | PASS |
| pooled C precision (site) | 0.3304 | 0.3304 | PASS |
| pooled C recall (summary) | 0.9375 | 0.9375 | PASS |
| pooled C recall (site) | 0.9375 | 0.9375 | PASS |
| pooled C f1 (summary) | 0.4886 | 0.4886 | PASS |
| pooled C f1 (site) | 0.4886 | 0.4886 | PASS |
| pooled C accuracy (summary) | 0.5154 | 0.5154 | PASS |
| pooled C accuracy (site) | 0.5154 | 0.5154 | PASS |
| pooled CL tp (summary) | 65 | 65 | PASS |
| pooled CL tp (site) | 65 | 65 | PASS |
| pooled CL fp (summary) | 77 | 77 | PASS |
| pooled CL fp (site) | 77 | 77 | PASS |
| pooled CL fn (summary) | 15 | 15 | PASS |
| pooled CL fn (site) | 15 | 15 | PASS |
| pooled CL tn (summary) | 167 | 167 | PASS |
| pooled CL tn (site) | 167 | 167 | PASS |
| pooled CL llm_calls (summary) | 227 | 227 | PASS |
| pooled CL llm_calls (site) | 227 | 227 | PASS |
| pooled CL precision (summary) | 0.4577 | 0.4577 | PASS |
| pooled CL precision (site) | 0.4577 | 0.4577 | PASS |
| pooled CL recall (summary) | 0.8125 | 0.8125 | PASS |
| pooled CL recall (site) | 0.8125 | 0.8125 | PASS |
| pooled CL f1 (summary) | 0.5856 | 0.5856 | PASS |
| pooled CL f1 (site) | 0.5856 | 0.5856 | PASS |
| pooled CL accuracy (summary) | 0.7160 | 0.7160 | PASS |
| pooled CL accuracy (site) | 0.7160 | 0.7160 | PASS |
| pooled Lf tp (summary) | 69 | 69 | PASS |
| pooled Lf tp (site) | 69 | 69 | PASS |
| pooled Lf fp (summary) | 100 | 100 | PASS |
| pooled Lf fp (site) | 100 | 100 | PASS |
| pooled Lf fn (summary) | 11 | 11 | PASS |
| pooled Lf fn (site) | 11 | 11 | PASS |
| pooled Lf tn (summary) | 144 | 144 | PASS |
| pooled Lf tn (site) | 144 | 144 | PASS |
| pooled Lf llm_calls (summary) | 324 | 324 | PASS |
| pooled Lf llm_calls (site) | 324 | 324 | PASS |
| pooled Lf precision (summary) | 0.4083 | 0.4083 | PASS |
| pooled Lf precision (site) | 0.4083 | 0.4083 | PASS |
| pooled Lf recall (summary) | 0.8625 | 0.8625 | PASS |
| pooled Lf recall (site) | 0.8625 | 0.8625 | PASS |
| pooled Lf f1 (summary) | 0.5542 | 0.5542 | PASS |
| pooled Lf f1 (site) | 0.5542 | 0.5542 | PASS |
| pooled Lf accuracy (summary) | 0.6574 | 0.6574 | PASS |
| pooled Lf accuracy (site) | 0.6574 | 0.6574 | PASS |
| pooled CLf tp (summary) | 64 | 64 | PASS |
| pooled CLf tp (site) | 64 | 64 | PASS |
| pooled CLf fp (summary) | 62 | 62 | PASS |
| pooled CLf fp (site) | 62 | 62 | PASS |
| pooled CLf fn (summary) | 16 | 16 | PASS |
| pooled CLf fn (site) | 16 | 16 | PASS |
| pooled CLf tn (summary) | 182 | 182 | PASS |
| pooled CLf tn (site) | 182 | 182 | PASS |
| pooled CLf llm_calls (summary) | 227 | 227 | PASS |
| pooled CLf llm_calls (site) | 227 | 227 | PASS |
| pooled CLf precision (summary) | 0.5079 | 0.5079 | PASS |
| pooled CLf precision (site) | 0.5079 | 0.5079 | PASS |
| pooled CLf recall (summary) | 0.8000 | 0.8000 | PASS |
| pooled CLf recall (site) | 0.8000 | 0.8000 | PASS |
| pooled CLf f1 (summary) | 0.6214 | 0.6214 | PASS |
| pooled CLf f1 (site) | 0.6214 | 0.6214 | PASS |
| pooled CLf accuracy (summary) | 0.7593 | 0.7593 | PASS |
| pooled CLf accuracy (site) | 0.7593 | 0.7593 | PASS |
| t1 CL-L:f1 estimate | 0.2029 | 0.2029 | PASS |
| t1 CL-L:f1 CI low (+-0.015) | 0.0727 | 0.0753 | PASS |
| t1 CL-L:f1 CI high (+-0.015) | 0.3333 | 0.3318 | PASS |
| t1 CL-L:precision estimate | 0.2366 | 0.2366 | PASS |
| t1 CL-L:precision CI low (+-0.015) | 0.1172 | 0.1200 | PASS |
| t1 CL-L:precision CI high (+-0.015) | 0.3806 | 0.3759 | PASS |
| t1 CL-C:f1 estimate | 0.2077 | 0.2077 | PASS |
| t1 CL-C:f1 CI low (+-0.015) | 0.1224 | 0.1254 | PASS |
| t1 CL-C:f1 CI high (+-0.015) | 0.3044 | 0.3055 | PASS |
| t1 Lf-L:f1 estimate | 0.0524 | 0.0524 | PASS |
| t1 Lf-L:f1 CI low (+-0.015) | 0.0131 | 0.0123 | PASS |
| t1 Lf-L:f1 CI high (+-0.015) | 0.1002 | 0.1005 | PASS |
| t2 CL-L:f1 estimate | 0.0320 | 0.0320 | PASS |
| t2 CL-L:f1 CI low (+-0.015) | 0.0103 | 0.0100 | PASS |
| t2 CL-L:f1 CI high (+-0.015) | 0.0602 | 0.0606 | PASS |
| t2 CL-L:precision estimate | 0.0249 | 0.0249 | PASS |
| t2 CL-L:precision CI low (+-0.015) | 0.0077 | 0.0074 | PASS |
| t2 CL-L:precision CI high (+-0.015) | 0.0503 | 0.0505 | PASS |
| t2 CL-C:f1 estimate | 0.0980 | 0.0980 | PASS |
| t2 CL-C:f1 CI low (+-0.015) | 0.0180 | 0.0171 | PASS |
| t2 CL-C:f1 CI high (+-0.015) | 0.1715 | 0.1713 | PASS |
| t2 Lf-L:f1 estimate | 0.0401 | 0.0401 | PASS |
| t2 Lf-L:f1 CI low (+-0.015) | -0.0272 | -0.0256 | PASS |
| t2 Lf-L:f1 CI high (+-0.015) | 0.0971 | 0.0977 | PASS |
| t3 CL-L:f1 estimate | 0.0631 | 0.0631 | PASS |
| t3 CL-L:f1 CI low (+-0.015) | 0.0034 | 0.0005 | PASS |
| t3 CL-L:f1 CI high (+-0.015) | 0.1253 | 0.1233 | PASS |
| t3 CL-L:precision estimate | 0.1251 | 0.1251 | PASS |
| t3 CL-L:precision CI low (+-0.015) | 0.0600 | 0.0585 | PASS |
| t3 CL-L:precision CI high (+-0.015) | 0.1995 | 0.1986 | PASS |
| t3 CL-C:f1 estimate | 0.0067 | 0.0067 | PASS |
| t3 CL-C:f1 CI low (+-0.015) | -0.0696 | -0.0695 | PASS |
| t3 CL-C:f1 CI high (+-0.015) | 0.0806 | 0.0801 | PASS |
| t3 Lf-L:f1 estimate | 0.0315 | 0.0315 | PASS |
| t3 Lf-L:f1 CI low (+-0.015) | -0.0190 | -0.0208 | PASS |
| t3 Lf-L:f1 CI high (+-0.015) | 0.0830 | 0.0833 | PASS |
| pooled CL-L:f1 estimate | 0.0783 | 0.0783 | PASS |
| pooled CL-L:f1 CI low (+-0.015) | 0.0368 | 0.0379 | PASS |
| pooled CL-L:f1 CI high (+-0.015) | 0.1176 | 0.1181 | PASS |
| pooled CL-L:precision estimate | 0.1006 | 0.1006 | PASS |
| pooled CL-L:precision CI low (+-0.015) | 0.0628 | 0.0635 | PASS |
| pooled CL-L:precision CI high (+-0.015) | 0.1405 | 0.1411 | PASS |
| pooled CL-C:f1 estimate | 0.0970 | 0.0970 | PASS |
| pooled CL-C:f1 CI low (+-0.015) | 0.0453 | 0.0463 | PASS |
| pooled CL-C:f1 CI high (+-0.015) | 0.1461 | 0.1458 | PASS |
| pooled Lf-L:f1 estimate | 0.0470 | 0.0470 | PASS |
| pooled Lf-L:f1 CI low (+-0.015) | 0.0132 | 0.0141 | PASS |
| pooled Lf-L:f1 CI high (+-0.015) | 0.0811 | 0.0802 | PASS |
| t1 McNemar L_right_CL_wrong | 2 | 2 | PASS |
| t1 McNemar CL_right_L_wrong | 25 | 25 | PASS |
| t1 McNemar exact p | 0.0000 | 0.0000 | PASS |
| t1 confusion L/complement | [16, 16, 0] | [16, 16, 0] | PASS |
| t1 confusion L/substitute | [65, 35, 30] | [65, 35, 30] | PASS |
| t1 confusion L/not_a_bundle | [17, 2, 15] | [17, 2, 15] | PASS |
| t1 confusion C/complement | [16, 14, 2] | [16, 14, 2] | PASS |
| t1 confusion C/substitute | [65, 14, 51] | [65, 14, 51] | PASS |
| t1 confusion C/not_a_bundle | [17, 17, 0] | [17, 17, 0] | PASS |
| t1 confusion CL/complement | [16, 14, 2] | [16, 14, 2] | PASS |
| t1 confusion CL/substitute | [65, 10, 55] | [65, 10, 55] | PASS |
| t1 confusion CL/not_a_bundle | [17, 2, 15] | [17, 2, 15] | PASS |
| t1 confusion Lf/complement | [16, 16, 0] | [16, 16, 0] | PASS |
| t1 confusion Lf/substitute | [65, 29, 36] | [65, 29, 36] | PASS |
| t1 confusion Lf/not_a_bundle | [17, 1, 16] | [17, 1, 16] | PASS |
| t1 confusion CLf/complement | [16, 14, 2] | [16, 14, 2] | PASS |
| t1 confusion CLf/substitute | [65, 9, 56] | [65, 9, 56] | PASS |
| t1 confusion CLf/not_a_bundle | [17, 1, 16] | [17, 1, 16] | PASS |
| t2 McNemar L_right_CL_wrong | 0 | 0 | PASS |
| t2 McNemar CL_right_L_wrong | 7 | 7 | PASS |
| t2 McNemar exact p | 0.0156 | 0.0156 | PASS |
| t2 confusion L/complement | [15, 13, 2] | [15, 13, 2] | PASS |
| t2 confusion L/substitute | [14, 13, 1] | [14, 13, 1] | PASS |
| t2 confusion L/not_a_bundle | [77, 38, 39] | [77, 38, 39] | PASS |
| t2 confusion C/complement | [15, 15, 0] | [15, 15, 0] | PASS |
| t2 confusion C/substitute | [14, 7, 7] | [14, 7, 7] | PASS |
| t2 confusion C/not_a_bundle | [77, 77, 0] | [77, 77, 0] | PASS |
| t2 confusion CL/complement | [15, 13, 2] | [15, 13, 2] | PASS |
| t2 confusion CL/substitute | [14, 6, 8] | [14, 6, 8] | PASS |
| t2 confusion CL/not_a_bundle | [77, 38, 39] | [77, 38, 39] | PASS |
| t2 confusion Lf/complement | [15, 12, 3] | [15, 12, 3] | PASS |
| t2 confusion Lf/substitute | [14, 12, 2] | [14, 12, 2] | PASS |
| t2 confusion Lf/not_a_bundle | [77, 26, 51] | [77, 26, 51] | PASS |
| t2 confusion CLf/complement | [15, 12, 3] | [15, 12, 3] | PASS |
| t2 confusion CLf/substitute | [14, 6, 8] | [14, 6, 8] | PASS |
| t2 confusion CLf/not_a_bundle | [77, 26, 51] | [77, 26, 51] | PASS |
| t3 McNemar L_right_CL_wrong | 3 | 3 | PASS |
| t3 McNemar CL_right_L_wrong | 17 | 17 | PASS |
| t3 McNemar exact p | 0.0026 | 0.0026 | PASS |
| t3 confusion L/complement | [49, 41, 8] | [49, 41, 8] | PASS |
| t3 confusion L/substitute | [48, 32, 16] | [48, 32, 16] | PASS |
| t3 confusion L/not_a_bundle | [23, 6, 17] | [23, 6, 17] | PASS |
| t3 confusion C/complement | [49, 46, 3] | [49, 46, 3] | PASS |
| t3 confusion C/substitute | [48, 22, 26] | [48, 22, 26] | PASS |
| t3 confusion C/not_a_bundle | [23, 15, 8] | [23, 15, 8] | PASS |
| t3 confusion CL/complement | [49, 38, 11] | [49, 38, 11] | PASS |
| t3 confusion CL/substitute | [48, 17, 31] | [48, 17, 31] | PASS |
| t3 confusion CL/not_a_bundle | [23, 4, 19] | [23, 4, 19] | PASS |
| t3 confusion Lf/complement | [49, 41, 8] | [49, 41, 8] | PASS |
| t3 confusion Lf/substitute | [48, 27, 21] | [48, 27, 21] | PASS |
| t3 confusion Lf/not_a_bundle | [23, 5, 18] | [23, 5, 18] | PASS |
| t3 confusion CLf/complement | [49, 38, 11] | [49, 38, 11] | PASS |
| t3 confusion CLf/substitute | [48, 17, 31] | [48, 17, 31] | PASS |
| t3 confusion CLf/not_a_bundle | [23, 3, 20] | [23, 3, 20] | PASS |
| pooled McNemar L_right_CL_wrong | 5 | 5 | PASS |
| pooled McNemar CL_right_L_wrong | 49 | 49 | PASS |
| pooled McNemar exact p | 0.0000 | 0.0000 | PASS |
| pooled confusion L/complement | [80, 70, 10] | [80, 70, 10] | PASS |
| pooled confusion L/substitute | [127, 80, 47] | [127, 80, 47] | PASS |
| pooled confusion L/not_a_bundle | [117, 46, 71] | [117, 46, 71] | PASS |
| pooled confusion C/complement | [80, 75, 5] | [80, 75, 5] | PASS |
| pooled confusion C/substitute | [127, 43, 84] | [127, 43, 84] | PASS |
| pooled confusion C/not_a_bundle | [117, 109, 8] | [117, 109, 8] | PASS |
| pooled confusion CL/complement | [80, 65, 15] | [80, 65, 15] | PASS |
| pooled confusion CL/substitute | [127, 33, 94] | [127, 33, 94] | PASS |
| pooled confusion CL/not_a_bundle | [117, 44, 73] | [117, 44, 73] | PASS |
| pooled confusion Lf/complement | [80, 69, 11] | [80, 69, 11] | PASS |
| pooled confusion Lf/substitute | [127, 68, 59] | [127, 68, 59] | PASS |
| pooled confusion Lf/not_a_bundle | [117, 32, 85] | [117, 32, 85] | PASS |
| pooled confusion CLf/complement | [80, 64, 16] | [80, 64, 16] | PASS |
| pooled confusion CLf/substitute | [127, 32, 95] | [127, 32, 95] | PASS |
| pooled confusion CLf/not_a_bundle | [117, 30, 87] | [117, 30, 87] | PASS |
| t1 invalid plain | 0 | 0 | PASS |
| t1 invalid informed | 0 | 0 | PASS |
| t2 invalid plain | 0 | 0 | PASS |
| t2 invalid informed | 0 | 0 | PASS |
| t3 invalid plain | 0 | 0 | PASS |
| t3 invalid informed | 0 | 0 | PASS |
| pooled invalid plain | 0 | 0 | PASS |
| pooled invalid informed | 0 | 0 | PASS |
| robustness n (swap plain / informed / case / rules) | [100, 100, 100, 100] | [100, 100, 100, 100] | PASS |
| swap flip L | 0.2600 | 0.2600 | PASS |
| swap flip Lf | 0.1400 | 0.1400 | PASS |
| case flip L | 0.1800 | 0.1800 | PASS |
| swap flip C (rules file) | 0.0000 | 0.0000 | PASS |
| case flip C (rules file) | 0.0000 | 0.0000 | PASS |
| robustness rules base_reject == my cl_reject | 0 | 0 | PASS |
| swap McNemar L-only flipped | 17 | 17 | PASS |
| swap McNemar Lf-only flipped | 5 | 5 | PASS |
| swap McNemar p | 0.0169 | 0.0169 | PASS |
| swap diff Lf-L estimate | -0.1200 | -0.1200 | PASS |
| swap diff CI low (+-0.015) | -0.2100 | -0.2100 | PASS |
| swap diff CI high (+-0.015) | -0.0300 | -0.0300 | PASS |
| site robustness swap_flip_rate | {'L': 0.26, 'Lf': 0.14, 'C': 0.0} | {'L': 0.26, 'Lf': 0.14, 'C': 0.0} | PASS |
| site robustness case_flip_rate | {'L': 0.18, 'C': 0.0} | {'L': 0.18, 'C': 0.0} | PASS |
| t1 annotator pair coverage | [98, 98, 98] | [98, 98, 98] | PASS |
| t1 kappa 3-class (site) | 0.8375 | 0.8375 | PASS |
| t1 kappa binary (site) | 0.6260 | 0.6260 | PASS |
| t1 gold counts (site) | {'not_a_bundle': 17, 'substitute': 65, 'complement': 16} | {'complement': 16, 'substitute': 65, 'not_a_bundle': 17} | PASS |
| t2 annotator pair coverage | [106, 106, 106] | [106, 106, 106] | PASS |
| t2 kappa 3-class (site) | 0.8208 | 0.8208 | PASS |
| t2 kappa binary (site) | 0.8208 | 0.8208 | PASS |
| t2 gold counts (site) | {'substitute': 14, 'complement': 15, 'not_a_bundle': 77} | {'complement': 15, 'substitute': 14, 'not_a_bundle': 77} | PASS |
| t3 annotator pair coverage | [120, 120, 120] | [120, 120, 120] | PASS |
| t3 kappa 3-class (site) | 0.9869 | 0.9869 | PASS |
| t3 kappa binary (site) | 0.9827 | 0.9827 | PASS |
| t3 gold counts (site) | {'substitute': 48, 'complement': 49, 'not_a_bundle': 23} | {'complement': 49, 'substitute': 48, 'not_a_bundle': 23} | PASS |
| rating coverage key/R1/R2 | [270, 270, 270] | [270, 270, 270] | PASS |
| copy pooled/raw publishable share | 0.2667 | 0.2667 | PASS |
| copy pooled/raw template share | 0.0000 | 0.0000 | PASS |
| copy pooled/raw publishable (site) | 0.2667 | 0.2667 | PASS |
| copy pooled/guarded publishable share | 0.5000 | 0.5000 | PASS |
| copy pooled/guarded template share | 0.0222 | 0.0222 | PASS |
| copy pooled/guarded publishable (site) | 0.5000 | 0.5000 | PASS |
| copy pooled/delex publishable share | 0.2000 | 0.2000 | PASS |
| copy pooled/delex template share | 0.0000 | 0.0000 | PASS |
| copy pooled/delex publishable (site) | 0.2000 | 0.2000 | PASS |
| copy de/raw publishable share | 0.2000 | 0.2000 | PASS |
| copy de/raw template share | 0.0000 | 0.0000 | PASS |
| copy de/raw publishable (site) | 0.2000 | 0.2000 | PASS |
| copy de/guarded publishable share | 0.4000 | 0.4000 | PASS |
| copy de/guarded template share | 0.0000 | 0.0000 | PASS |
| copy de/guarded publishable (site) | 0.4000 | 0.4000 | PASS |
| copy de/delex publishable share | 0.0000 | 0.0000 | PASS |
| copy de/delex template share | 0.0000 | 0.0000 | PASS |
| copy de/delex publishable (site) | 0.0000 | 0.0000 | PASS |
| copy en/raw publishable share | 0.5000 | 0.5000 | PASS |
| copy en/raw template share | 0.0000 | 0.0000 | PASS |
| copy en/raw publishable (site) | 0.5000 | 0.5000 | PASS |
| copy en/guarded publishable share | 0.8333 | 0.8333 | PASS |
| copy en/guarded template share | 0.0000 | 0.0000 | PASS |
| copy en/guarded publishable (site) | 0.8333 | 0.8333 | PASS |
| copy en/delex publishable share | 0.6000 | 0.6000 | PASS |
| copy en/delex template share | 0.0000 | 0.0000 | PASS |
| copy en/delex publishable (site) | 0.6000 | 0.6000 | PASS |
| copy el/raw publishable share | 0.1000 | 0.1000 | PASS |
| copy el/raw template share | 0.0000 | 0.0000 | PASS |
| copy el/raw publishable (site) | 0.1000 | 0.1000 | PASS |
| copy el/guarded publishable share | 0.2667 | 0.2667 | PASS |
| copy el/guarded template share | 0.0667 | 0.0667 | PASS |
| copy el/guarded publishable (site) | 0.2667 | 0.2667 | PASS |
| copy el/delex publishable share | 0.0000 | 0.0000 | PASS |
| copy el/delex template share | 0.0000 | 0.0000 | PASS |
| copy el/delex publishable (site) | 0.0000 | 0.0000 | PASS |
| rater kappa on publishable | 0.7563 | 0.7563 | PASS |
| rater percent agreement | 88.5185 | 88.5185 | PASS |
| site pair count | 324 | 324 | PASS |
| site per-pair reject (stop_term or same_head or compound) vs my cl_reject, mismatches | 0 | 0 | PASS |
| site per-pair verdict/gold mismatches vs raw | 0 | 0 | PASS |
| site per-pair records t1 L tp/fp/fn/tn | [16, 37, 0, 45] | [16, 37, 0, 45] | PASS |
| site per-pair records t1 C tp/fp/fn/tn | [14, 31, 2, 51] | [14, 31, 2, 51] | PASS |
| site per-pair records t1 CL tp/fp/fn/tn | [14, 12, 2, 70] | [14, 12, 2, 70] | PASS |
| site per-pair records t1 Lf tp/fp/fn/tn | [16, 30, 0, 52] | [16, 30, 0, 52] | PASS |
| site per-pair records t1 CLf tp/fp/fn/tn | [14, 10, 2, 72] | [14, 10, 2, 72] | PASS |
| site per-pair records t2 L tp/fp/fn/tn | [13, 51, 2, 40] | [13, 51, 2, 40] | PASS |
| site per-pair records t2 C tp/fp/fn/tn | [15, 84, 0, 7] | [15, 84, 0, 7] | PASS |
| site per-pair records t2 CL tp/fp/fn/tn | [13, 44, 2, 47] | [13, 44, 2, 47] | PASS |
| site per-pair records t2 Lf tp/fp/fn/tn | [12, 38, 3, 53] | [12, 38, 3, 53] | PASS |
| site per-pair records t2 CLf tp/fp/fn/tn | [12, 32, 3, 59] | [12, 32, 3, 59] | PASS |
| site per-pair records t3 L tp/fp/fn/tn | [41, 38, 8, 33] | [41, 38, 8, 33] | PASS |
| site per-pair records t3 C tp/fp/fn/tn | [46, 37, 3, 34] | [46, 37, 3, 34] | PASS |
| site per-pair records t3 CL tp/fp/fn/tn | [38, 21, 11, 50] | [38, 21, 11, 50] | PASS |
| site per-pair records t3 Lf tp/fp/fn/tn | [41, 32, 8, 39] | [41, 32, 8, 39] | PASS |
| site per-pair records t3 CLf tp/fp/fn/tn | [38, 20, 11, 51] | [38, 20, 11, 51] | PASS |
| site per-pair records pooled L tp/fp/fn/tn | [70, 126, 10, 118] | [70, 126, 10, 118] | PASS |
| site per-pair records pooled C tp/fp/fn/tn | [75, 152, 5, 92] | [75, 152, 5, 92] | PASS |
| site per-pair records pooled CL tp/fp/fn/tn | [65, 77, 15, 167] | [65, 77, 15, 167] | PASS |
| site per-pair records pooled Lf tp/fp/fn/tn | [69, 100, 11, 144] | [69, 100, 11, 144] | PASS |
| site per-pair records pooled CLf tp/fp/fn/tn | [64, 62, 16, 182] | [64, 62, 16, 182] | PASS |
| spot-check 10 random pairs (verdicts, names, gold) vs raw | 0 | 0 | PASS |
| spot-check 10 random copy cells (verbatim text, publishable, source) vs copy_runs/ratings | 0 | 0 | PASS |
| all 270 site copy cells vs raw (extra) | 0 | 0 | PASS |
| REPORT.md pooled F1 headline | True | True | PASS |
| REPORT.md call counts (CL vs L) | True | True | PASS |
| human check pooled n | 100 | 100 | PASS |
| human check pooled kappa 3-class | 0.6872 | 0.6872 | PASS |
| human check pooled kappa binary | 0.6591 | 0.6591 | PASS |
| human check t1 n | 30 | 30 | PASS |
| human check t1 kappa 3-class | 0.6000 | 0.6000 | PASS |
| human check t1 kappa binary | 0.7097 | 0.7097 | PASS |
| human check t2 n | 33 | 33 | PASS |
| human check t2 kappa 3-class | 0.3700 | 0.3700 | PASS |
| human check t2 kappa binary | 0.3189 | 0.3189 | PASS |
| human check t3 n | 37 | 37 | PASS |
| human check t3 kappa 3-class | 0.9575 | 0.9575 | PASS |
| human check t3 kappa binary | 0.9433 | 0.9433 | PASS |
| human check disagreements | 21 | 21 | PASS |
| human check systems silver L f1 | 0.5750 | 0.5750 | PASS |
| human check systems silver L precision | 0.4182 | 0.4182 | PASS |
| human check systems silver C f1 | 0.4719 | 0.4719 | PASS |
| human check systems silver C precision | 0.3281 | 0.3281 | PASS |
| human check systems silver CL f1 | 0.6333 | 0.6333 | PASS |
| human check systems silver CL precision | 0.5429 | 0.5429 | PASS |
| human check systems silver Lf f1 | 0.6053 | 0.6053 | PASS |
| human check systems silver Lf precision | 0.4510 | 0.4510 | PASS |
| human check systems silver CLf f1 | 0.6441 | 0.6441 | PASS |
| human check systems silver CLf precision | 0.5588 | 0.5588 | PASS |
| human check systems human L f1 | 0.5806 | 0.5806 | PASS |
| human check systems human L precision | 0.4909 | 0.4909 | PASS |
| human check systems human C f1 | 0.6471 | 0.6471 | PASS |
| human check systems human C precision | 0.5156 | 0.5156 | PASS |
| human check systems human CL f1 | 0.6027 | 0.6027 | PASS |
| human check systems human CL precision | 0.6286 | 0.6286 | PASS |
| human check systems human Lf f1 | 0.5618 | 0.5618 | PASS |
| human check systems human Lf precision | 0.4902 | 0.4902 | PASS |
| human check systems human CLf f1 | 0.5556 | 0.5556 | PASS |
| human check systems human CLf precision | 0.5882 | 0.5882 | PASS |
| REPORT.md has the human check section | True | True | PASS |
