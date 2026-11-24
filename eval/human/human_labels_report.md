# Human Audit vs. Adjudicated Silver-Standard Gold

Labelled pairs: **100 / 100**

| Slice | n | 3-class κ | Binary κ (complement) | 3-class agreement | Binary agreement |
|---|--:|--:|--:|--:|--:|
| **Pooled** | 100 | 0.687 | 0.659 | 79.0% | 85.0% |
| `t1` | 30 | 0.600 | 0.710 | 76.7% | 90.0% |
| `t2` | 33 | 0.370 | 0.319 | 60.6% | 66.7% |
| `t3` | 37 | 0.958 | 0.943 | 97.3% | 97.3% |

## Triage conditions on these 100 pairs (exploratory)

| Condition | P silver | R silver | F1 silver | P human | R human | F1 human |
|---|--:|--:|--:|--:|--:|--:|
| L | 0.418 | 0.920 | 0.575 | 0.491 | 0.711 | 0.581 |
| C | 0.328 | 0.840 | 0.472 | 0.516 | 0.868 | 0.647 |
| CL | 0.543 | 0.760 | 0.633 | 0.629 | 0.579 | 0.603 |
| Lf | 0.451 | 0.920 | 0.605 | 0.490 | 0.658 | 0.562 |
| CLf | 0.559 | 0.760 | 0.644 | 0.588 | 0.526 | 0.556 |

- silver CL-L:f1: +0.058 [-0.052, +0.161]
- silver CL-L:precision: +0.125 [+0.031, +0.233]
- human CL-L:f1: +0.022 [-0.076, +0.112]
- human CL-L:precision: +0.138 [+0.039, +0.250]
