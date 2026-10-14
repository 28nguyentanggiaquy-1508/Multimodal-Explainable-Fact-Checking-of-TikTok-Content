# ABSA V1 vs V2

## ATE - Exact Character Span

| Metric | V1 | V2 | Delta |
|---|---:|---:|---:|
| Precision | 31.07% | 54.55% | +23.48 pp |
| Recall | 51.61% | 48.39% | -3.23 pp |
| F1 | 38.79% | 51.28% | +12.49 pp |
| FP | 71 | 25 | -46 |
| FN | 30 | 32 | +2 |

## ASC - Gold Aspect / Oracle ATE

| Metric | V1 | V2 | Delta |
|---|---:|---:|---:|
| Accuracy | 67.74% | 70.97% | +3.23 pp |
| Macro-F1 | 47.78% | 61.03% | +13.25 pp |
| NEGATIVE F1 | 0.00% | 33.33% | +33.33 pp |
| NEUTRAL F1 | 76.67% | 76.67% | +0.00 pp |
| POSITIVE F1 | 66.67% | 73.08% | +6.41 pp |

## End-to-End ABSA

Exact match = `(start, end, sentiment)`.

| Metric | V1 | V2 | Delta |
|---|---:|---:|---:|
| Precision | 24.27% | 38.18% | +13.91 pp |
| Recall | 40.32% | 33.87% | -6.45 pp |
| F1 | 30.30% | 35.90% | +5.59 pp |

Frozen-test results are for final comparison, not further tuning.