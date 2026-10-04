# Final model evaluation - Logistic Regression

Test rows: 77,071. Dataset period: 2021-2025.

| Metric | Value |
|---|---:|
| accuracy | 0.5057 |
| macro_precision | 0.3840 |
| macro_recall | 0.5054 |
| macro_f1 | 0.3455 |
| weighted_f1 | 0.5767 |
| balanced_accuracy | 0.5054 |
| fatal_recall | 0.6531 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Slight | 0.830 | 0.565 | 0.672 | 58,416 |
| Serious | 0.283 | 0.298 | 0.290 | 17,522 |
| Fatal | 0.039 | 0.653 | 0.074 | 1,133 |

Fatal collisions were predicted as: {'Fatal': 740, 'Serious': 232, 'Slight': 161}.
Poor Fatal performance flag: True.

## Permutation importance (families, macro-F1 drop)

- Vehicles involved: 0.0212
- Road class: 0.0076
- Special conditions / hazards: 0.0040
- Speed limit: 0.0034
- Road type: 0.0016
- Day of week: 0.0015
- Month / season: 0.0009
- Weather: 0.0008
- Time of day: 0.0001
- Junction: -0.0000
- Lighting: -0.0004
- Road surface: -0.0009
- Urban / rural: -0.0011
