# Experiment: phase4_r7_final_holdout

Folds: selection, final | seed 42 | city check: on

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | grouped_median | hgb_r3 | hgb_r3_noqs | ridge_r3_noqs | blend_70 | blend_70_noqs | final |
|---|---|---|---|---|---|---|---|
| MAE all | 156.94 | 110.32 | 112.94 | 121.31 | 109.68 | 112.20 | 112.20 |
| MAE clean | 103.66 | 56.55 | 59.14 | 67.64 | 55.89 | 58.40 | 58.40 |
| RMSE clean | 140.46 | 84.90 | 86.37 | 111.52 | 84.55 | 87.31 | 87.31 |
| MAPE% clean | 4.55 | 2.42 | 2.49 | 2.74 | 2.35 | 2.42 | 2.42 |
| bias clean | -58.09 | -39.63 | -43.97 | -38.23 | -39.52 | -42.41 | -42.41 |
| MAE clean m1 | 98.09 | 47.12 | 50.75 | 64.89 | 48.86 | 51.60 | 51.60 |
| MAE clean m2 | 109.39 | 66.39 | 67.88 | 70.65 | 63.27 | 65.54 | 65.54 |
| MAE clean fold_1 | 148.57 | 53.59 | 63.46 | 72.58 | 54.66 | 63.51 | 63.51 |
| MAE clean fold_2 | 83.16 | 52.45 | 50.15 | 54.09 | 47.91 | 46.48 | 46.48 |
| MAE clean holdout | 79.26 | 63.60 | 63.80 | 76.25 | 65.09 | 65.20 | 65.20 |
| MAE unseen city | 105.01 | 58.05 | 61.48 | 69.91 | 57.67 | 60.82 | 60.82 |
| MAE same loads, city seen | 104.98 | 55.85 | 58.62 | 67.59 | 55.54 | 58.28 | 58.28 |
| unseen-city cost % | 0.03 | 3.93 | 4.88 | 3.43 | 3.84 | 4.36 | 4.36 |
| sec per fit | 0.09 | 1.72 | 1.66 | 0.20 | 1.81 | 1.74 | 1.73 |

## Model configs

```json
{
  "models": {
    "grouped_median": {
      "type": "grouped_median",
      "n_bins": 10
    },
    "hgb_r3": {
      "extends": "hgb_r30_no_flags"
    },
    "hgb_r3_noqs": {
      "extends": "hgb_r3",
      "features_drop": [
        "quote_signal"
      ]
    },
    "ridge_r3_noqs": {
      "extends": "ridge_r3",
      "features_drop": [
        "quote_signal"
      ]
    },
    "blend_70": {
      "type": "blend",
      "members": [
        "hgb_r3",
        "ridge_r3"
      ],
      "weights": [
        0.7,
        0.3
      ]
    },
    "blend_70_noqs": {
      "type": "blend",
      "members": [
        "hgb_r3_noqs",
        "ridge_r3_noqs"
      ],
      "weights": [
        0.7,
        0.3
      ]
    },
    "final": {
      "type": "blend",
      "members": [
        "final_hgb",
        "final_ridge"
      ],
      "weights": [
        0.7,
        0.3
      ]
    }
  },
  "feature_sets": {
    "base": [
      "log_distance",
      "equipment",
      "weight",
      "weight_missing",
      "market_index",
      "market_index_missing",
      "quote_signal",
      "coords"
    ],
    "final": [
      "log_distance",
      "equipment",
      "weight",
      "market_index",
      "coords"
    ]
  }
}
```