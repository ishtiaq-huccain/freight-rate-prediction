# Experiment: phase4_r6_normal

Folds: selection | seed 42 | city check: on

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | grouped_median | hgb_r3 | hgb_r3_noqs | ridge_r3_noqs | blend_70 | blend_70_noqs | final |
|---|---|---|---|---|---|---|---|
| MAE all | 167.96 | 105.82 | 109.64 | 116.03 | 104.12 | 107.84 | 107.84 |
| MAE clean | 115.87 | 53.02 | 56.80 | 63.34 | 51.28 | 55.00 | 55.00 |
| RMSE clean | 156.41 | 80.77 | 83.20 | 104.01 | 77.99 | 82.40 | 82.40 |
| MAPE% clean | 4.99 | 2.32 | 2.40 | 2.54 | 2.19 | 2.29 | 2.29 |
| bias clean | -79.20 | -31.01 | -37.65 | -26.22 | -29.98 | -34.39 | -34.39 |
| MAE clean m1 | 107.19 | 35.95 | 41.39 | 54.89 | 37.44 | 41.41 | 41.41 |
| MAE clean m2 | 124.76 | 70.62 | 72.67 | 72.02 | 65.54 | 68.97 | 68.97 |
| MAE clean fold_1 | 148.57 | 53.59 | 63.46 | 72.58 | 54.66 | 63.51 | 63.51 |
| MAE clean fold_2 | 83.16 | 52.45 | 50.15 | 54.09 | 47.91 | 46.48 | 46.48 |
| MAE unseen city | 119.30 | 56.02 | 60.81 | 67.12 | 54.30 | 58.74 | 58.74 |
| MAE same loads, city seen | 119.54 | 53.78 | 57.91 | 64.44 | 52.10 | 56.14 | 56.14 |
| unseen-city cost % | -0.20 | 4.17 | 5.00 | 4.15 | 4.22 | 4.63 | 4.63 |
| sec per fit | 0.08 | 1.50 | 1.43 | 0.17 | 1.71 | 1.55 | 1.52 |

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