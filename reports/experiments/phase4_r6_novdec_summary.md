# Experiment: phase4_r6_novdec

Folds: selection | seed 42 | city check: on | test columns shuffled: quote_signal

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | grouped_median | hgb_r3 | hgb_r3_noqs | ridge_r3_noqs | blend_70 | blend_70_noqs | final |
|---|---|---|---|---|---|---|---|
| MAE all | 167.96 | 127.52 | 109.64 | 116.03 | 118.11 | 107.84 | 107.84 |
| MAE clean | 115.87 | 74.97 | 56.80 | 63.34 | 65.44 | 55.00 | 55.00 |
| RMSE clean | 156.41 | 104.54 | 83.20 | 104.01 | 94.07 | 82.40 | 82.40 |
| MAPE% clean | 4.99 | 3.49 | 2.40 | 2.54 | 2.95 | 2.29 | 2.29 |
| bias clean | -79.20 | -40.34 | -37.65 | -26.22 | -37.21 | -34.39 | -34.39 |
| MAE clean m1 | 107.19 | 64.13 | 41.39 | 54.89 | 53.97 | 41.41 | 41.41 |
| MAE clean m2 | 124.76 | 86.12 | 72.67 | 72.02 | 77.24 | 68.97 | 68.97 |
| MAE clean fold_1 | 148.57 | 83.20 | 63.46 | 72.58 | 73.79 | 63.51 | 63.51 |
| MAE clean fold_2 | 83.16 | 66.75 | 50.15 | 54.09 | 57.10 | 46.48 | 46.48 |
| MAE unseen city | 118.02 | 79.73 | 61.14 | 71.43 | 69.80 | 59.37 | 59.37 |
| MAE same loads, city seen | 117.89 | 76.65 | 58.92 | 68.08 | 66.99 | 57.15 | 57.15 |
| unseen-city cost % | 0.11 | 4.01 | 3.78 | 4.92 | 4.19 | 3.90 | 3.90 |
| sec per fit | 0.07 | 1.72 | 1.67 | 0.16 | 1.87 | 1.75 | 1.68 |

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