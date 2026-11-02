# Experiment: phase4_r4_finalists

Folds: selection | seed 42 | city check: on

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | grouped_median | hgb | hgb_r3 | ridge_r3 | blend_70 |
|---|---|---|---|---|---|
| MAE all | 167.96 | 109.00 | 105.82 | 116.07 | 104.12 |
| MAE clean | 115.87 | 56.28 | 53.02 | 63.37 | 51.28 |
| RMSE clean | 156.41 | 85.91 | 80.77 | 103.39 | 77.99 |
| MAPE% clean | 4.99 | 2.47 | 2.32 | 2.56 | 2.19 |
| bias clean | -79.20 | -32.53 | -31.01 | -26.81 | -29.98 |
| MAE clean m1 | 107.19 | 37.10 | 35.95 | 56.78 | 37.44 |
| MAE clean m2 | 124.76 | 76.05 | 70.62 | 70.14 | 65.54 |
| MAE clean fold_1 | 148.57 | 53.30 | 53.59 | 70.07 | 54.66 |
| MAE clean fold_2 | 83.16 | 59.27 | 52.45 | 56.66 | 47.91 |
| MAE unseen city | 119.30 | 57.97 | 56.02 | 66.82 | 54.30 |
| MAE same loads, city seen | 119.54 | 57.11 | 53.78 | 64.38 | 52.10 |
| unseen-city cost % | -0.20 | 1.51 | 4.17 | 3.78 | 4.22 |
| sec per fit | 0.07 | 1.50 | 1.51 | 0.15 | 1.63 |

## Model configs

```json
{
  "models": {
    "grouped_median": {
      "type": "grouped_median",
      "n_bins": 10
    },
    "hgb": {
      "type": "hgb",
      "features": "base",
      "params": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "early_stopping": false
      }
    },
    "hgb_r3": {
      "extends": "hgb_r30_no_flags"
    },
    "ridge_r3": {
      "extends": "ridge",
      "features_drop": [
        "weight_missing",
        "market_index_missing"
      ],
      "recency_half_life_days": 30
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
    ]
  }
}
```