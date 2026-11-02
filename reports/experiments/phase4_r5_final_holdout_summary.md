# Experiment: phase4_r5_final_holdout

Folds: selection, final | seed 42 | city check: on

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | grouped_median | hgb | hgb_r3 | ridge_r3 | blend_70 |
|---|---|---|---|---|---|
| MAE all | 156.94 | 120.04 | 110.32 | 121.29 | 109.68 |
| MAE clean | 103.66 | 66.45 | 56.55 | 67.61 | 55.89 |
| RMSE clean | 140.46 | 95.66 | 84.90 | 111.14 | 84.55 |
| MAPE% clean | 4.55 | 2.87 | 2.42 | 2.75 | 2.35 |
| bias clean | -58.09 | -50.13 | -39.63 | -38.57 | -39.52 |
| MAE clean m1 | 98.09 | 56.77 | 47.12 | 65.68 | 48.86 |
| MAE clean m2 | 109.39 | 76.62 | 66.39 | 69.75 | 63.27 |
| MAE clean fold_1 | 148.57 | 53.30 | 53.59 | 70.07 | 54.66 |
| MAE clean fold_2 | 83.16 | 59.27 | 52.45 | 56.66 | 47.91 |
| MAE clean holdout | 79.26 | 86.78 | 63.60 | 76.10 | 65.09 |
| MAE unseen city | 105.01 | 66.02 | 58.05 | 69.65 | 57.67 |
| MAE same loads, city seen | 104.98 | 65.36 | 55.85 | 67.48 | 55.54 |
| unseen-city cost % | 0.03 | 1.01 | 3.93 | 3.21 | 3.84 |
| sec per fit | 0.08 | 1.61 | 1.62 | 0.18 | 1.75 |

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