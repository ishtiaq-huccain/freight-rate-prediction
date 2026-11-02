# Experiment: phase4_r1_drift

Folds: selection | seed 42 | city check: off

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | hgb | hgb_time | hgb_recency_30 | hgb_recency_60 | hgb_recency_120 | hgb_time_recency_60 | ridge | ridge_time | ridge_recency_60 |
|---|---|---|---|---|---|---|---|---|---|
| MAE all | 109.00 | 129.60 | 105.76 | 106.91 | 107.74 | 130.46 | 114.28 | 125.36 | 115.28 |
| MAE clean | 56.28 | 77.19 | 52.97 | 54.13 | 54.99 | 78.03 | 61.52 | 72.86 | 62.55 |
| RMSE clean | 85.91 | 123.36 | 80.71 | 82.27 | 84.35 | 124.73 | 91.40 | 104.33 | 97.80 |
| MAPE% clean | 2.47 | 2.99 | 2.32 | 2.39 | 2.42 | 3.03 | 2.62 | 3.04 | 2.60 |
| bias clean | -32.53 | 12.87 | -31.12 | -32.67 | -32.50 | 11.76 | -36.50 | 43.87 | -33.42 |
| MAE clean m1 | 37.10 | 54.70 | 35.96 | 36.50 | 36.60 | 56.75 | 52.52 | 62.37 | 54.47 |
| MAE clean m2 | 76.05 | 100.33 | 70.48 | 72.29 | 73.94 | 99.94 | 70.79 | 83.69 | 70.85 |
| MAE clean fold_1 | 53.30 | 62.80 | 53.58 | 53.93 | 53.16 | 62.73 | 63.96 | 52.39 | 66.95 |
| MAE clean fold_2 | 59.27 | 91.57 | 52.36 | 54.33 | 56.82 | 93.34 | 59.09 | 93.33 | 58.14 |
| MAE unseen city | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| MAE same loads, city seen | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| unseen-city cost % | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| sec per fit | 1.70 | 1.87 | 1.98 | 1.85 | 1.84 | 1.95 | 0.20 | 0.19 | 0.20 |

## Model configs

```json
{
  "models": {
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
    "hgb_time": {
      "extends": "hgb",
      "features_add": [
        "time_index"
      ]
    },
    "hgb_recency_30": {
      "extends": "hgb",
      "recency_half_life_days": 30
    },
    "hgb_recency_60": {
      "extends": "hgb",
      "recency_half_life_days": 60
    },
    "hgb_recency_120": {
      "extends": "hgb",
      "recency_half_life_days": 120
    },
    "hgb_time_recency_60": {
      "extends": "hgb_time",
      "recency_half_life_days": 60
    },
    "ridge": {
      "type": "ridge",
      "features": "base",
      "alpha": 1.0,
      "spline_knots": 6
    },
    "ridge_time": {
      "extends": "ridge",
      "features_add": [
        "time_index"
      ]
    },
    "ridge_recency_60": {
      "extends": "ridge",
      "recency_half_life_days": 60
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