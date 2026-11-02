# Experiment: phase3_baselines

Folds: selection | seed 42 | city check: on

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | global_median | grouped_median | ridge | hgb |
|---|---|---|---|---|
| MAE all | 245.40 | 167.96 | 114.28 | 109.00 |
| MAE clean | 194.49 | 115.87 | 61.52 | 56.28 |
| RMSE clean | 267.38 | 156.41 | 91.40 | 85.91 |
| MAPE% clean | 9.42 | 4.99 | 2.62 | 2.47 |
| bias clean | 10.12 | -79.20 | -36.50 | -32.53 |
| MAE clean m1 | 184.65 | 107.19 | 52.52 | 37.10 |
| MAE clean m2 | 204.63 | 124.76 | 70.79 | 76.05 |
| MAE clean fold_1 | 189.83 | 148.57 | 63.96 | 53.30 |
| MAE clean fold_2 | 199.15 | 83.16 | 59.09 | 59.27 |
| MAE unseen city | 203.77 | 119.30 | 63.04 | 57.97 |
| MAE same loads, city seen | 202.35 | 119.54 | 61.92 | 57.11 |
| unseen-city cost % | 0.70 | -0.20 | 1.82 | 1.51 |
| sec per fit | 0.07 | 0.08 | 0.15 | 1.50 |

## Model configs

```json
{
  "models": {
    "global_median": {
      "type": "global_median"
    },
    "grouped_median": {
      "type": "grouped_median",
      "n_bins": 10
    },
    "ridge": {
      "type": "ridge",
      "features": "base",
      "alpha": 1.0,
      "spline_knots": 6
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