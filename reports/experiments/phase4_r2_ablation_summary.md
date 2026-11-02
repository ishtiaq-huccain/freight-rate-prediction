# Experiment: phase4_r2_ablation

Folds: selection | seed 42 | city check: off

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | hgb_recency_30 | hgb_r30_no_quote | hgb_r30_no_coords | hgb_r30_no_weight | hgb_r30_no_flags | hgb_r30_no_market | hgb_r30_dow | hgb_r30_abs_loss | hgb_r30_keep_anomalies | hgb_r30_keep_anomalies_abs |
|---|---|---|---|---|---|---|---|---|---|---|
| MAE all | 105.76 | 109.62 | 113.15 | 121.00 | 105.82 | 123.42 | 106.59 | 109.89 | 133.45 | 110.24 |
| MAE clean | 52.97 | 56.79 | 60.51 | 68.46 | 53.02 | 70.90 | 53.82 | 57.18 | 80.91 | 57.54 |
| RMSE clean | 80.71 | 83.22 | 91.99 | 107.66 | 80.77 | 108.29 | 81.67 | 91.73 | 121.07 | 91.60 |
| MAPE% clean | 2.32 | 2.40 | 2.73 | 2.93 | 2.32 | 2.94 | 2.37 | 2.44 | 3.55 | 2.47 |
| bias clean | -31.12 | -37.56 | -25.69 | -25.24 | -31.01 | -17.10 | -32.73 | -24.83 | -32.40 | -25.76 |
| MAE clean m1 | 35.96 | 41.40 | 39.50 | 42.89 | 35.95 | 46.86 | 36.20 | 36.62 | 72.65 | 36.72 |
| MAE clean m2 | 70.48 | 72.62 | 82.17 | 94.81 | 70.62 | 95.70 | 71.96 | 78.37 | 89.43 | 79.00 |
| MAE clean fold_1 | 53.58 | 63.46 | 55.18 | 63.17 | 53.59 | 69.46 | 53.53 | 56.16 | 87.35 | 56.76 |
| MAE clean fold_2 | 52.36 | 50.12 | 65.84 | 73.74 | 52.45 | 72.34 | 54.10 | 58.21 | 74.47 | 58.32 |
| MAE unseen city | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| MAE same loads, city seen | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| unseen-city cost % | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| sec per fit | 1.99 | 1.72 | 1.95 | 1.65 | 1.74 | 1.70 | 1.77 | 5.18 | 1.74 | 5.21 |

## Model configs

```json
{
  "models": {
    "hgb_recency_30": {
      "extends": "hgb",
      "recency_half_life_days": 30
    },
    "hgb_r30_no_quote": {
      "extends": "hgb_recency_30",
      "features_drop": [
        "quote_signal"
      ]
    },
    "hgb_r30_no_coords": {
      "extends": "hgb_recency_30",
      "features_drop": [
        "coords"
      ]
    },
    "hgb_r30_no_weight": {
      "extends": "hgb_recency_30",
      "features_drop": [
        "weight",
        "weight_missing"
      ]
    },
    "hgb_r30_no_flags": {
      "extends": "hgb_recency_30",
      "features_drop": [
        "weight_missing",
        "market_index_missing"
      ]
    },
    "hgb_r30_no_market": {
      "extends": "hgb_recency_30",
      "features_drop": [
        "market_index",
        "market_index_missing"
      ]
    },
    "hgb_r30_dow": {
      "extends": "hgb_recency_30",
      "features_add": [
        "day_of_week"
      ]
    },
    "hgb_r30_abs_loss": {
      "extends": "hgb_recency_30",
      "params": {
        "loss": "absolute_error"
      }
    },
    "hgb_r30_keep_anomalies": {
      "extends": "hgb_recency_30",
      "drop_anomalies": false
    },
    "hgb_r30_keep_anomalies_abs": {
      "extends": "hgb_r30_keep_anomalies",
      "params": {
        "loss": "absolute_error"
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