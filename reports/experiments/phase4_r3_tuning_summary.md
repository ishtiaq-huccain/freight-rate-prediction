# Experiment: phase4_r3_tuning

Folds: selection | seed 42 | city check: off

MAE/RMSE/bias in USD, mean over folds. 'clean' excludes test rows flagged as corrupted labels. m1/m2 = first/second month of each 2-month test window.

|  | hgb_r3 | hgb_r3_lr10 | hgb_r3_iter600 | hgb_r3_lr03_iter1000 | hgb_r3_leaves15 | hgb_r3_leaves63 | hgb_r3_minleaf20 | hgb_r3_minleaf100 | hgb_r3_l2 | hgb_r3_hl15 | hgb_r3_hl45 | ridge_r3 | blend_50 | blend_70 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAE all | 105.82 | 106.29 | 105.80 | 105.20 | 107.34 | 105.95 | 105.41 | 105.49 | 105.54 | 106.41 | 106.55 | 116.07 | 105.61 | 104.12 |
| MAE clean | 53.02 | 53.52 | 53.02 | 52.42 | 54.52 | 53.18 | 52.63 | 52.72 | 52.75 | 53.64 | 53.77 | 63.37 | 52.79 | 51.28 |
| RMSE clean | 80.77 | 81.48 | 81.16 | 80.56 | 81.51 | 82.56 | 80.12 | 80.68 | 81.06 | 82.35 | 82.17 | 103.39 | 81.23 | 77.99 |
| MAPE% clean | 2.32 | 2.36 | 2.34 | 2.31 | 2.35 | 2.35 | 2.30 | 2.33 | 2.30 | 2.32 | 2.37 | 2.56 | 2.21 | 2.19 |
| bias clean | -31.01 | -28.34 | -28.19 | -27.77 | -35.31 | -28.55 | -30.84 | -30.15 | -29.69 | -25.80 | -32.20 | -26.81 | -29.18 | -29.98 |
| MAE clean m1 | 35.95 | 35.68 | 34.99 | 34.78 | 39.47 | 35.02 | 35.81 | 35.99 | 35.50 | 36.48 | 35.86 | 56.78 | 41.20 | 37.44 |
| MAE clean m2 | 70.62 | 71.90 | 71.60 | 70.61 | 70.02 | 71.91 | 69.95 | 69.94 | 70.52 | 71.32 | 72.20 | 70.14 | 64.70 | 65.54 |
| MAE clean fold_1 | 53.59 | 50.90 | 51.01 | 50.56 | 57.41 | 51.07 | 53.21 | 53.22 | 53.29 | 55.34 | 53.99 | 70.07 | 57.66 | 54.66 |
| MAE clean fold_2 | 52.45 | 56.14 | 55.04 | 54.28 | 51.63 | 55.29 | 52.04 | 52.21 | 52.21 | 51.95 | 53.54 | 56.66 | 47.91 | 47.91 |
| MAE unseen city | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| MAE same loads, city seen | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| unseen-city cost % | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| sec per fit | 1.79 | 1.56 | 2.98 | 4.80 | 1.28 | 2.57 | 1.71 | 1.81 | 1.77 | 1.76 | 1.70 | 0.20 | 2.12 | 1.92 |

## Model configs

```json
{
  "models": {
    "hgb_r3": {
      "extends": "hgb_r30_no_flags"
    },
    "hgb_r3_lr10": {
      "extends": "hgb_r3",
      "params": {
        "learning_rate": 0.1
      }
    },
    "hgb_r3_iter600": {
      "extends": "hgb_r3",
      "params": {
        "max_iter": 600
      }
    },
    "hgb_r3_lr03_iter1000": {
      "extends": "hgb_r3",
      "params": {
        "learning_rate": 0.03,
        "max_iter": 1000
      }
    },
    "hgb_r3_leaves15": {
      "extends": "hgb_r3",
      "params": {
        "max_leaf_nodes": 15
      }
    },
    "hgb_r3_leaves63": {
      "extends": "hgb_r3",
      "params": {
        "max_leaf_nodes": 63
      }
    },
    "hgb_r3_minleaf20": {
      "extends": "hgb_r3",
      "params": {
        "min_samples_leaf": 20
      }
    },
    "hgb_r3_minleaf100": {
      "extends": "hgb_r3",
      "params": {
        "min_samples_leaf": 100
      }
    },
    "hgb_r3_l2": {
      "extends": "hgb_r3",
      "params": {
        "l2_regularization": 1.0
      }
    },
    "hgb_r3_hl15": {
      "extends": "hgb_r3",
      "recency_half_life_days": 15
    },
    "hgb_r3_hl45": {
      "extends": "hgb_r3",
      "recency_half_life_days": 45
    },
    "ridge_r3": {
      "extends": "ridge",
      "features_drop": [
        "weight_missing",
        "market_index_missing"
      ],
      "recency_half_life_days": 30
    },
    "blend_50": {
      "type": "blend",
      "members": [
        "hgb_r3",
        "ridge_r3"
      ],
      "weights": [
        0.5,
        0.5
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