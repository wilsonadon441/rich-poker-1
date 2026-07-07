"""Variant identity for this miner deployment.

rich-poker-1 runs the "boost" variant: a gradient-boosting-heavy stack
(LightGBM + XGBoost + ExtraTrees) over the baseline chunk-feature space with a
hand-ngram side model, calibrated holdout-first against the validator reward.
"""
from __future__ import annotations

VARIANT: dict = {
    "key": "boost",
    "name": "rich-poker-1",
    "framework": "stack-lgb-xgb-et-hgram-boost-r1",
    "seed": 1013,
    "cv_folds": 5,
    "human_weight": 25.0,
    "recency_boost": 6.0,
    "recent_days": 8,
    "holdout_days": 2,
    "quantile_blend": 0.0,
    "hgram_lgb_weight": 0.6,
    "hgram_min_token": 40,
    "meta_c": 1.0,
    "description": (
        "Gradient-boosting-heavy stacked ensemble (LightGBM, XGBoost, "
        "ExtraTrees) with logistic meta-learner and hand-ngram side features "
        "over the baseline aggregate chunk-feature space."
    ),
}
