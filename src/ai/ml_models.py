"""
ML Models: XGBoost and LightGBM for price direction prediction.

Trains on historical OHLCV + technical indicators to predict
next-day price direction (1 = up, 0 = down).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# Feature columns used for ML models
FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_pct", "BB_width",
    "ATR", "ADX", "STOCH_K", "STOCH_D",
    "EMA_9", "EMA_21", "SMA_50",
    "Returns", "Volatility_20",
    "Golden_Cross", "Price_above_EMA21",
    "Volume",
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y from indicator DataFrame.

    Target: next-day close > current close (binary classification).
    """
    data = df.copy()
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    cols = [c for c in FEATURE_COLS if c in data.columns]
    X = data[cols].copy()
    y = data["target"].copy()

    # Drop last row (no next-day target) and rows with NaN
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    return X, y


class MLSignalModel:
    """Wraps XGBoost + LightGBM ensemble for directional signal generation."""

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.is_trained = False
        self.last_accuracy: float = 0.0
        self.last_report: str = ""

    def train(self, df: pd.DataFrame) -> dict:
        """Train models on the given DataFrame. Returns training metrics."""
        X, y = prepare_features(df)

        if len(X) < 60:
            return {"error": "Not enough data to train (need 60+ rows after indicators)"}

        self.feature_cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        xgb_scores, lgb_scores = [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if XGB_AVAILABLE:
                m = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, verbosity=0,
                )
                m.fit(X_tr, y_tr)
                xgb_scores.append(accuracy_score(y_val, m.predict(X_val)))

            if LGB_AVAILABLE:
                m2 = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1,
                )
                m2.fit(X_tr, y_tr)
                lgb_scores.append(accuracy_score(y_val, m2.predict(X_val)))

        # Final fit on all data
        if XGB_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            self.xgb_model.fit(X_scaled, y)

        if LGB_AVAILABLE:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
            self.lgb_model.fit(X_scaled, y)

        self.is_trained = True

        avg_xgb = float(np.mean(xgb_scores)) if xgb_scores else 0.0
        avg_lgb = float(np.mean(lgb_scores)) if lgb_scores else 0.0
        self.last_accuracy = max(avg_xgb, avg_lgb)

        return {
            "xgb_cv_accuracy": round(avg_xgb, 4) if xgb_scores else "N/A",
            "lgb_cv_accuracy": round(avg_lgb, 4) if lgb_scores else "N/A",
            "samples_trained": len(X),
            "features": self.feature_cols,
        }

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """Predict signal for the latest row of df.

        Returns:
            direction: 1 (bullish) or 0 (bearish)
            probability: float 0-1 (probability of up move)
            signal: 'BUY' | 'SELL' | 'HOLD'
            confidence: int 0-100
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}

        cols = [c for c in self.feature_cols if c in df.columns]
        latest = df[cols].iloc[-1:].copy()
        if latest.isna().any(axis=1).iloc[0]:
            return {"error": "Latest row has NaN features — need more history"}

        X = self.scaler.transform(latest)

        probs = []
        if self.xgb_model:
            p = self.xgb_model.predict_proba(X)[0][1]
            probs.append(p)
        if self.lgb_model:
            p = self.lgb_model.predict_proba(X)[0][1]
            probs.append(p)

        if not probs:
            return {"error": "No models available"}

        avg_prob = float(np.mean(probs))
        direction = 1 if avg_prob >= 0.5 else 0

        # Map to signal
        if avg_prob > 0.62:
            signal = "BUY"
        elif avg_prob < 0.38:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = int(abs(avg_prob - 0.5) * 200)  # 0–100 scale

        return {
            "direction": direction,
            "probability_up": round(avg_prob, 4),
            "signal": signal,
            "confidence": confidence,
            "model_accuracy": round(self.last_accuracy, 4),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance from XGBoost (if available)."""
        if self.xgb_model is None or not self.feature_cols:
            return pd.DataFrame()
        imp = self.xgb_model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# Module-level cache: one model per ticker
_model_cache: dict[str, MLSignalModel] = {}


def get_or_train_model(ticker: str, df: pd.DataFrame) -> Tuple[MLSignalModel, dict]:
    """Get cached model or train a new one for the ticker."""
    if ticker not in _model_cache:
        model = MLSignalModel()
        metrics = model.train(df)
        _model_cache[ticker] = model
        return model, metrics
    return _model_cache[ticker], {}


def clear_model_cache(ticker: str = None):
    """Clear cached model(s) to force retraining."""
    global _model_cache
    if ticker:
        _model_cache.pop(ticker, None)
    else:
        _model_cache.clear()
