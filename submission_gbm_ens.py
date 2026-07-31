# /// script
# dependencies = ["catboost>=1.2"]
# ///
"""Competition 5 — basic gradient-boosted cross-sectional ranker (4-booster ensemble).

A deliberately SIMPLE baseline: rank-transform the six features cross-sectionally,
add a few causal rolling summaries of Feature.1, fit one gradient-boosted tree
model to the cross-sectionally ranked target, then csrank + de-mean the output.

No blending, no hand-tuned constants beyond the model hyper-parameters, no
whole-block statistics.  Every feature is either a per-row cross-sectional rank
(row-local) or a trailing rolling window, so the predictor is causal by
construction and invariant to how the scored block is chunked.

Training uses only the rows handed to train(); the last TAIL rows are used so the
fit reflects the most recent regime without any look-ahead.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from predictor import Predictor


class GbmEnsemblePredictor(Predictor):
    """Equal-weight ensemble of LightGBM + XGBoost + CatBoost + AdaBoost.

    Each booster is fit on the SAME rank features and the SAME ranked target,
    its raw output is cross-sectionally ranked, and the four ranked views are
    averaged 1:1.  Ranking before averaging makes the combination scale-free,
    so no booster can dominate through output magnitude.  No fitted weights.
    """

    TAIL = 20000      # training rows retained (most recent)
    ROUNDS = 200
    LR = 0.05
    SEED = 7

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _mat(features, i, tk):
        return np.nan_to_num(
            features[f"Feature.{i}"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _csr(a):
        """Per-row cross-sectional rank mapped to [-1, +1]. Row-local => causal."""
        j = a.shape[1]
        r = pd.DataFrame(a).rank(axis=1, method="average").to_numpy()
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @classmethod
    def _feat(cls, features):
        """(T, J, K) float32 stack of causal features + the (T, J) shape."""
        tk = cls._tickers(features)
        j = len(tk)
        M = {i: cls._mat(features, i, tk) for i in range(1, 7)}
        C = [cls._csr(M[i]) for i in range(1, 7)]           # r1..r6
        f1 = pd.DataFrame(M[1])
        for w in (5, 20):                                    # trailing sums of F1
            C.append(cls._csr(f1.rolling(w, min_periods=1).sum().to_numpy()))
        C.append(cls._csr(                                   # trailing vol of F1
            f1.rolling(20, min_periods=2).std().fillna(0.0).to_numpy()))
        C.append(cls._csr(M[3] - M[4]))                      # F3/F4 disagreement
        X = np.stack(C, axis=2).astype(np.float32)           # (T, J, K)
        return X, tk, j

    # ------------------------------------------------------------------- API
    def __init__(self):
        self.models = None

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tk, j = self._feat(features)
        y = self._csr(np.nan_to_num(target[tk].to_numpy(dtype=np.float64)))
        T = X.shape[0]
        lo = max(0, T - self.TAIL)
        Xf = X[lo:].reshape(-1, X.shape[2])
        yf = y[lo:].reshape(-1).astype(np.float32)
        good = np.isfinite(Xf).all(1) & np.isfinite(yf)
        Xg, yg = Xf[good], yf[good]
        self.models = []
        self.models.append(("lgb", lgb.train(
            dict(objective="regression", num_leaves=15, max_depth=4,
                 learning_rate=self.LR, min_data_in_leaf=500, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=2,
                 verbosity=-1, seed=self.SEED, force_row_wise=True),
            lgb.Dataset(Xg, label=yg), num_boost_round=self.ROUNDS)))
        m = xgb.XGBRegressor(n_estimators=self.ROUNDS, max_depth=4, learning_rate=self.LR,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=50,
            random_state=self.SEED, n_jobs=2, tree_method="hist", verbosity=0)
        m.fit(Xg, yg); self.models.append(("xgb", m))
        m = CatBoostRegressor(iterations=self.ROUNDS, depth=4, learning_rate=self.LR,
            l2_leaf_reg=5.0, random_seed=self.SEED, thread_count=2, verbose=False,
            allow_writing_files=False)
        m.fit(Xg, yg); self.models.append(("cat", m))
        m = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=200),
            n_estimators=self.ROUNDS, learning_rate=self.LR, random_state=self.SEED)
        m.fit(Xg, yg); self.models.append(("ada", m))
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tk, j = self._feat(features)
        T = X.shape[0]
        flat = X.reshape(-1, X.shape[2])
        acc = None
        for nm, mdl in self.models:
            r = (mdl.predict(flat, num_threads=2) if nm == "lgb" else mdl.predict(flat))
            rr = self._csr(np.asarray(r, dtype=np.float64).reshape(T, j))
            acc = rr if acc is None else acc + rr
        raw = (acc / len(self.models)).reshape(-1)
        p = raw.reshape(T, j)
        out = self._csr(p)
        out = out - out.mean(axis=1, keepdims=True)
        return pd.DataFrame(out, index=features.index, columns=tk).fillna(0.0)
