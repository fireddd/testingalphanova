"""Competition 5 — AXIS 2: a learned model fitted to a DIFFERENT label.

Every other learned leg in this pool is fitted to something that contains the 1-bar
forward return: either the provided `target` (the de-meaned cumulative return over bars
t+1..t+5) or the 1-bar label (the next row's Feature.1, which IS the 1-bar forward
return).  Both put the immediate cross-sectional mean-reversion term front and centre,
which is why the learned legs all end up in the same lane.

This model is fitted to the tail of the forward path with the FIRST BAR DELETED:

    y_t = csrank( F1_{t+2} + F1_{t+3} + F1_{t+4} + F1_{t+5} )

i.e. the de-meaned cumulative return from t+1 to t+5 with the t -> t+1 bar removed.  The
scored quantity, r_{t+1}, does not appear in the label at all, so the tree cannot be
rewarded for short-horizon reversal; what it can learn is the persistent cross-sectional
drift that still shows up bar by bar.  The provided `target` is never used.

The label is assembled from forward shifts of Feature.1 entirely inside the rows handed
to train().  Nothing in predict() looks forward: every input is either a per-row
cross-sectional rank (row-local) or a strictly trailing rolling window, so the predictor
is causal by construction and invariant to how the scored block is chunked.

Feature set is deliberately slow.  The bare cross-sectional rank of Feature.1 — the
reversal lane's engine — is excluded from the inputs; Feature.1 enters only through
trailing sums.  The final score is additionally neutralised, row by row, against
csrank(Feature.1_t), so the leg carries zero loading on the contemporaneous-return
direction that every fast leg in this pool trades.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from predictor import Predictor


class LaggedPathPredictor(Predictor):
    """LightGBM on the lag-2..5 forward path -> csrank -> row de-mean."""

    LAG_LO = 2          # first forward bar included in the label
    LAG_HI = 5          # last forward bar included in the label
    TAIL = 20000        # most recent training rows retained
    ROUNDS = 250
    LEAVES = 15
    LR = 0.05
    MIN_CHILD = 200
    SEED = 31

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
        """(T, J, K) float32 stack of causal features, plus tickers and J."""
        tk = cls._tickers(features)
        j = len(tk)
        M = {i: cls._mat(features, i, tk) for i in range(1, 7)}
        f1 = pd.DataFrame(M[1])
        C = []
        for w in (5, 20, 60):                       # multi-scale trailing momentum
            C.append(cls._csr(f1.rolling(w, min_periods=1).sum().to_numpy()))
        C.append(cls._csr(                          # own trailing vol state
            f1.rolling(20, min_periods=2).std().fillna(0.0).to_numpy()))
        for i in (2, 3, 4, 5, 6):                   # row-local level of the others
            C.append(cls._csr(M[i]))
        for i in (2, 5, 6):                         # slow level of the odd features
            C.append(cls._csr(
                pd.DataFrame(M[i]).rolling(20, min_periods=1).mean().to_numpy()))
        X = np.stack(C, axis=2).astype(np.float32)  # (T, J, K)
        return X, tk, j

    # ------------------------------------------------------------------- API
    def __init__(self):
        self.model = None

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tk, j = self._feat(features)
        T = X.shape[0]

        # Label: de-meaned cumulative return over forward bars t+LAG_LO .. t+LAG_HI.
        # Feature.1 IS the contemporaneous de-meaned return (corr 1.000), so summing
        # its forward shifts reconstructs the forward path.  Deleting lag 1 removes the
        # scored 1-bar horizon from the label, which is the entire point of this leg.
        # Train-time only: the shifts live inside the rows handed to train(), and
        # predict() never touches them.
        f1 = self._mat(features, 1, tk)
        acc = np.zeros_like(f1)
        for k in range(self.LAG_LO, self.LAG_HI + 1):
            sh = np.zeros_like(f1)
            if k < T:
                sh[:T - k] = f1[k:]
            acc += sh
        y = self._csr(acc)

        lo = max(0, T - self.TAIL)
        Xf = X[lo:].reshape(-1, X.shape[2])
        yf = y[lo:].reshape(-1).astype(np.float32)
        good = np.isfinite(Xf).all(1) & np.isfinite(yf)
        # the last LAG_HI rows have an incomplete forward path -> drop them
        n_drop = min(self.LAG_HI * j, good.size)
        good[good.size - n_drop:] = False

        self.model = LGBMRegressor(
            n_estimators=self.ROUNDS, num_leaves=self.LEAVES,
            learning_rate=self.LR, min_child_samples=self.MIN_CHILD,
            reg_lambda=5.0, random_state=self.SEED, n_jobs=2,
            deterministic=True, force_row_wise=True, verbose=-1)
        self.model.fit(Xf[good], yf[good])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tk, j = self._feat(features)
        T = X.shape[0]
        raw = self.model.predict(X.reshape(-1, X.shape[2]))
        out = self._csr(np.asarray(raw, dtype=np.float64).reshape(T, j))
        # Row-local neutralisation against the contemporaneous-return direction.
        # csrank(Feature.1) is the axis every fast leg in this pool loads on; removing
        # the component of the score along it leaves only the slow content the lag-2..5
        # label was built to isolate. Purely a per-row projection (row t only), so it
        # preserves causality and chunk-invariance, and it costs Sharpe rather than
        # buying it — this is an independence device, not a fit.
        q = self._csr(self._mat(features, 1, tk))
        num = (out * q).sum(axis=1, keepdims=True)
        den = (q * q).sum(axis=1, keepdims=True)
        out = out - np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0) * q
        out = out - out.mean(axis=1, keepdims=True)
        return pd.DataFrame(out, index=features.index, columns=tk).fillna(0.0)
