"""Competition 5 submission "ind3" — LAGGED multi-horizon idiosyncratic reversal.

AXIS: cross-sectional residual / relative value, but deliberately displaced OFF the
one-bar residual-reversal lane that every other object on this axis occupies.

    e_t   = Feature.1_t  -  A_t beta_t ,   A_t = [1, F2_t .. F6_t]   (per-row OLS,
            batched pseudo-inverse, so rank-deficient rows get the min-norm fit)
    S_t^K = sum_{s=1..K} e_{t-s}                      (STRICTLY TRAILING; the current
                                                       row's own residual is EXCLUDED)
    Leg_K = -csrank( S_t^K )                          K in {2, 5, 20}
    sig   = unit_row(Leg_2) + unit_row(Leg_5) + unit_row(Leg_20),  row de-meaned.

WHY THE CURRENT BAR IS EXCLUDED — this is the whole point of the object.  De-meaned
Feature.1 IS the contemporaneous return, so e_t is the idiosyncratic return of bar t
and -csrank(e_t) is a one-bar reversal on the residual: that is exactly the known
residual baseline (t8_resid_allfeat, Sharpe 0.0765) and it sits in the same reversal
lane as apex6.  Every variation of the FIT (robust / Huber / weighted / PCA /
rank-space / regressor subsets) perturbs e_t by a few percent and reproduces that
lane at corr 0.85-0.98 — a strictly worse copy.  The only lever on this axis that
buys INDEPENDENCE is the HORIZON: which residual rows enter the signal.  Sharing no
residual row at all with the baseline, this object measures the multi-bar
idiosyncratic spread, and the measurement bears that out (PnL corr vs
t8_resid_allfeat -0.064, vs apex6 +0.250, vs ticket2 +0.003, vs unique -0.003).

The sign is NEGATIVE (mean-reversion of the idiosyncratic component) by declaration,
the classical relative-value prior, not by reading the data.

NO FITTED WEIGHTS, NO ARGMAX.  The K ladder {2, 5, 20} is a fixed dyadic ladder
declared before measuring; the three legs are equalised to unit cross-sectional
dispersion per row and summed 1:1.  train() fits nothing — it only stores the last
20 training residual rows so the first validation bars have a full trailing window.

CAUSALITY: the prediction for row t is a function of residual rows t-20 .. t-1 ONLY
(it does not even use row t), and the trailing sums are formed from a cumulative sum
over [warm-up buffer ; block], so a prefix of the block reproduces the same values
bit-identically.  csrank and de-mean are per-row cross-sectional operations.  No
whole-block statistic, no backward fill, no negative shift.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ind3ResidualHorizonPredictor(Predictor):
    """Lagged multi-horizon cross-sectional idiosyncratic reversal."""

    KS = (2, 5, 20)          # fixed dyadic horizon ladder (declared, not searched)
    WARM = 20                # = max(KS): rows of residual history kept from train()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _resid(features, tk):
        """Per-row cross-sectional OLS residual of Feature.1 on [1, F2..F6]."""
        y = np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))
        cols = [np.ones_like(y)] + [
            np.nan_to_num(features["Feature.%d" % i][tk].to_numpy(dtype=np.float64))
            for i in range(2, 7)
        ]
        a = np.stack(cols, axis=2)
        beta = np.einsum("tij,tj->ti", np.linalg.pinv(a), y)   # min-norm OLS
        return y - np.einsum("tji,ti->tj", a, beta)

    @staticmethod
    def _csrank_c(a):
        """Cross-sectional rank of each row mapped to [-1, +1]."""
        j = a.shape[1]
        r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
        return (r - 0.5 * (j - 1.0)) / (0.5 * (j - 1.0))

    @staticmethod
    def _unit_row(a):
        """Row de-mean, then scale each row to unit cross-sectional dispersion."""
        a = a - a.mean(axis=1, keepdims=True)
        return a / np.maximum(a.std(axis=1, keepdims=True), 1e-12)

    # -------------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """WARM-UP STATE ONLY: the last WARM residual rows of the training block.

        Nothing is fitted, no scale is measured, the target is not used.
        """
        tk = self._tickers(features)
        e = self._resid(features, tk)
        buf = e[-self.WARM:]
        if len(buf) < self.WARM:                      # pad short history with zeros
            buf = np.vstack([np.zeros((self.WARM - len(buf), e.shape[1])), buf])
        self.tk_ = tk
        self.buf_ = buf
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        e = self._resid(features, tk)
        n, j = e.shape

        buf = getattr(self, "buf_", None)
        if buf is None or buf.shape[1] != j:
            buf = np.zeros((self.WARM, j))
        ee = np.vstack([buf, e])

        # C[i] = sum of the first i rows of ee.  Strictly trailing windows are
        # differences of C, so predicting a prefix of the block gives bit-identical
        # values on the overlap.
        c = np.vstack([np.zeros((1, j)), np.cumsum(ee, axis=0)])
        hi = self.WARM + np.arange(n)

        sig = np.zeros((n, j))
        for k in self.KS:
            s = c[hi] - c[hi - k]                     # sum_{s=1..k} e_{t-s}
            sig += self._unit_row(self._csrank_c(-s))

        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk).fillna(0.0)
