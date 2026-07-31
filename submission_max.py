"""Competition 5 submission: max-Sharpe reversal-cluster candidate.

Fixed-weight combination selected offline by deterministic greedy forward
selection + Nelder-Mead re-polish (enumerated grid, name-sorted tie-breaks,
no RNG), objective = full-sample Sharpe + 0.5*min(first-half, second-half)
Sharpe, with anomalous periods 076/077 sealed out of selection.

Composition: convex reversal core (extreme movers), big-move gate, volume- and
idio-conditioned reversal interactions, a vol-regime reversal/momentum mixture,
an F6-state fade blend, and the F4-F1 spread.

train() is a no-op: nothing is fit at run time. All operations are causal
(rolling / expanding / ewm / cumsum with min_periods); no future access.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class MaxSharpePredictor(Predictor):
    """Fixed-weight max-Sharpe combination (reversal-cluster upgrade)."""

    BETA_W = 250  # rolling window for market-beta estimation

    WEIGHTS = {
        "rev_convex": 0.087539,
        "rev_bigmove": -0.054032,
        "vm_nov60": 0.051471,
        "rev1_x_vol": -0.037949,
        "idio_rev_x_f5": -0.035091,
        "mix_vol_rev_mom": -0.035084,
        "idio_rev1": 0.031424,
        "f4_minus_f1": 0.019143,
    }

    # ------------------------------------------------------------- helpers
    def _tickers(self, features):
        return sorted(features.columns.get_level_values(1).unique(),
                      key=lambda c: int(c.split(".")[-1]))

    def _csrank(self, df, J):
        r = df.rank(axis=1, method="average")
        if J < 2:
            return r * 0.0
        return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

    def _demean(self, a):
        return a - np.nanmean(a, axis=1, keepdims=True)

    def _zs(self, s, w=120, mp=10):
        m = s.rolling(w, min_periods=mp).mean()
        sd = s.rolling(w, min_periods=mp).std()
        return ((s - m) / (sd + 1e-12)).fillna(0.0)

    # ---------------------------------------------------------- components
    def _components(self, features):
        need = set(self.WEIGHTS)
        tk = self._tickers(features)
        J = len(tk)
        f1 = features["Feature.1"][tk]
        C = {}

        def rank(df):
            return self._csrank(df, J)

        mkt = f1.mean(axis=1)
        sd20 = f1.rolling(20, min_periods=2).std()
        sd60 = f1.rolling(60, min_periods=5).std()
        rk1 = rank(f1)

        # ---- market beta / residual (precompute2 conventions) ----
        if need & {"idio_rev1", "idio_rev_x_f5"}:
            W = self.BETA_W
            f1m = f1.rolling(W, min_periods=20).mean()
            mm = mkt.rolling(W, min_periods=20).mean()
            cov = f1.mul(mkt, axis=0).rolling(W, min_periods=20).mean() - f1m.mul(mm, axis=0)
            varm = (mkt * mkt).rolling(W, min_periods=20).mean() - mm ** 2
            beta = cov.div(varm + 1e-12, axis=0).fillna(1.0)
            resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)

        # ---- reversal family ----
        if "rev_convex" in need:
            C["rev_convex"] = self._demean(-(rk1.values * rk1.abs().values))
        if "rev_bigmove" in need:
            thr = f1.abs() >= 2.0 * sd60
            C["rev_bigmove"] = self._demean(np.where(thr.values, -rk1.values, 0.0))
        if "rev1_x_vol" in need:
            C["rev1_x_vol"] = self._demean(
                -rk1.values * rank(sd20.fillna(0.0)).values)
        if "idio_rev1" in need:
            C["idio_rev1"] = -rank(resid)

        # ---- vol-regime mixture of reversal and long momentum ----
        if "mix_vol_rev_mom" in need:
            mvol = mkt.abs().ewm(span=20, min_periods=3).mean()
            z_mvol = np.tanh(self._zs(mvol, 120, 10).values)
            w_v = 1.0 / (1.0 + np.exp(-2.0 * z_mvol))
            mom_long = rank(f1.ewm(span=240, min_periods=1).mean())
            C["mix_vol_rev_mom"] = self._demean(
                w_v[:, None] * (-rk1.values) + (1 - w_v)[:, None] * mom_long.values)

        # ---- F4 spread ----
        if "f4_minus_f1" in need:
            f4 = features["Feature.4"][tk]
            C["f4_minus_f1"] = rank(f4 - f1)

        # ---- F5 / F6 family ----
        if need & {"idio_rev_x_f5", "vm_nov60"}:
            f5 = features["Feature.5"][tk]
            a5 = f5.abs()
            if "idio_rev_x_f5" in need:
                a5n = a5 / (a5.rolling(250, min_periods=1).mean() + 1e-9)
                C["idio_rev_x_f5"] = self._demean(
                    -rank(resid).values * rank(a5n).values)
            if "vm_nov60" in need:
                f6 = features["Feature.6"][tk]
                l5 = np.log1p(a5)
                ep6 = f6.expanding(min_periods=10).rank(pct=True).fillna(0.5)
                neg_f6_exppct = -rank(ep6)
                neg_f6_exppct_sm10 = -rank(ep6.rolling(10, min_periods=1).mean())
                a5n_l = l5 / (l5.rolling(250, min_periods=5).mean() + 1e-9)
                low_vshock5 = -rank(a5n_l.rolling(5, min_periods=1).mean())
                t5 = np.tanh(f5 / (f5.rolling(250, min_periods=10).std() + 1e-9))
                fade_sflow5 = -rank(t5.rolling(5, min_periods=1).sum())
                fade_sgnimb20 = -rank(np.sign(f5).rolling(20, min_periods=1).mean())
                neg_f6chg60 = -rank(f6 - f6.rolling(60, min_periods=1).mean())
                C["vm_nov60"] = self._demean(
                    0.48 * neg_f6_exppct.values + 0.22 * neg_f6_exppct_sm10.values
                    + 0.18 * low_vshock5.values + 0.05 * fade_sflow5.values
                    + 0.04 * fade_sgnimb20.values + 0.03 * neg_f6chg60.values)

        return C, tk, features.index

    # ----------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """No-op: weights are fixed offline; nothing is fit to the target."""
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        C, tk, index = self._components(features)
        out = np.zeros((len(index), len(tk)), dtype=float)
        for name, w in self.WEIGHTS.items():
            arr = C[name]
            if isinstance(arr, pd.DataFrame):
                arr = arr.values
            arr = np.nan_to_num(arr)
            arr = arr - arr.mean(axis=1, keepdims=True)
            out += w * arr
        pred = pd.DataFrame(out, index=index, columns=tk)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)
