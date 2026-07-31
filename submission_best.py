"""Competition 5 submission: max-Sharpe upgrade of the reversal/x34 cluster.

Fixed-weight, ten-component combination selected offline by deterministic
greedy forward selection + Nelder-Mead re-polish (enumerated grids,
name-sorted tie-breaks, no RNG), objective = Sharpe(periods 001-075)
+ 0.5*min(first-half, second-half Sharpe), subject to PnL correlation
<= 0.45 against submission_sparse, submission_unique and submission_set4
(so those signals are never evicted by this one), correlation vs
submission_sharpe left free (this entry is the direct upgrade of that
signal and competes for its slot). Weights were then pruned 12 -> 10 and
shrunk 25% toward sign-preserving equal contribution for robustness.
Anomalous periods 076/077 were sealed out of every selection step.

Composition: idiosyncratic ewm-10 reversal, F3+F4 cross-sectional residual
(ewm-3), lead-lag follow gated by market sign, beta x market-momentum tilts
(5 and 20 bar), F6-state + accumulation/distribution fade blend, F4/F3
reversals, F6 level fade, dispersion-scaled reversal hedge.

train() is a no-op: nothing is fit at run time. All operations are causal
(rolling / expanding / ewm / cumsum with min_periods, positive shifts only);
no future access, no bfill, no full-sample statistics.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class BestUpgradePredictor(Predictor):
    """Fixed-weight max-Sharpe combination (upgrade play for the sharpe slot)."""

    BETA_W = 250  # rolling window for market-beta estimation

    WEIGHTS = {
        "idio_rev_ewm10": 0.545872,
        "x34_sum_ewm3": 1.145712,
        "leadlag_sign": 0.660906,
        "beta_x_mmom5": -0.353031,
        "vmix2": 0.274694,
        "rev_f4": 0.370824,
        "beta_x_mmom20": 0.309714,
        "f6": -0.429251,
        "rev_f3": 0.510290,
        "rev_dispz": -0.413130,
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

    def _exp_corr(self, x, y, mp=30):
        exy = (x * y).expanding(min_periods=mp).mean()
        ex = x.expanding(min_periods=mp).mean()
        ey = y.expanding(min_periods=mp).mean()
        ex2 = (x * x).expanding(min_periods=mp).mean()
        ey2 = (y * y).expanding(min_periods=mp).mean()
        num = exy - ex * ey
        den = np.sqrt((ex2 - ex ** 2).clip(lower=0)
                      * (ey2 - ey ** 2).clip(lower=0)) + 1e-12
        return (num / den).fillna(0.0)

    def _cs_resid2(self, y, x1, x2):
        """Per-timestamp cross-sectional OLS residual of y on [1, x1, x2]."""
        ym = y.sub(y.mean(axis=1), axis=0)
        x1m = x1.sub(x1.mean(axis=1), axis=0)
        x2m = x2.sub(x2.mean(axis=1), axis=0)
        a11 = (x1m * x1m).sum(axis=1)
        a12 = (x1m * x2m).sum(axis=1)
        a22 = (x2m * x2m).sum(axis=1)
        b1 = (x1m * ym).sum(axis=1)
        b2 = (x2m * ym).sum(axis=1)
        det = a11 * a22 - a12 * a12 + 1e-18
        beta1 = (a22 * b1 - a12 * b2) / det
        beta2 = (a11 * b2 - a12 * b1) / det
        return ym - x1m.mul(beta1, axis=0) - x2m.mul(beta2, axis=0)

    # ---------------------------------------------------------- components
    def _components(self, features):
        tk = self._tickers(features)
        J = len(tk)
        f1 = features["Feature.1"][tk]
        f2 = features["Feature.2"][tk]
        f3 = features["Feature.3"][tk]
        f4 = features["Feature.4"][tk]
        f5 = features["Feature.5"][tk]
        f6 = features["Feature.6"][tk]
        C = {}

        def rank(df):
            return self._csrank(df, J)

        mkt = f1.mean(axis=1)

        # ---- market beta / idio residual (precompute2 conventions) ----
        W = self.BETA_W
        f1m = f1.rolling(W, min_periods=20).mean()
        mm = mkt.rolling(W, min_periods=20).mean()
        cov = f1.mul(mkt, axis=0).rolling(W, min_periods=20).mean() \
            - f1m.mul(mm, axis=0)
        varm = (mkt * mkt).rolling(W, min_periods=20).mean() - mm ** 2
        beta = cov.div(varm + 1e-12, axis=0).fillna(1.0)
        resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
        C["idio_rev_ewm10"] = -rank(resid.ewm(span=10, min_periods=1).mean())

        # ---- beta x market-momentum tilts ----
        rkb = rank(beta)
        mmom5 = mkt.rolling(5, min_periods=1).sum()
        mmom20 = mkt.rolling(20, min_periods=1).sum()
        C["beta_x_mmom5"] = self._demean(
            rkb.values * np.sign(mmom5.values)[:, None])
        C["beta_x_mmom20"] = self._demean(
            rkb.values * np.sign(mmom20.values)[:, None])

        # ---- F3/F4 cross-sectional residual on [F1, F2] ----
        e3 = self._cs_resid2(f3, f1, f2)
        e4 = self._cs_resid2(f4, f1, f2)
        C["x34_sum_ewm3"] = rank((e3 + e4).ewm(span=3, min_periods=1).mean())

        # ---- lead-lag follow gated by market sign ----
        mkl = mkt.shift(1)
        ll = self._exp_corr(
            f1.iloc[1:],
            pd.DataFrame({c: mkl.iloc[1:] for c in f1.columns}),
            mp=30).reindex(f1.index).fillna(0.0)
        C["leadlag_sign"] = self._demean(
            rank(ll).values * np.sign(mkt.values)[:, None])

        # ---- F6-state fade + accumulation/distribution fade blend ----
        ep6 = f6.expanding(min_periods=10).rank(pct=True)
        negf6ep = -rank(ep6.fillna(0.5))
        adl = np.sign(f1) * np.log1p(f5.abs())
        fa60 = -rank(adl.rolling(60, min_periods=1).sum())
        C["vmix2"] = self._demean(negf6ep.values + fa60.values)

        # ---- simple fades ----
        C["rev_f3"] = -rank(f3)
        C["rev_f4"] = -rank(f4)
        C["f6"] = rank(f6)

        # ---- dispersion-scaled reversal (hedge, negative weight) ----
        disp = f1.std(axis=1)
        z_disp = np.tanh(self._zs(disp, 120, 10).values)
        C["rev_dispz"] = self._demean(
            -rank(f1).values * z_disp[:, None])

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
