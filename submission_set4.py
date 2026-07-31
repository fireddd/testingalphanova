"""Competition 5 submission: fourth-set-member candidate (decorrelated max-Sharpe).

Fixed-weight combination selected offline by deterministic greedy forward
selection + Nelder-Mead re-polish (enumerated grid, name-sorted tie-breaks,
no RNG), objective = full-sample Sharpe + 0.5*min(first-half, second-half)
Sharpe, subject to PnL correlation <= 0.45 against ALL THREE of our previously
submitted signals (sparse, sharpe, unique), so it can be admitted alongside
them. Anomalous periods 076/077 sealed out of selection.

Composition: F4-F1 spread, F6-state fade blend, idiosyncratic 60-bar reversal,
3-bar lead-lag follow, F6xvol interaction, vol-regime drawdown/momentum
mixture, beta instability, idio-reversal x flow interaction.

train() is a no-op: nothing is fit at run time. All operations are causal
(rolling / expanding / ewm / cumsum with min_periods); no future access.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class SetMemberPredictor(Predictor):
    """Fixed-weight combination decorrelated from all prior submissions."""

    BETA_W = 250  # rolling window for market-beta estimation

    WEIGHTS = {
        "f4_minus_f1": 0.090150,
        "vm_nov60": 0.082374,
        "idio_revsum60": 0.067962,
        "leadlag_follow3": 0.043999,
        "f6_x_vol": 0.034605,
        "mix_mvol_mdd_mom": -0.026232,
        "beta_instab": 0.024466,
        "idio_rev_x_f5": -0.018856,
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

    def _rolling_beta(self, f1, mkt, W, mp):
        f1m = f1.rolling(W, min_periods=mp).mean()
        mm = mkt.rolling(W, min_periods=mp).mean()
        cov = f1.mul(mkt, axis=0).rolling(W, min_periods=mp).mean() - f1m.mul(mm, axis=0)
        varm = (mkt * mkt).rolling(W, min_periods=mp).mean() - mm ** 2
        return cov.div(varm + 1e-12, axis=0)

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
        price = f1.cumsum()
        sd20 = f1.rolling(20, min_periods=2).std()
        sd60 = f1.rolling(60, min_periods=5).std()

        # ---- market beta / residual (precompute2 conventions) ----
        if need & {"idio_revsum60", "idio_rev_x_f5"}:
            beta = self._rolling_beta(f1, mkt, self.BETA_W, 20).fillna(1.0)
            resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
            if "idio_revsum60" in need:
                C["idio_revsum60"] = -rank(resid.rolling(60, min_periods=1).sum())

        # ---- F4 spread ----
        if "f4_minus_f1" in need:
            f4 = features["Feature.4"][tk]
            C["f4_minus_f1"] = rank(f4 - f1)

        # ---- lead-lag (3-bar market impulse variant) ----
        if "leadlag_follow3" in need:
            mkl = mkt.shift(1)
            ll = self._exp_corr(
                f1.iloc[1:],
                pd.DataFrame({c: mkl.iloc[1:] for c in f1.columns}),
                mp=30).reindex(f1.index).fillna(0.0)
            mk3 = mkt.rolling(3, min_periods=1).sum()
            mk3z = (mk3 / (mk3.rolling(120, min_periods=10).std() + 1e-12)).fillna(0.0)
            C["leadlag_follow3"] = self._demean(
                rank(ll).values * np.tanh(mk3z.values)[:, None])

        # ---- vol-regime mixture of vol-adjusted drawdown depth and momentum ----
        if "mix_mvol_mdd_mom" in need:
            mvol = mkt.abs().ewm(span=20, min_periods=3).mean()
            z_mvol = np.tanh(self._zs(mvol, 120, 10).values)
            w_v = 1.0 / (1.0 + np.exp(-2.0 * z_mvol))
            rmax = price.rolling(40, min_periods=1).max()
            dd = rmax - price
            depth40 = dd.rolling(40, min_periods=1).max() / (sd60 + 1e-9)
            rk_depth = rank(depth40.fillna(0.0))
            mom_long = rank(f1.ewm(span=240, min_periods=1).mean())
            C["mix_mvol_mdd_mom"] = self._demean(
                w_v[:, None] * rk_depth.values + (1 - w_v)[:, None] * mom_long.values)

        # ---- beta instability (30 vs 120 divergence) ----
        if "beta_instab" in need:
            beta_s = self._rolling_beta(f1, mkt, 30, 10)
            beta_l = self._rolling_beta(f1, mkt, 120, 10)
            C["beta_instab"] = -rank((beta_s - beta_l).abs().fillna(0.0))

        # ---- F5 / F6 family ----
        if need & {"idio_rev_x_f5", "vm_nov60", "f6_x_vol"}:
            f5 = features["Feature.5"][tk]
            f6 = features["Feature.6"][tk]
            a5 = f5.abs()
            if "idio_rev_x_f5" in need:
                a5n = a5 / (a5.rolling(250, min_periods=1).mean() + 1e-9)
                C["idio_rev_x_f5"] = self._demean(
                    -rank(resid).values * rank(a5n).values)
            if "f6_x_vol" in need:
                C["f6_x_vol"] = self._demean(
                    rank(f6).values * rank(sd20.fillna(0.0)).values)
            if "vm_nov60" in need:
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
