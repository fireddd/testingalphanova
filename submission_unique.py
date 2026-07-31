"""Competition 5 submission: uniqueness-first multi-family signal.

Fixed-weight combination selected offline (deterministic Nelder-Mead from
enumerated starting points) under hard constraints:

  - pooled city novelty >= 75 deg vs the 266 published signals (0 within 60 deg)
  - trajectory correlation with plain reversal ~ 0.00
  - PnL correlation vs our previously submitted signal (signal_b22b5c14) <= 0.30
  - anomalous periods 076/077 sealed out of all selection windows

Families combined (no single family dominates):
  - lead-lag: laggards follow the market leader (expanding within-slice corr)
  - drawdown ladder: distance-from-rolling-high at 10/20/40 bars (long/short mix)
  - F3/F4 cross-sectional residuals on [F1, F2] (forward info, reversal-orthogonal)
  - F6 state fades via causal expanding percentile + signed F5 flow fades
  - path-shape: age of high, aged deep drawdown, beta instability, trend-quality
  - risk carriers: vol / low-beta / low-flow tilts (steer the city, small weights)

train() is a no-op: nothing is fit at run time. All operations are causal
(rolling / expanding / ewm / cumsum with min_periods); no future access.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class UniqueMultiFamilyPredictor(Predictor):
    """Fixed-weight, novelty-first combination across six signal families."""

    BETA_W = 250  # rolling window for market-beta estimation

    WEIGHTS = {
        "leadlag_follow": 1.890644,
        "dist_high20": -1.286372,
        "vm_prize50": 1.213596,
        "x3_ewm3": 1.002326,
        "mix_disp3_dd_mom": 0.904485,
        "dist_high40": 0.768528,
        "f6_x_vol": -0.309328,
        "x4p_ewm5": 0.307333,
        "age_high40": -0.297122,
        "dist_high10": 0.290528,
        "fade_sflow_ewm10": 0.273248,
        "fade_adl_ewm100": -0.244467,
        "lowbeta": -0.236717,
        "beta_instab": 0.202221,
        "lowf5abs": 0.184939,
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

    def _zs(self, s, w=120, mp=10):
        """Causal rolling z-score of a Series."""
        m = s.rolling(w, min_periods=mp).mean()
        sd = s.rolling(w, min_periods=mp).std()
        return ((s - m) / (sd + 1e-12)).fillna(0.0)

    def _exp_corr(self, x, y, mp=30):
        """Expanding correlation between two aligned DataFrames."""
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
        if need & {"lowbeta", "idio_dist_high20"}:
            beta = self._rolling_beta(f1, mkt, self.BETA_W, 20).fillna(1.0)
            resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
            if "lowbeta" in need:
                C["lowbeta"] = -rank(beta)
            if "idio_dist_high20" in need:
                ip = resid.cumsum()
                C["idio_dist_high20"] = rank(ip.rolling(20, min_periods=1).max() - ip)

        # ---- reversal remnants ----
        if "revsum5" in need:
            C["revsum5"] = -rank(f1.rolling(5, min_periods=1).sum())
        if "rev_bigmove" in need:
            rk1 = rank(f1)
            thr = f1.abs() >= 2.0 * sd60
            C["rev_bigmove"] = self._demean(np.where(thr.values, -rk1.values, 0.0))

        # ---- drawdown ladder ----
        for w in (10, 20, 40):
            if f"dist_high{w}" in need:
                C[f"dist_high{w}"] = rank(price.rolling(w, min_periods=1).max() - price)

        # ---- vol / carrier tilts ----
        if "vol20" in need:
            C["vol20"] = rank(sd20.fillna(0.0))
        if "lowvol60" in need:
            C["lowvol60"] = -rank(sd60.fillna(0.0))

        # ---- F3/F4 residual family ----
        if need & {"x3_ewm3", "x4p_ewm5", "x34_agree"}:
            f2 = features["Feature.2"][tk]
            f3 = features["Feature.3"][tk]
            f4 = features["Feature.4"][tk]
            e3 = self._cs_resid2(f3, f1, f2)
            e4 = self._cs_resid2(f4, f1, f2)
            r3 = rank(e3.ewm(span=3, min_periods=1).mean())
            if "x3_ewm3" in need:
                C["x3_ewm3"] = r3
            if "x4p_ewm5" in need:
                C["x4p_ewm5"] = rank(e4.ewm(span=5, min_periods=1).mean())
            if "x34_agree" in need:
                r4e3 = rank(e4.ewm(span=3, min_periods=1).mean())
                agree20 = (np.sign(f3) * np.sign(f4)).rolling(20, min_periods=1).mean()
                C["x34_agree"] = self._demean(
                    r3.values + r4e3.values + 0.75 * rank(agree20).values)

        # ---- lead-lag: laggards follow the leader ----
        if "leadlag_follow" in need:
            mkl = mkt.shift(1)
            ll = self._exp_corr(
                f1.iloc[1:],
                pd.DataFrame({c: mkl.iloc[1:] for c in f1.columns}),
                mp=30).reindex(f1.index).fillna(0.0)
            mvs = (mkt / (mkt.rolling(120, min_periods=10).std() + 1e-12)).fillna(0.0)
            C["leadlag_follow"] = self._demean(
                rank(ll).values * np.tanh(mvs.values)[:, None])

        # ---- dispersion-gated mixture of drawdown and long momentum ----
        if "mix_disp3_dd_mom" in need:
            disp = f1.std(axis=1)
            z_disp = np.tanh(self._zs(disp, 120, 10).values)
            w_d3 = 1.0 / (1.0 + np.exp(-3.5 * z_disp))
            dh20 = rank(price.rolling(20, min_periods=1).max() - price)
            mom_long = rank(f1.ewm(span=240, min_periods=1).mean())
            C["mix_disp3_dd_mom"] = self._demean(
                w_d3[:, None] * dh20.values + (1 - w_d3)[:, None] * mom_long.values)

        # ---- path shape: age of high, aged deep drawdown ----
        if need & {"age_high40", "aged_deep_dd"}:
            T = len(f1)
            rmax = price.rolling(40, min_periods=1).max()
            is_high = (price >= rmax - 1e-12).values
            pos = np.arange(T)[:, None] * np.ones((1, J))
            last_high = np.where(is_high, pos, -1.0)
            last_high = np.maximum.accumulate(last_high, axis=0)
            tsm = pos - last_high
            tsm_df = pd.DataFrame(tsm, index=f1.index, columns=f1.columns)
            if "age_high40" in need:
                C["age_high40"] = self._demean(tsm_df.rank(axis=1).values / J * 2 - 1)
            if "aged_deep_dd" in need:
                dd = rmax - price
                depth40 = dd.rolling(40, min_periods=1).max() / (sd60 + 1e-9)
                rk_depth = rank(depth40.fillna(0.0))
                rk_age = rank(tsm_df)
                C["aged_deep_dd"] = self._demean(rank(rk_depth + rk_age).values)

        # ---- beta instability (30 vs 120 divergence) ----
        if "beta_instab" in need:
            beta_s = self._rolling_beta(f1, mkt, 30, 10)
            beta_l = self._rolling_beta(f1, mkt, 120, 10)
            C["beta_instab"] = -rank((beta_s - beta_l).abs().fillna(0.0))

        # ---- F5 flow / F6 state family ----
        if need & {"vm_prize50", "fade_adl_ewm100", "fade_sflow_ewm10",
                   "lowf5abs", "f6_x_vol"}:
            f5 = features["Feature.5"][tk]
            f6 = features["Feature.6"][tk]
            a5 = f5.abs()
            l5 = np.log1p(a5)
            t5 = np.tanh(f5 / (f5.rolling(250, min_periods=10).std() + 1e-9))
            adl = np.sign(f1) * l5
            if "vm_prize50" in need:
                neg_f6 = -rank(f6)
                ep6 = f6.expanding(min_periods=10).rank(pct=True).fillna(0.5)
                neg_f6_exppct = -rank(ep6)
                fade_sflow5 = -rank(t5.rolling(5, min_periods=1).sum())
                fade_sflow20 = -rank(t5.rolling(20, min_periods=1).sum())
                a5n_l = l5 / (l5.rolling(250, min_periods=5).mean() + 1e-9)
                low_vshock5 = -rank(a5n_l.rolling(5, min_periods=1).mean())
                neg_f6chg60 = -rank(f6 - f6.rolling(60, min_periods=1).mean())
                fade_adl_ewm60 = -rank(adl.ewm(span=60, min_periods=1).mean())
                C["vm_prize50"] = self._demean(
                    0.07 * neg_f6.values + 0.10 * neg_f6_exppct.values
                    + 0.19 * fade_sflow5.values + 0.13 * fade_sflow20.values
                    + 0.15 * low_vshock5.values + 0.18 * neg_f6chg60.values
                    + 0.18 * fade_adl_ewm60.values)
            if "fade_adl_ewm100" in need:
                C["fade_adl_ewm100"] = -rank(adl.ewm(span=100, min_periods=1).mean())
            if "fade_sflow_ewm10" in need:
                C["fade_sflow_ewm10"] = -rank(t5.ewm(span=10, min_periods=1).mean())
            if "lowf5abs" in need:
                a5n = a5 / (a5.rolling(250, min_periods=1).mean() + 1e-9)
                C["lowf5abs"] = -rank(a5n)
            if "f6_x_vol" in need:
                C["f6_x_vol"] = self._demean(
                    rank(f6).values * rank(sd20.fillna(0.0)).values)

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
