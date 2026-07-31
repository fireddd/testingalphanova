"""Competition 5 submission: IND5 — distributional / second-moment axis.

An INDEPENDENCE-first leg, not a Sharpe-first one. Eight pre-declared
second-and-higher-moment statistics of Feature.1, combined equal-risk with
frozen scales. Zero fitted weights; train() is a no-op.

Mechanism. The cross-section is de-meaned and near-martingale, so even moments
carry no *level* information. They can only carry direction through a risk /
preference channel: a name whose realised path was generated asymmetrically
(variance concentrated in down-moves, positive idiosyncratic skew, fat tails,
unstable vol, asymmetric market beta) is a different risk object than a name
with the same mean and the same total variance, and the cross-section pays for
that difference. Every leg's SIGN comes from that theory, fixed before any
measurement (research/agents/t31/PREDECLARE_ind5_axis5.md); two of the eight
realised the opposite sign locally and were deliberately NOT flipped.

Legs (window / pre-declared sign / channel)
  L1 semiasym_60        60   +  downside share of variance -> downside-risk premium
  L2 semiasym_240      240   +  same channel, slow timescale
  L3 idio_semiasym_120 120   +  same on the market-beta residual (name-specific)
  L4 volofvol_60        60   +  vol-of-vol -> uncertainty premium
  L5 skew_120          120   -  lottery preference: positive skew is over-held
  L6 tailratio_60       60   +  mean-abs-dev / std; low = fat-tailed (same channel)
  L7 extremefreq_20     20   -  frequency in either cross-sectional tail quintile
  L8 betaasym_120      120   +  downside-minus-upside beta (Ang-Chen-Xing)

Combination: sum_i sign_i * csrank(leg_i) / sigma_i, with sigma_i the leg's own
sel75 (periods 001-075) PnL volatility, ddof=0, frozen as a literal constant.
Equal risk, no weight is fit to the target.

Local evidence (61,243 validation rows, periods 001-077):
  full +0.0217  h1 +0.0245  h2 +0.0191  rec20 +0.0251  hold +0.0107
  positive on every window including the sealed holdout.
  PnL corr: ticket2 +0.032, apex6 +0.249, unique -0.074, catboost +0.282,
            levy +0.018, resid +0.069  (max |corr| 0.282, all well under 0.45)

Causality: every statistic is a trailing rolling window or a row-wise
cross-sectional rank. No statistic is taken over the prediction block as a
whole, so predicting a prefix and predicting the full block agree bit-for-bit
on the overlap. train() fits nothing, so predict() is deterministic.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ind5DistributionalPredictor(Predictor):
    """Equal-risk blend of eight pre-declared distributional statistics of F1."""

    EPS = 1e-12

    # leg name -> (pre-declared sign, frozen sel75 PnL volatility, ddof=0)
    LEGS = (
        ("semiasym_60",       +1.0, 2.28979007e-02),
        ("semiasym_240",      +1.0, 2.09426220e-02),
        ("idio_semiasym_120", +1.0, 2.17198979e-02),
        ("volofvol_60",       +1.0, 2.19749945e-02),
        ("skew_120",          -1.0, 1.99313611e-02),
        ("tailratio_60",      +1.0, 1.87222037e-02),
        ("extremefreq_20",    -1.0, 1.53846244e-02),
        ("betaasym_120",      +1.0, 2.16261050e-02),
    )

    # ------------------------------------------------------------- helpers
    def _tickers(self, features):
        return sorted(features.columns.get_level_values(1).unique(),
                      key=lambda c: int(c.split(".")[-1]))

    def _minp(self, w):
        """One structural rule for every window: a quarter of it, at least 2."""
        return max(2, w // 4)

    def _csrank(self, df):
        """Row-wise cross-sectional rank scaled to [-1, 1], then row de-meaned.

        Row-wise only: no column ever sees another row's value, so this is
        causal and identical whether the block is passed whole or in prefixes.
        """
        r = df.rank(axis=1, method="average", na_option="keep")
        n = df.notna().sum(axis=1)
        z = (r.sub(0.5 * (n + 1), axis=0)).div(
            (0.5 * (n - 1)).replace(0, np.nan), axis=0)
        z = z.fillna(0.0)
        return z.sub(z.mean(axis=1), axis=0)

    def _semiasym(self, x, w):
        """(down-variance - up-variance) / total, trailing window w."""
        mp = self._minp(w)
        dn = (x.clip(upper=0.0) ** 2).rolling(w, min_periods=mp).mean()
        up = (x.clip(lower=0.0) ** 2).rolling(w, min_periods=mp).mean()
        return (dn - up) / (dn + up + self.EPS)

    def _build_legs(self, f1):
        """The eight legs. Every operation is trailing-rolling or row-wise."""
        eps = self.EPS
        out = {}

        # ---- L1 / L2 semivariance asymmetry at two timescales -------------
        out["semiasym_60"] = self._csrank(self._semiasym(f1, 60))
        out["semiasym_240"] = self._csrank(self._semiasym(f1, 240))

        # ---- L3 idiosyncratic semivariance asymmetry ----------------------
        mkt = f1.mean(axis=1)
        W, mp = 120, self._minp(120)
        mm = mkt.rolling(W, min_periods=mp).mean()
        varm = (mkt * mkt).rolling(W, min_periods=mp).mean() - mm * mm
        cov = (f1.mul(mkt, axis=0).rolling(W, min_periods=mp).mean()
               - f1.rolling(W, min_periods=mp).mean().mul(mm, axis=0))
        resid = f1 - cov.div(varm + eps, axis=0).mul(mkt, axis=0)
        out["idio_semiasym_120"] = self._csrank(self._semiasym(resid, W))

        # ---- L4 vol-of-vol -------------------------------------------------
        sd20 = f1.rolling(20, min_periods=self._minp(20)).std()
        out["volofvol_60"] = self._csrank(
            sd20.rolling(60, min_periods=self._minp(60)).std())

        # ---- L5 skewness from trailing raw moments -------------------------
        W, mp = 120, self._minp(120)
        m1 = f1.rolling(W, min_periods=mp).mean()
        m2 = (f1 ** 2).rolling(W, min_periods=mp).mean()
        m3 = (f1 ** 3).rolling(W, min_periods=mp).mean()
        sd = np.sqrt((m2 - m1 * m1).clip(lower=0.0))
        out["skew_120"] = self._csrank(
            (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / (sd ** 3 + eps))

        # ---- L6 robust-vs-standard dispersion ------------------------------
        W, mp = 60, self._minp(60)
        mu = f1.rolling(W, min_periods=mp).mean()
        mad = (f1 - mu).abs().rolling(W, min_periods=mp).mean()
        sdw = f1.rolling(W, min_periods=mp).std()
        out["tailratio_60"] = self._csrank(mad / (sdw + eps))

        # ---- L7 symmetric extremeness frequency (order statistic) ----------
        pct = f1.rank(axis=1, method="average", pct=True, na_option="keep")
        ext = ((pct <= 0.2) | (pct >= 0.8)).astype(float)
        out["extremefreq_20"] = self._csrank(
            ext.rolling(20, min_periods=self._minp(20)).mean())

        # ---- L8 downside-minus-upside beta ---------------------------------
        W, mp = 120, self._minp(120)
        d = (mkt < 0).astype(float)
        u = (mkt > 0).astype(float)
        fm = f1.mul(mkt, axis=0)
        m2k = mkt * mkt
        bd = fm.mul(d, axis=0).rolling(W, min_periods=mp).sum().div(
            (m2k * d).rolling(W, min_periods=mp).sum() + eps, axis=0)
        bu = fm.mul(u, axis=0).rolling(W, min_periods=mp).sum().div(
            (m2k * u).rolling(W, min_periods=mp).sum() + eps, axis=0)
        out["betaasym_120"] = self._csrank(bd - bu)

        return out

    # ----------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """No-op: signs come from theory, scales are frozen offline."""
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        f1 = features["Feature.1"][tk].astype(float)
        legs = self._build_legs(f1)

        out = np.zeros((len(f1.index), len(tk)), dtype=float)
        for name, sign, sigma in self.LEGS:
            arr = np.nan_to_num(legs[name].values, nan=0.0,
                                posinf=0.0, neginf=0.0)
            out += (sign / sigma) * arr

        pred = pd.DataFrame(out, index=features.index, columns=tk)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)
