"""Competition 5 submission: SPARSE upgraded with a decorrelated multi-family sleeve.

P = P_sparse + P_sleeve, both fixed-weight and offline-selected. The base is the
five-component SPARSE signal (our best platform performer); the sleeve is an
eight-component blend that was constrained to <=0.45 PnL correlation vs SPARSE,
scaled by a fixed offline constant (prediction-std alignment over periods
001-075, lambda=1.0 from an enumerated sweep). Diversification raises Sharpe:
local full-sample 0.0684 -> 0.0799 with the second half and recent windows both
improving. Anomalous periods 076/077 were sealed out of every selection step.

train() is a no-op: nothing is fit at run time. All operations are causal
(rolling / expanding / ewm / cumsum / cumprod with min_periods); no future access.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class SparsePlusPredictor(Predictor):
    """SPARSE base + scaled decorrelated sleeve, fixed weights throughout."""

    BAR = 5              # window for the synthetic high/low used by A/D
    VOLNORM_N = 500      # window normalising the synthetic volume
    BETA_W = 250         # rolling window for market-beta estimation

    # ---- base: the five SPARSE components (sparse_weights.csv, unchanged) ----
    W_BASE = {
        "rev_F1": 0.4008013843917598,
        "AD_F2_perp": 0.4,
        "F3_resid": 0.20001094126150296,
        "revsum_F1_w120_perp": 0.15000000026968516,
        "vol20": 0.0499999947679182,
    }

    # ---- sleeve: eight decorrelated components, scale 4.2504 folded in ----
    W_SLEEVE = {
        "f4_minus_f1": 0.383174,
        "vm_nov60": 0.350122,
        "idio_revsum60": 0.288866,
        "leadlag_follow3": 0.187013,
        "f6_x_vol": 0.147085,
        "mix_mvol_mdd_mom": -0.111496,
        "beta_instab": 0.103990,
        "idio_rev_x_f5": -0.080146,
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

    def _ad(self, price, high, low, vol):
        rng = (high - low).replace(0.0, np.nan)
        clv = ((price - low) - (high - price)) / rng
        return pd.DataFrame(np.cumsum(np.nan_to_num((clv * vol).values), axis=0),
                            index=price.index, columns=price.columns)

    def _cs_residual(self, y_df, basis_dfs):
        Y = y_df.values
        X = np.stack([b.values for b in basis_dfs], axis=-1)
        Yc = Y - Y.mean(1, keepdims=True)
        Xc = X - X.mean(1, keepdims=True)
        XtX = np.einsum("tjp,tjq->tpq", Xc, Xc)
        Xty = np.einsum("tjp,tj->tp", Xc, Yc)
        P = XtX.shape[-1]
        XtX = XtX + np.eye(P)[None] * 1e-10
        try:
            beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = np.zeros_like(Xty)
        resid = Yc - np.einsum("tjp,tp->tj", Xc, beta)
        return pd.DataFrame(np.nan_to_num(resid), index=y_df.index,
                            columns=y_df.columns)

    # ------------------------------------------------- base (SPARSE) block
    def _base_components(self, features, tk, J):
        F = {i: features[f"Feature.{i}"][tk] for i in (1, 2, 3, 5)}
        v = F[5].abs()
        vol = (v / (v.rolling(self.VOLNORM_N, min_periods=1).mean() + 1e-12)) * 1e6

        C = {}
        C["rev_F1"] = -self._csrank(F[1], J).values
        C["revsum_F1_w120"] = -self._csrank(
            F[1].rolling(120, min_periods=1).sum(), J).values
        C["F3_resid"] = self._csrank(
            self._cs_residual(F[3], [F[1], F[2]]), J).values
        C["vol20"] = self._csrank(
            F[1].rolling(20, min_periods=2).std().fillna(0.0), J).values

        price = (1.0 + F[2]).cumprod()
        high = price.rolling(self.BAR, min_periods=1).max()
        low = price.rolling(self.BAR, min_periods=1).min()
        C["AD_F2"] = self._csrank(self._ad(price, high, low, vol), J).values

        C = {k: self._demean(np.nan_to_num(vv)) for k, vv in C.items()}

        R = C["rev_F1"]
        rr = (R * R).sum(axis=1, keepdims=True)
        rr = np.where(rr > 1e-12, rr, np.nan)
        for k in ("AD_F2", "revsum_F1_w120"):
            coef = np.nan_to_num((C[k] * R).sum(axis=1, keepdims=True) / rr)
            C[f"{k}_perp"] = self._demean(C[k] - coef * R)
        return C

    # ------------------------------------------------------- sleeve block
    def _sleeve_components(self, features, tk, J):
        f1 = features["Feature.1"][tk]
        C = {}

        def rank(df):
            return self._csrank(df, J)

        mkt = f1.mean(axis=1)
        price = f1.cumsum()
        sd20 = f1.rolling(20, min_periods=2).std()
        sd60 = f1.rolling(60, min_periods=5).std()

        beta = self._rolling_beta(f1, mkt, self.BETA_W, 20).fillna(1.0)
        resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
        C["idio_revsum60"] = -rank(resid.rolling(60, min_periods=1).sum())

        f4 = features["Feature.4"][tk]
        C["f4_minus_f1"] = rank(f4 - f1)

        mkl = mkt.shift(1)
        ll = self._exp_corr(
            f1.iloc[1:],
            pd.DataFrame({c: mkl.iloc[1:] for c in f1.columns}),
            mp=30).reindex(f1.index).fillna(0.0)
        mk3 = mkt.rolling(3, min_periods=1).sum()
        mk3z = (mk3 / (mk3.rolling(120, min_periods=10).std() + 1e-12)).fillna(0.0)
        C["leadlag_follow3"] = self._demean(
            rank(ll).values * np.tanh(mk3z.values)[:, None])

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

        beta_s = self._rolling_beta(f1, mkt, 30, 10)
        beta_l = self._rolling_beta(f1, mkt, 120, 10)
        C["beta_instab"] = -rank((beta_s - beta_l).abs().fillna(0.0))

        f5 = features["Feature.5"][tk]
        f6 = features["Feature.6"][tk]
        a5 = f5.abs()
        a5n = a5 / (a5.rolling(250, min_periods=1).mean() + 1e-9)
        C["idio_rev_x_f5"] = self._demean(-rank(resid).values * rank(a5n).values)
        C["f6_x_vol"] = self._demean(rank(f6).values * rank(sd20.fillna(0.0)).values)

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
        return C

    # ----------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """No-op: weights are fixed offline; nothing is fit to the target."""
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        J = len(tk)
        out = np.zeros((len(features.index), J), dtype=float)

        CB = self._base_components(features, tk, J)
        for name, w in self.W_BASE.items():
            out += w * np.nan_to_num(CB[name])

        CS = self._sleeve_components(features, tk, J)
        for name, w in self.W_SLEEVE.items():
            arr = CS[name]
            if isinstance(arr, pd.DataFrame):
                arr = arr.values
            arr = np.nan_to_num(arr)
            arr = arr - arr.mean(axis=1, keepdims=True)
            out += w * arr

        pred = pd.DataFrame(out, index=features.index, columns=tk)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        return pred.fillna(0.0)
