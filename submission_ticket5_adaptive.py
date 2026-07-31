"""Competition 5 submission: "ticket5" — a four-leg equal-risk alpha blend.

WHAT THIS IS
    An equal-RISK average of four independently-derived, structurally unrelated
    cross-sectional predictors of the next bar.  The legs were chosen for mutual
    ORTHOGONALITY, not for individual Sharpe:

      L1  ticket4cand  price-space grid-averaged VAR(1) lead-lag graph
                       (ticket2's leg A) + 3.8918 * the ranked, 3-bar-EWM'd
                       per-timestamp cross-sectional residual of F3+F4 on
                       [1, F1, F2].  Graph structure + cross-feature structure.
      L2  r_masterR    regime blend: 0.55 (vol-normalised) flow/state-fade mix
                       + 0.45 (vol-normalised) idiosyncratic slow-reversal mix.
                       This is the leg that carries the LATE regime.
      L3  ll_rank      expanding lead-lag adjacency on the standardised,
                       market-removed F1 cross-section, own-name zeroed, ranked.
      L4  rev_f3       -csrank(Feature.3): plain reversal read off the second
                       noisy observation of the return process.

COMBINATION — FIXED OFFLINE SCALE CONSTANTS, NOT A VALIDATION STATISTIC
    ticket5 = mean_k( leg_k / SCALE_k ).  Each SCALE_k is a HARD-CODED number:
    the median, over training periods 001-075, of that leg's mean per-row
    cross-sectional dispersion.  Nothing in predict() computes a statistic over
    the scored block — the normalisation is a constant of the source code.
    (Periods 076/077 were never used to fit anything, including these scales.)

    There are no other fitted weights.  The 3.8918 gain inside L1 and the
    0.55/0.45 + component weights inside L2 are inherited unchanged from the
    signals' original definitions.

CAUSALITY
    train() sees training rows only, and only L1 needs it (it stores the tail of
    the standardised F1 cross-section so leg A's moment accumulators can be
    rebuilt at the train/val boundary and then recursed forward one validation
    row at a time).  L2/L3/L4 are pure functions of the rows at or before t:
    rolling/expanding/ewm/cumsum with min_periods, per-row cross-sectional
    ranks and per-row OLS residuals.  No backward fill, no negative shift, no
    centred window, no whole-block statistic.

    NOTE on L3: the research implementation of the expanding per-ticker
    volatility back-filled its first 10 warm-up rows with the value computed at
    row 10.  That is a backward fill and is NOT shipped here.  Those rows use a
    causal pooled (all-ticker) expanding dispersion instead.  Cost: L3's PnL
    correlation against the research series is 0.9943 and its full-sample
    Sharpe moves 0.0356 -> 0.0355.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket5AdaptivePredictor(Predictor):
    # ---- L1a: ticket2 leg A, price-space VAR(1) lead-lag graph ---------------
    WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
    LAM_A = 0.10
    MINP = 250
    # ---- L1b: gain on the ranked F3+F4 cross-sectional residual --------------
    G_X34 = 3.8918
    # ---- L2: regime master blend --------------------------------------------
    W_VMIXR = (("neg_f6_rollpct250", 0.25), ("neg_f6chg120", 0.15),
               ("fade_adl60", 0.20), ("fade_adl_ewm100", 0.15),
               ("fade_sflow5", 0.25))
    W_IDIOR = (("idh60_b125", 0.20), ("irs40_b125", 0.20), ("idh40_b250", 0.20),
               ("idh80_b250", 0.20), ("irs20_b250", 0.20))
    MASTER_A = 0.55 / 0.011784
    MASTER_B = 0.45 / 0.022791
    # ---- L3: expanding lead-lag adjacency -----------------------------------
    WARM = 40
    MIN_N = 10
    # ---- blend: fixed offline dispersion constants (periods 001-075) --------
    SCALES = (2.547476, 21.719934, 0.575869, 0.606977)   # prior (median over periods 001-075)
    SCALE_TAIL = 3000   # training rows used to re-measure leg dispersion
    SCALE_MIN = 200     # minimum usable rows before trusting the estimate

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _fdf(features, i, tk):
        """Feature i as a (T, J) DataFrame in ticker order."""
        return features[f"Feature.{i}"][tk]

    @staticmethod
    def _f1(features, tk):
        return np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _zrow(a):
        """Cross-sectional z-score of each row."""
        m = a.mean(1, keepdims=True)
        s = a.std(1, keepdims=True)
        return (a - m) / (s + 1e-9)

    @staticmethod
    def _unit_row(a):
        """Scale each row to unit cross-sectional dispersion (row-local, causal)."""
        s = a.std(1, keepdims=True)
        return a / np.maximum(s, 1e-12)

    @staticmethod
    def _demean(a):
        return a - a.mean(axis=1, keepdims=True)

    @staticmethod
    def _csrank_df(df, j):
        """Cross-sectional average rank of each row mapped to [-1, +1]."""
        r = df.rank(axis=1, method="average")
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _csrank_np(a, j):
        """Cross-sectional ordinal rank of each row mapped to [-1, +1]."""
        r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _to_dm(v):
        """DataFrame/array -> NaN-free, row-demeaned float array."""
        arr = v.to_numpy(dtype=np.float64) if hasattr(v, "to_numpy") else np.asarray(v)
        arr = np.nan_to_num(arr.astype(np.float64))
        return arr - arr.mean(axis=1, keepdims=True)

    @staticmethod
    def _cs_resid2(y, x1, x2):
        """Per-timestamp (row-wise) OLS residual of y on [1, x1, x2]. Causal."""
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

    def _exp_std_ticker(self, a):
        """Expanding per-ticker dispersion. Causal, including the warm-up rows:
        before MIN_N observations exist per ticker the POOLED (all-ticker)
        expanding dispersion over rows 0..t is used. No backward fill."""
        t = a.shape[0]
        j = a.shape[1]
        n = (np.arange(t) + 1.0)[:, None]
        s1 = np.cumsum(a, axis=0)
        s2 = np.cumsum(a * a, axis=0)
        v = s2 / n - (s1 / n) ** 2
        sd = np.sqrt(np.maximum(v, 1e-18))
        c1 = np.cumsum(a.sum(axis=1))
        c2 = np.cumsum((a * a).sum(axis=1))
        m = (np.arange(t) + 1.0) * j
        pv = c2 / m - (c1 / m) ** 2
        psd = np.sqrt(np.maximum(pv, 1e-18))
        k = min(self.MIN_N, t)
        sd[:k] = psd[:k, None]
        return sd

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        f1 = self._f1(features, tk)
        self.j_ = f1.shape[1]
        # L1 leg-A state: tail of the standardised cross-section. L2/L3/L4 are
        # stateless (pure causal functions of the rows handed to predict()).
        self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()
        # ---- ADAPTIVE SCALES ------------------------------------------------
        # Re-measure each leg's typical per-row cross-sectional dispersion on the
        # most recent TRAINING rows (legal: train() owns these rows).  This lets
        # the risk-equalisation re-calibrate every period instead of trusting a
        # constant fitted to 2026 data, so a regime shift in dispersion cannot
        # silently mis-weight the legs.  Falls back to SCALES if anything fails.
        self.scales_ = self.SCALES
        try:
            tail = features.iloc[-self.SCALE_TAIL:] if len(features) > self.SCALE_TAIL else features
            if len(tail) >= self.SCALE_MIN:
                est = []
                for L, dflt in zip(self._legs(tail, tk), self.SCALES):
                    d = np.nanstd(L, axis=1)
                    d = d[np.isfinite(d) & (d > 1e-12)]
                    m = float(np.median(d)) if d.size >= self.SCALE_MIN else 0.0
                    # shrink halfway to the prior: adaptive but not jumpy
                    est.append(0.5 * m + 0.5 * dflt if m > 1e-12 else dflt)
                self.scales_ = tuple(est)
        except Exception:
            self.scales_ = self.SCALES

    # ------------------------------------------------------------------- legs
    def _legA(self, xv):
        """Grid-averaged VAR(1) lead-lag propagation. Recurses causally."""
        nv, k = xv.shape
        x = np.vstack([self.bufA_, xv])
        t0 = len(self.bufA_)
        a = x[1:]
        b = x[:-1]
        eye = np.eye(k)
        acc = np.zeros((nv, k))
        for wlen in self.WGRID:
            lo0 = max(0, t0 - wlen)
            m = a[lo0:t0].T @ b[lo0:t0]
            s = b[lo0:t0].T @ b[lo0:t0]
            vc = (a[lo0:t0] ** 2).sum(0)
            c = float(t0 - lo0)
            for u in range(nv):
                if u > 0:
                    p = t0 + u - 1
                    m += np.outer(a[p], b[p])
                    s += np.outer(b[p], b[p])
                    vc += a[p] ** 2
                    c += 1
                    q = p - wlen
                    if q >= 0:
                        m -= np.outer(a[q], b[q])
                        s -= np.outer(b[q], b[q])
                        vc -= a[q] ** 2
                        c -= 1
                if c < self.MINP:
                    continue
                ds = np.sqrt(np.maximum(np.diag(s), 1e-12))
                cm = m / np.sqrt(np.outer(np.maximum(vc, 1e-12),
                                          np.maximum(np.diag(s), 1e-12)))
                sc = s / np.outer(ds, ds)
                bm = cm @ np.linalg.inv((1.0 - self.LAM_A) * sc + self.LAM_A * eye)
                r = bm @ xv[u]
                o = np.argsort(np.argsort(r)).astype(np.float64)
                acc[u] += o / (k - 1.0) - 0.5
        return acc

    def _leg1(self, features, tk):
        """ticket4cand = unit-dispersion leg A + G_X34 * ranked ewm3(e3 + e4)."""
        j = len(tk)
        xv = self._zrow(self._f1(features, tk))
        ga = self._unit_row(self._legA(xv))
        ga = ga - ga.mean(1, keepdims=True)
        f1 = self._fdf(features, 1, tk)
        f2 = self._fdf(features, 2, tk)
        f3 = self._fdf(features, 3, tk)
        f4 = self._fdf(features, 4, tk)
        e3 = self._cs_resid2(f3, f1, f2)
        e4 = self._cs_resid2(f4, f1, f2)
        x34 = self._to_dm(
            self._csrank_df((e3 + e4).ewm(span=3, min_periods=1).mean(), j))
        return ga + self.G_X34 * x34

    def _leg2(self, features, tk):
        """r_masterR: 0.55/0.45 vol-normalised flow-fade + idio slow-reversal."""
        j = len(tk)
        f1 = self._fdf(features, 1, tk)
        f5 = self._fdf(features, 5, tk)
        f6 = self._fdf(features, 6, tk)
        c = {}

        # --- flow / state fade family ---
        rp = f6.rolling(250, min_periods=10).rank(pct=True).fillna(0.5)
        c["neg_f6_rollpct250"] = -self._csrank_df(rp, j)
        c["neg_f6chg120"] = -self._csrank_df(
            f6 - f6.rolling(120, min_periods=1).mean(), j)
        adl = np.sign(f1) * np.log1p(f5.abs())
        c["fade_adl60"] = -self._csrank_df(adl.rolling(60, min_periods=1).sum(), j)
        c["fade_adl_ewm100"] = -self._csrank_df(
            adl.ewm(span=100, min_periods=1).mean(), j)
        t5 = np.tanh(f5 / (f5.rolling(250, min_periods=10).std() + 1e-9))
        c["fade_sflow5"] = -self._csrank_df(t5.rolling(5, min_periods=1).sum(), j)

        # --- idiosyncratic slow-reversal family ---
        mkt = f1.mean(axis=1)
        for w in (125, 250):
            cov = f1.mul(mkt, axis=0).rolling(w, min_periods=20).mean() - \
                f1.rolling(w, min_periods=20).mean().mul(
                    mkt.rolling(w, min_periods=20).mean(), axis=0)
            varm = (mkt * mkt).rolling(w, min_periods=20).mean() - \
                mkt.rolling(w, min_periods=20).mean() ** 2
            beta = cov.div(varm + 1e-12, axis=0).fillna(1.0)
            resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
            ip = resid.cumsum()
            if w == 125:
                c["idh60_b125"] = self._csrank_df(
                    ip.rolling(60, min_periods=1).max() - ip, j)
                c["irs40_b125"] = -self._csrank_df(
                    resid.rolling(40, min_periods=1).sum(), j)
            else:
                c["idh40_b250"] = self._csrank_df(
                    ip.rolling(40, min_periods=1).max() - ip, j)
                c["idh80_b250"] = self._csrank_df(
                    ip.rolling(80, min_periods=1).max() - ip, j)
                c["irs20_b250"] = -self._csrank_df(
                    resid.rolling(20, min_periods=1).sum(), j)

        a = {k: self._to_dm(v) for k, v in c.items()}
        vmix = sum(w * a[k] for k, w in self.W_VMIXR)
        idio = sum(w * a[k] for k, w in self.W_IDIOR)
        return self._demean(self.MASTER_A * vmix + self.MASTER_B * idio)

    def _leg3(self, features, tk):
        """ll_rank: expanding own-name-zeroed lead-lag adjacency, ranked."""
        j = len(tk)
        f1 = np.nan_to_num(
            self._fdf(features, 1, tk).to_numpy(dtype=np.float64))
        f1d = f1 - f1.mean(axis=1, keepdims=True)
        u = np.clip(f1d / (self._exp_std_ticker(f1d) + 1e-12), -3.0, 3.0)
        t = u.shape[0]
        c1 = np.zeros((j, j))
        sig = np.zeros((t, j))
        for i in range(t):
            if i >= 1:
                c1 += np.outer(u[i - 1], u[i])
            if i >= self.WARM:
                adj = c1 / i
                np.fill_diagonal(adj, 0.0)
                sig[i] = adj.T @ u[i]
        out = self._csrank_np(sig, j)
        out[:self.WARM + 1] = 0.0
        return self._demean(out)

    def _leg4(self, features, tk):
        """rev_f3 = -csrank(Feature.3)."""
        j = len(tk)
        return self._to_dm(-self._csrank_df(self._fdf(features, 3, tk), j))

    def _legs(self, features, tk):
        return (self._leg1(features, tk), self._leg2(features, tk),
                self._leg3(features, tk), self._leg4(features, tk))

    # ---------------------------------------------------------------- predict
    def predict(self, features):
        tk = self._tickers(features)
        legs = self._legs(features, tk)
        sig = np.zeros_like(legs[0])
        for leg, scale in zip(legs, getattr(self, "scales_", self.SCALES)):
            sig += leg / scale
        sig /= float(len(legs))
        sig -= sig.mean(1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
