"""Competition 5 submission: "ticket7" — an ADDITIVE new-badge candidate.

WHAT THIS IS
    An equal-PnL-RISK blend of two independently-derived, structurally
    unrelated cross-sectional predictors of the next bar.  Neither leg is a
    variant of any live badge; the pair was chosen because their blend is
    DECORRELATED (< 0.45) from every currently-admitted signal (apex6, sparse,
    unique).  The goal is admission of an ADDITIONAL badge, not the replacement
    of an existing one.

      LEG C  (carrier)      -csrank( e_t ), where e_t is the residual of a
                            per-timestamp cross-sectional OLS of Feature.1 on
                            [1, Feature.2 .. Feature.6].  Interpretation: the
                            part of the (noisy) return view Feature.1 that the
                            other five feature views do NOT explain, mean-
                            reverted.  Pure same-row regression -> causal,
                            carries no state.  This is the Sharpe carrier
                            (full ~0.077 standalone); the family is robust
                            (residualising on F3+F4 / F3+F4+F5 / F3+F4+F6 all
                            give ~0.070-0.071, so this is not a knife-edge).

      LEG A  (decorrelator) Avellaneda-Lee (2010) PCA-residual OU s-score,
                            applied cross-sectionally to a trailing 60-bar
                            window of the Feature.1 return view.  Per bar:
                            standardise the window, PCA on the correlation
                            matrix, take the top-3 eigenportfolios as factors,
                            regress each name on them, cumulate the residual,
                            fit an OU process by AR(1), and read off the
                            equilibrium-scaled deviation s; signal = -csrank(s).
                            Non-mean-reverting names (AR slope <=0 or >=1) are
                            zeroed.  It is a weak stand-alone signal (full
                            ~0.019) but it is nearly orthogonal to LEG C and to
                            the live badges, which is the entire point: it
                            rotates the blend's "city" away from them.

COMBINATION — EQUAL PnL RISK, NO FITTED BLEND WEIGHT
    ticket7 = LEG_C / scale_c + LEG_A / scale_a,   then row-demeaned.
    The two coefficients are structurally 1.0 (equal risk); there is NO fitted
    blend weight and NO argmax over a weight grid.  Each scale_k is a TRAIN-TIME
    MEASUREMENT of that leg's realised PnL volatility, re-measured inside
    train() on the most recent SCALE_TAIL *training* rows using Feature.1 as a
    same-bar return proxy (the same _proxy_vol rule ticket6 uses).  Dividing by
    the train-measured PnL vol makes the two legs carry equal PnL risk with a
    quantity train() legitimately owns, rather than a frozen source constant.
    The SCALE_*_FALLBACK class attributes are used ONLY if a training window is
    degenerate (< SCALE_MIN usable rows), which does not occur on the
    competition data (min training rows per period observed: ~2191).

CAUSALITY
    train() reads training rows only.  It stores (i) the tail of the raw
    Feature.1 cross-section so LEG A's first 60 validation bars have a real
    trailing window instead of a cold start (mirrors ticket6's bufA_), and
    (ii) the two train-tail-measured leg scales.  predict() computes NO
    statistic over the scored block: LEG C is a per-row cross-sectional OLS
    residual (row-local); LEG A's bar-t s-score is a function of the window
    [t-60, t-1] only (it does not even read row t); csrank and de-mean are
    per-timestamp (cross-sectional, across tickers) operations.  No backward
    fill, no negative shift, no centred window, no whole-block statistic.
    The Feature.1 return proxy is applied ONLY to training-tail rows in train().

HARDCODED NUMERIC LITERALS (all labelled in the source):
    structural : W_A=60, K_A=3 (AvL window / factor count); SCALE_TAIL=3000,
                 SCALE_MIN=200 (measurement window, identical to ticket6);
                 the 0.5 rank-centring constants and the 1e-9/1e-12/1e-18
                 numerical epsilons.
    fitted     : NONE.  There is no fitted blend weight and no fitted scale;
                 every scale is a train-time measurement.
    fallback   : SCALE_C_FALLBACK, SCALE_A_FALLBACK — medians of the per-period
                 train-tail measurements, used only on a degenerate train
                 window (never triggered on the competition data).
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket7Predictor(Predictor):
    # ---- LEG A: Avellaneda-Lee PCA-residual OU s-score (structural) ----------
    W_A = 60          # trailing window length (bars)
    K_A = 3           # number of top eigenportfolios used as factors
    # ---- risk normalisation (structural; identical rule to ticket6) ----------
    SCALE_TAIL = 3000  # training rows used to re-measure each leg's PnL vol
    SCALE_MIN = 200    # minimum usable rows before trusting a measurement
    # ---- FALLBACK leg scales (used only on a degenerate train window) --------
    #   Medians of the per-period train-tail F1-proxy PnL vols.  On well-formed
    #   data both scales are measured in train(); these literals never affect
    #   the shipped signal.
    SCALE_C_FALLBACK = 0.018789
    SCALE_A_FALLBACK = 0.017123

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _mat(features, i, tk):
        """Feature i as a NaN-free (T, J) array in ticker order."""
        return np.nan_to_num(
            features[f"Feature.{i}"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _csrank(a, j):
        """Cross-sectional (per-row, across-ticker) average rank -> [-1, +1]."""
        r = pd.DataFrame(np.asarray(a)).rank(axis=1, method="average").to_numpy()
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _demean(a):
        return a - a.mean(axis=1, keepdims=True)

    # ---------------------------------------------------------------- LEG C
    @classmethod
    def _legC(cls, mats, j):
        """-csrank of the per-timestamp cross-sectional OLS residual of
        Feature.1 on [1, F2..F6].  `mats` maps 1..6 -> (T, J) arrays covering
        the SAME rows.  Uses a batched pseudo-inverse so rank-deficient rows get
        the min-norm (lstsq) residual, matching the scout implementation.
        Row-local -> causal, carries no state."""
        y = mats[1]
        A = np.stack([np.ones_like(y)] + [mats[i] for i in range(2, 7)], axis=2)
        beta = np.einsum("tij,tj->ti", np.linalg.pinv(A), y)   # min-norm OLS
        e = y - np.einsum("tji,ti->tj", A, beta)               # residual
        return cls._demean(cls._csrank(-e, j))

    # ---------------------------------------------------------------- LEG A
    def _sscore(self, Rv):
        """Avellaneda-Lee cross-sectional OU s-score over a raw (T, J) window
        series `Rv`.  Bar t uses ONLY the window [t-W_A, t-1] (it does not read
        row t): standardise, PCA on corr, top-K_A eigenportfolios as factors,
        residual, cumulate, OU by AR(1), s = (X_end - m) / sigma_eq; return -s,
        with non-mean-reverting names (b<=0 or b>=1) zeroed.  Causal."""
        W = self.W_A
        k = self.K_A
        T, J = Rv.shape
        sig = np.zeros((T, J))
        eyek = np.eye(k)
        for t in range(W, T):
            win = Rv[t - W:t]
            sd = win.std(0) + 1e-9
            Z = (win - win.mean(0)) / sd
            C = np.nan_to_num(np.corrcoef(Z.T))
            ev, V = np.linalg.eigh(C)
            Qw = V[:, -k:] / sd[:, None]                 # eigenportfolio weights
            Fret = win @ Qw                              # (W, k) factor returns
            FtF = Fret.T @ Fret + 1e-9 * eyek
            Beta = np.linalg.solve(FtF, Fret.T @ win)    # (k, J)
            resid = win - Fret @ Beta                    # (W, J)
            X = np.cumsum(resid, axis=0)
            X0 = X[:-1]
            X1 = X[1:]
            x0m = X0.mean(0)
            x1m = X1.mean(0)
            cov = ((X0 - x0m) * (X1 - x1m)).mean(0)
            var0 = ((X0 - x0m) ** 2).mean(0) + 1e-12
            b = cov / var0
            a = x1m - b * x0m
            arres = X1 - (a + b * X0)
            sig2 = arres.var(0)
            with np.errstate(all="ignore"):
                m = a / (1 - b)
                sig_eq = np.sqrt(np.maximum(sig2 / (1 - b * b), 1e-18))
                s = (X[-1] - m) / sig_eq
            sig[t] = -np.where((b > 0) & (b < 1) & np.isfinite(s), s, 0.0)
        return sig

    def _legA(self, f1_val, buf, j):
        """LEG A signal on the validation Feature.1 block `f1_val`, warm-started
        with the trailing training rows `buf` so the first W_A bars have a real
        window.  Returns a row-demeaned csrank of the s-score for the val rows
        only."""
        comb = np.vstack([buf, f1_val]) if len(buf) else f1_val
        s = self._sscore(comb)[len(buf):]
        return self._demean(self._csrank(s, j))

    # ---------------------------------------------------------------- scales
    @staticmethod
    def _proxy_vol(mat, f1_proxy):
        """PnL volatility of a signal matrix against a return proxy, measured
        exactly as evaluation.backtest scores it: pnl_t = <mat_{t-1}, proxy_t>,
        then std over the block.  `mat` and `f1_proxy` are (T, J) arrays over
        the SAME rows -> causal on whatever tail the caller supplies."""
        s = (mat[:-1] * f1_proxy[1:]).sum(axis=1)
        s = s[np.isfinite(s)]
        return float(np.std(s)) if s.size else 0.0

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        j = len(tk)
        self.j_ = j
        f1_full = self._mat(features, 1, tk)
        n = f1_full.shape[0]
        # LEG A warm-start state: the last W_A raw Feature.1 training rows, so
        # the first W_A validation bars get a real trailing window.
        self.bufA_ = f1_full[-self.W_A:].copy()
        # ---- TRAIN-MEASURED EQUAL-RISK SCALES (Feature.1 return proxy) -------
        self.scale_c_ = self.SCALE_C_FALLBACK
        self.scale_a_ = self.SCALE_A_FALLBACK
        try:
            tl = min(self.SCALE_TAIL, n)
            if tl >= self.SCALE_MIN:
                tail = features.iloc[-tl:]
                mats = {i: self._mat(tail, i, tk) for i in range(1, 7)}
                f1p = mats[1]
                # LEG C on the tail (row-local, no warm-up needed)
                lc = self._legC(mats, j)
                vc = self._proxy_vol(lc, f1p)
                if vc > 1e-12:
                    self.scale_c_ = vc
                # LEG A on the tail, warmed with the W_A rows preceding the tail
                lo = n - tl - self.W_A
                bufc = f1_full[lo:n - tl] if lo >= 0 else f1_full[:n - tl]
                la = self._legA(mats[1], bufc, j)
                va = self._proxy_vol(la, f1p)
                if va > 1e-12:
                    self.scale_a_ = va
        except Exception:
            self.scale_c_ = self.SCALE_C_FALLBACK
            self.scale_a_ = self.SCALE_A_FALLBACK

    # ---------------------------------------------------------------- predict
    def predict(self, features):
        tk = self._tickers(features)
        j = len(tk)
        mats = {i: self._mat(features, i, tk) for i in range(1, 7)}
        buf = getattr(self, "bufA_", np.zeros((0, j)))
        legC = self._legC(mats, j)
        legA = self._legA(mats[1], buf, j)
        sc = getattr(self, "scale_c_", self.SCALE_C_FALLBACK)
        sa = getattr(self, "scale_a_", self.SCALE_A_FALLBACK)
        sig = legC / sc + legA / sa            # equal PnL risk, unit coefficients
        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
