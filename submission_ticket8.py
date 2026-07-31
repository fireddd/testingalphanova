"""Competition 5 submission: "ticket8" — build-on-ticket2 podium shot (NO apex).

WHAT THIS IS
    A FROZEN equal-RISK sum of three conceptual legs, row-demeaned.  The carrier
    is ticket2 (the only signal that transferred well to the platform, 0.769) and
    the two additive legs are graph-orthogonal decorrelators borrowed verbatim
    from ticket7 (residual reversion + Avellaneda-Lee OU).  Their job is to push
    ticket8's correlation against the external GRAPH blocker under the 0.50
    admission gate WITHOUT a runtime-fitting train() — the ticket6 evidence said a
    fitting train() dragged transfer 0.55 -> 0.44, so every scale here is a FROZEN
    CONSTANT and train() only prepares warm-up state.

      L_t2   ticket2's full signal  = the "two-graph" ticket: a price-space
             grid-averaged VAR(1) lead-lag leg (_legA) plus a rank-space
             own-name-zeroed ridge VAR leg (_legB), summed at equal risk.  This
             is the Sharpe carrier and the transfer carrier (~74% of the blend).
             Inlined verbatim as the non-inheriting nested helper `_Ticket2`
             (byte-for-byte the logic of submission_ticket2.Ticket2Predictor,
             the same way submission_apex8 inlines its ticket2 sleeve).
      L_resid ticket7 legC = -csrank( per-row cross-sectional OLS residual of
             Feature.1 on [1, F2..F6] ).  Row-local, stateless, causal.
      L_avl  ticket7 legA = Avellaneda-Lee (2010) PCA-residual OU s-score
             (W=60, k=3), warm-started from the last 60 training Feature.1 rows.
             Bar t uses only the window [t-60, t-1]; causal.

BLEND — FROZEN EQUAL RISK, NO FITTED WEIGHT, NO RUNTIME SCALE MEASUREMENT
    sig = L_t2 / SCALE_T2 + L_resid / SCALE_RESID + L_avl / SCALE_AVL,  row-demean.
    The three SCALE_* are FROZEN CLASS CONSTANTS (per-leg sel75 PnL vols measured
    once offline), NOT re-measured in train().  This is deliberate: ticket2's 0.769
    transfer came from a SIMPLE frozen construction, and a fitting train() is
    exactly what hurt ticket6's transfer.  train() therefore prepares ONLY leg-A's
    60-row warm-up buffer and ticket2's own leg-A/leg-B state; it measures nothing
    over the scored block and no scale.

CAUSALITY
    train() reads training rows only: it calls the ticket2 sleeve's train (which
    stores leg A's standardised tail buffer bufA_ and leg B's two ridge coefficient
    matrices WB_) and stores the last 60 raw Feature.1 training rows for the AvL
    leg's warm start.  predict() computes NO statistic over the scored block:
    L_resid is a per-row cross-sectional OLS residual (row-local); L_avl's bar-t
    s-score is a function of the window [t-60, t-1] only (it does not read row t);
    L_t2's two legs recurse causally forward one row at a time.  csrank and de-mean
    are per-timestamp cross-sectional operations.  No backward fill, no negative
    shift, no whole-block statistic.

HARDCODED NUMERIC LITERALS (all labelled):
    frozen-scale : SCALE_T2, SCALE_RESID, SCALE_AVL (per-leg sel75 PnL vols).
    structural   : W_A=60, K_A=3 (AvL window / factor count); ticket2's WGRID,
                   LAM_A, MINP, TAILS, LAM_B, KLAG, HM; the 0.5 rank-centring
                   constants and the 1e-9/1e-12/1e-18 numerical epsilons.
    fitted       : NONE.  No fitted blend weight, no runtime-measured scale.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket8Predictor(Predictor):
    """ticket2 carrier + ticket7 resid + ticket7 AvL, frozen equal-risk blend."""

    # ---- FROZEN equal-risk scales (per-leg sel75 PnL vol; NOT re-measured) ----
    SCALE_T2 = 0.04933074223312276     # frozen-scale: ticket2 full-signal PnL vol
    SCALE_RESID = 0.018357239918563376  # frozen-scale: ticket7 legC PnL vol
    SCALE_AVL = 0.01674839194002865     # frozen-scale: ticket7 legA (AvL) PnL vol

    # ---- LEG A (AvL) structural constants -----------------------------------
    W_A = 60          # trailing window length (bars)
    K_A = 3           # number of top eigenportfolios used as factors

    # ====================================================================
    #  ticket2 sleeve — verbatim from submission_ticket2.Ticket2Predictor,
    #  inlined as a non-inheriting helper (mirrors submission_apex8._Ticket2).
    # ====================================================================
    class _Ticket2:
        """The two-graph ticket: price-space VAR(1) lead-lag leg + rank-space
        own-name-zeroed ridge VAR leg, each normalised to unit cross-sectional
        dispersion per row and summed 1:1.  No fitted weights."""

        # ---- leg A: price-space VAR(1) lead-lag graph -----------------------
        WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
        LAM_A = 0.10
        MINP = 250
        # ---- leg B: rank-space own-name-zeroed ridge VAR --------------------
        TAILS = (6000, 25000)
        LAM_B = 0.10
        KLAG = 2      # lags 0 and 1 in the design
        HM = 5        # rows held back at the tail so the design never runs off the end

        # ------------------------------------------------------------- utils
        @staticmethod
        def _tickers(features):
            cols = [c for c in features.columns if c[0] == "Feature.1"]
            return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

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
        def _rank_u(a):
            """Cross-sectional rank of each row mapped to [-1, +1]."""
            j = a.shape[1]
            r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
            return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

        @staticmethod
        def _csrank_c(a):
            """Cross-sectional rank of each row mapped to [-0.5, +0.5]."""
            j = a.shape[1]
            r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
            return r / (j - 1.0) - 0.5

        @staticmethod
        def _design(u, k):
            """[u_t, u_{t-1}, ...] with NaN where the lag does not exist."""
            t, j = u.shape
            x = np.full((t, k * j), np.nan)
            x[:, :j] = u
            for i in range(1, k):
                x[i:, i * j:(i + 1) * j] = u[:-i]
            return x

        @staticmethod
        def _tozc(s):
            """Row-demean, row-standardise, clip; non-finite rows -> 0."""
            bad = ~np.isfinite(s)
            x = np.where(bad, 0.0, s)
            x = x - x.mean(1, keepdims=True)
            x = x / np.maximum(x.std(1, keepdims=True), 1e-12)
            x[bad.any(1)] = 0.0
            return np.clip(x, -3.0, 3.0)

        @staticmethod
        def _unit_row(a):
            """Scale each row to unit cross-sectional dispersion (causal, row-local)."""
            s = a.std(1, keepdims=True)
            return a / np.maximum(s, 1e-12)

        # ------------------------------------------------------------- train
        def train(self, features, target):
            tk = self._tickers(features)
            self.tk_ = tk
            f1 = self._f1(features, tk)
            j = f1.shape[1]
            self.j_ = j

            # --- leg A state: tail of the standardised cross-section --------
            self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()

            # --- leg B state: one ridge coefficient matrix per tail ---------
            tgt = np.nan_to_num(target[tk].to_numpy(dtype=np.float64))
            ntr = len(f1)
            kj = self.KLAG * j
            own = np.zeros((kj, j), dtype=bool)
            for k in range(self.KLAG):
                own[k * j:(k + 1) * j] = np.eye(j, dtype=bool)

            self.WB_ = []
            for tail in self.TAILS:
                lo = max(0, ntr - tail)
                u = self._rank_u(f1[lo:])
                y = tgt[lo:]
                y = y - y.mean(1, keepdims=True)
                y = y / np.maximum(y.std(1, keepdims=True), 1e-12)
                t = len(u)
                if t < kj + self.HM + 8:
                    self.WB_.append(np.zeros((kj, j)))
                    continue
                # design rows: block k is u[K-1-k : T-1-HM-k]; here K = 3 blocks
                # are laid out but only the first KLAG are kept, so the row window
                # is [2 : T-1-HM] and the target is aligned to block 0.
                base = 3 - 1
                x = np.concatenate(
                    [u[base - k:t - 1 - self.HM - k] for k in range(self.KLAG)], axis=1)
                yy = y[base:t - 1 - self.HM]
                a = x.T @ x / len(x)
                b = x.T @ yy / len(x)
                ridge = self.LAM_B * float(np.diag(a).mean())
                w = np.linalg.solve(a + ridge * np.eye(kj), b)
                self.WB_.append(np.where(own, 0.0, w))

        # ----------------------------------------------------------- predict
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

        def _legB(self, f1v):
            """Own-name-zeroed rank-space VAR, averaged over the two tails."""
            u = self._rank_u(f1v)
            x = self._design(u, self.KLAG)
            xn = np.nan_to_num(x)
            badrow = np.isnan(x).any(1)
            out = np.zeros((len(u), self.j_))
            for w in self.WB_:
                s = xn @ w
                s[badrow] = np.nan
                out += self._tozc(s)
            return out / len(self.WB_)

        def predict(self, features):
            tk = self._tickers(features)
            f1 = self._f1(features, tk)
            xv = self._zrow(f1)

            a = self._legA(xv)
            b = self._legB(f1)

            # equal risk in signal space: each leg to unit cross-sectional
            # dispersion at every timestamp, then summed 1:1. No fitted weight.
            sig = self._unit_row(a) + self._unit_row(b)
            sig -= sig.mean(1, keepdims=True)
            return pd.DataFrame(sig, index=features.index, columns=tk)

    # ====================================================================
    #  ticket7 legs — verbatim from submission_ticket7.Ticket7Predictor
    # ====================================================================
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

    # ---------------------------------------------------------------- L_resid
    @classmethod
    def _legC(cls, mats, j):
        """-csrank of the per-timestamp cross-sectional OLS residual of
        Feature.1 on [1, F2..F6].  `mats` maps 1..6 -> (T, J) arrays covering
        the SAME rows.  Uses a batched pseudo-inverse so rank-deficient rows get
        the min-norm (lstsq) residual.  Row-local -> causal, carries no state."""
        y = mats[1]
        A = np.stack([np.ones_like(y)] + [mats[i] for i in range(2, 7)], axis=2)
        beta = np.einsum("tij,tj->ti", np.linalg.pinv(A), y)   # min-norm OLS
        e = y - np.einsum("tji,ti->tj", A, beta)               # residual
        return cls._demean(cls._csrank(-e, j))

    # ---------------------------------------------------------------- L_avl
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
        """L_avl on the validation Feature.1 block `f1_val`, warm-started with
        the trailing training rows `buf` so the first W_A bars have a real
        window.  Returns a row-demeaned csrank of the s-score for the val rows
        only."""
        comb = np.vstack([buf, f1_val]) if len(buf) else f1_val
        s = self._sscore(comb)[len(buf):]
        return self._demean(self._csrank(s, j))

    # ------------------------------------------------------------------ API
    def __init__(self):
        self._t2 = self._Ticket2()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # WARM-UP STATE ONLY (no scale measurement, no fitting):
        #   * ticket2 sleeve stores leg A's bufA_ and leg B's WB_ (trains on the
        #     rows handed in here).
        #   * L_avl stores the last W_A raw Feature.1 training rows so its first
        #     W_A validation bars get a real trailing window (mirrors ticket7).
        self._t2.train(features, target)
        tk = self._tickers(features)
        self.tk_ = tk
        self.j_ = len(tk)
        f1_full = self._mat(features, 1, tk)
        self.bufA_ = f1_full[-self.W_A:].copy()
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        j = len(tk)
        mats = {i: self._mat(features, i, tk) for i in range(1, 7)}

        # L_resid (ticket7 legC) — row-local, stateless
        legC = self._legC(mats, j)
        # L_avl (ticket7 legA) — warm-started AvL OU s-score
        buf = getattr(self, "bufA_", np.zeros((0, j)))
        legA = self._legA(mats[1], buf, j)
        # L_t2 (ticket2 full signal) — carrier
        pt2 = self._t2.predict(features)
        pt2 = pt2[tk]                          # enforce identical ticker order
        L_t2 = pt2.to_numpy(dtype=np.float64)

        # FROZEN equal-risk blend, then per-row cross-sectional de-mean.
        sig = (L_t2 / self.SCALE_T2
               + legC / self.SCALE_RESID
               + legA / self.SCALE_AVL)
        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk).fillna(0.0)
