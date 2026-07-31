"""Competition 5 submission: "two-graph" ticket — a SECOND decorrelated badge.

WHAT THIS IS
    An equal-risk sum of two independently-derived, independently-adversarially-
    verified network estimators of the SAME underlying object (next bar's
    cross-section) built in two DIFFERENT spaces:

      LEG A  price space  — a grid-averaged VAR(1) lead-lag coefficient matrix
                            estimated on the standardised, de-marketed return
                            cross-section, applied to today's cross-section.
      LEG B  rank space   — a ridge VAR on cross-sectional RANKS whose OWN-NAME
                            coefficients are set to zero, so it is orthogonal to
                            own-name reversal BY CONSTRUCTION.

    Leg A carries the Sharpe; leg B carries the decorrelation.  Their PnL
    correlation is +0.169, so the sum is close to a variance-weighted average of
    two nearly independent draws on the same signal.

WHY BOTH
    Leg A alone is strong (local full Sharpe 0.0976) but sits at 0.424 PnL
    correlation against `sparse` on the full sample and 0.479 on the first half —
    over the 0.45 admission line on that sub-window.  Leg B alone is weak
    (0.0397) but is the most decorrelated positive-Sharpe object we have found in
    this project (max guard correlation 0.062, and NEGATIVE against `sparse`).
    Summed at equal risk the pair keeps most of the full Sharpe while collapsing
    the worst guard correlation — an order of magnitude more admission headroom
    than either the incumbent or leg A alone.

MEASURED VALUES OF THE CODE AS SHIPPED (harness-verified, independently audited)
    Legs are combined in SIGNAL space (each normalised to unit per-row
    cross-sectional dispersion), which differs slightly from the PnL-space
    inverse-vol combination used during subset enumeration; the two are 0.9965
    correlated but are NOT the same object, so quote these numbers, not the
    enumeration's (0.0894 / 0.276 / 0.294 / legB 0.0397):

      runner.py --full --gauge-fix : Sharpe +0.0922, IC +0.0256 (std 0.2676),
                                     concentration 0.0261, compression -0.9547,
                                     city novelty 10.7 deg
      windows : full 0.0922 | h1 0.1194 | h2 0.0591 | rec20 0.0500
      max guard PnL corr 0.2920 (apex6); vs ticket1 0.2498; leg B alone 0.0406
      max guard corr on the h1 sub-window 0.3140
      platform cross-sectional corr (the metric admission actually uses):
        0.2817 vs apex6, 0.2456 vs ticket1 — roughly half the 0.5 threshold

    EXPECTATION: the platform scores ~0.040-0.050, NOT 0.0922. The full sample is
    dominated by the early high-Sharpe regime (h1 0.1194 vs rec20 0.0500); this
    decay is data-wide, not specific to this signal (plain reversal decays harder,
    0.1028 -> 0.0285). The slot is bought for DECORRELATION, not for Sharpe.

PARAMETERS
    There are no fitted weights.  The two legs are each normalised to unit
    CROSS-SECTIONAL dispersion at every timestamp (a per-row, self-contained,
    causal statistic) and added 1:1.  The only numeric choices are the two ridge
    shrinkages, both of which are structurally required (each design matrix is
    singular because the rows are cross-sectionally demeaned) and both of which
    are flat: leg A's lambda moves full Sharpe by 3% over a 10x range, leg B's by
    0.003 over a 100x range.  The window grid in leg A is AVERAGED, not selected.

CAUSALITY
    train() sees only training rows.  Leg A stores a tail of the standardised
    feature matrix and rebuilds its moment accumulators at the train/val
    boundary, then recurses forward one validation row at a time.  Leg B stores
    two ridge coefficient matrices fitted on training rows only.  No statistic
    computed at row t uses any row > t, and no whole-validation-block statistic
    is ever applied backward.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket2Predictor(Predictor):
    # ---- leg A: price-space VAR(1) lead-lag graph ----------------------------
    WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
    LAM_A = 0.10
    MINP = 250
    # ---- leg B: rank-space own-name-zeroed ridge VAR -------------------------
    TAILS = (6000, 25000)
    LAM_B = 0.10
    KLAG = 2      # lags 0 and 1 in the design
    HM = 5        # rows held back at the tail so the design never runs off the end

    # ------------------------------------------------------------------ utils
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

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        f1 = self._f1(features, tk)
        j = f1.shape[1]
        self.j_ = j

        # --- leg A state: tail of the standardised cross-section -------------
        self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()

        # --- leg B state: one ridge coefficient matrix per tail --------------
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
            # design rows: block k is u[K-1-k : T-1-HM-k]; here K = 3 blocks are
            # laid out but only the first KLAG are kept, so the row window is
            # [2 : T-1-HM] and the target is aligned to block 0.
            base = 3 - 1
            x = np.concatenate(
                [u[base - k:t - 1 - self.HM - k] for k in range(self.KLAG)], axis=1)
            yy = y[base:t - 1 - self.HM]
            a = x.T @ x / len(x)
            b = x.T @ yy / len(x)
            ridge = self.LAM_B * float(np.diag(a).mean())
            w = np.linalg.solve(a + ridge * np.eye(kj), b)
            self.WB_.append(np.where(own, 0.0, w))

    # ---------------------------------------------------------------- predict
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
