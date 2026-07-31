"""Competition 5 — AXIS 4: antisymmetric level-2 path-signature (Levy area) lead-lag.

WHAT THIS IS
    ONE mechanism, nothing else in the file, so it can be priced cleanly as an
    independence leg.

    Let a_t be the cross-sectional rank of Feature.1 at row t, mapped to [-0.5, +0.5],
    and let P_t = sum_{s=t-m+1..t} a_s be the trailing m-bar path of that cross-section.
    The level-2 term of the path signature of (P, a) is the LEVY AREA

        A = sum_t w_t ( P_{t-1} a_t^T  -  a_t P_{t-1}^T ),      w_t = rho^{(T-1)-t}

    A is ANTISYMMETRIC BY CONSTRUCTION: its diagonal is exactly zero, so it carries no
    own-name reversal and no own-name momentum whatsoever — only the directed, signed
    area swept between name i's path and name j's increments.  Because P aggregates m
    bars, A sums lead-lag information over m lags rather than lag 1 alone.

    A is then reduced to its LEADING 2-PLANE.  Every real antisymmetric matrix is a sum
    of plane rotations; keeping the largest one gives

        A_hat = s ( f e^T - e f^T ),   e = top eigenvector of -A@A,   f = A e / |A e|,

    which is parameter-free, cuts 190 free numbers to 40, and has an explicit reading:
    portfolio e LEADS portfolio f.  The prediction is P_t A_hat^T, row-de-meaned.

    Every graph is estimated on TRAINING rows only and frozen across the validation
    block.  `target` is never touched — the object is estimated purely from the feature
    block, so the period-076/077 label permutation cannot contaminate it.

WHY THIS AND NOT submission_t29ll.py
    t29ll is this leg PLUS a second leg (a cross-feature marginal Granger graph over
    names).  That second leg is not a path signature and is not antisymmetric — it is
    off this axis — and it is the entire reason t29ll sits at PnL-correlation +0.257
    against the incumbent lead-lag object ticket2.  It buys +0.007 of Sharpe for +0.16
    of correlation.  On an axis whose scarce resource is independence, not Sharpe, that
    is the wrong trade, so it is removed.  What remains is the pure Levy-area object:

        submission_t29ll (leg1+leg2) : full +0.0223, corr ticket2 +0.257, apex6 +0.052
        THIS FILE        (leg1 only) : full +0.0157, corr ticket2 +0.099, apex6 +0.023
                                       unique +0.008, catboost +0.004, resid -0.002

    Two alternative constructions on this axis were pre-declared and measured, and both
    were REJECTED by the pre-declared rule (research/agents/t31/PREDECLARE_ind4.md):
      * per-name cross-feature Levy area (the other level-2 signature term), 15 pooled-
        OLS coefficients: full -0.0046, weight sign-stability at chance.  Dead.
      * the identical estimator averaged over all six feature channels instead of
        Feature.1 alone: full +0.0142, i.e. no improvement.  Dropped.
    M = 3 constructions examined in total; no grid was searched and no argmax was shipped.

HONEST READING
    This is a WEAK leg and is meant to be.  Data fact 4: in a cross-sectionally de-meaned
    universe every row of the lag-1 cross-moment sums to zero, so own-reversal and
    cross-name lead-lag are algebraically inseparable; only the antisymmetric part is
    genuinely independent, and the antisymmetric part measures ~0.02 on every feature,
    lag and estimator tried.  0.0157 is at that structural ceiling.  It is offered as an
    almost-orthogonal portfolio member, not as a performer.

PARAMETERS
    Path horizons m in {5, 20} and halflives {1000, 8000} — BOTH members of each pair are
    included and averaged, never selected.  Clip at 3 sd.  The rank-1 reduction and the
    exactly-zero diagonal are parameter-free structural constraints.  No fitted weights.

CAUSALITY
    The graphs are frozen at the train/validation boundary.  The prediction at row t is a
    fixed linear map of row t and its 19 predecessors (through the trailing path), taken
    from the stored training tail when they precede the block.  No statistic is ever
    computed over the prediction block; every operation is row-wise (axis=1) or a
    strictly-trailing window, so predictions are chunk-invariant bit-for-bit.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ind4LevyAreaPredictor(Predictor):
    MS = (5, 20)                 # trailing path horizons (averaged, not selected)
    HLS = (1000.0, 8000.0)       # graph halflives (averaged, not selected)
    CLIP = 3.0
    TAILMULT = 6.0               # exponential-weight truncation (6 halflives)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _f1(features, tk):
        return np.nan_to_num(features["Feature.1"][tk].to_numpy(np.float64))

    @staticmethod
    def _rank_u(a):
        j = a.shape[1]
        r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
        return r / (j - 1.0) - 0.5

    @staticmethod
    def _path(u, m):
        """Trailing m-bar sum, expanding at the head (row-local, strictly trailing)."""
        c = np.cumsum(np.vstack([np.zeros((1, u.shape[1])), u]), 0)
        out = c[1:].copy()
        if len(u) > m:
            out[m:] = c[m + 1:] - c[1:-m]
        return out

    @staticmethod
    def _zclip(s, clip):
        s = s - s.mean(1, keepdims=True)
        s = s / np.maximum(s.std(1, keepdims=True), 1e-12)
        return np.clip(s, -clip, clip)

    @staticmethod
    def _lead_mode(A):
        """Leading antisymmetric 2-plane of A, returned in FACTORED form (s, e, f)
        with A_hat = s (f e^T - e f^T).  Keeping the factors rather than the dense
        20x20 matrix is what makes predict() bit-identical under chunking: the dense
        form needs P @ A_hat.T, and BLAS selects a different gemm kernel for a short
        block than for a long one, which perturbs the last bit.  In factored form
        every row is evaluated from its own 20 numbers only."""
        B = -(A @ A)
        B = 0.5 * (B + B.T)
        w, V = np.linalg.eigh(B)
        e = V[:, -1]
        if e[int(np.argmax(np.abs(e)))] < 0.0:
            e = -e
        f = A @ e
        s = float(np.linalg.norm(f))
        if s < 1e-14:
            return 0.0, e, np.zeros_like(e)
        return s, e, f / s

    def _ew(self, Y, B, ntr, hl):
        """sum_{t=1..ntr-1} rho^{(ntr-1)-t} Y[t] B[t-1]^T, truncated at TAILMULT halflives."""
        rho = 0.5 ** (1.0 / hl)
        lo = max(1, ntr - int(self.TAILMULT * hl))
        w = rho ** ((ntr - 1) - np.arange(lo, ntr))
        return (Y[lo:ntr] * w[:, None]).T @ B[lo - 1:ntr - 1]

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        a = self._rank_u(self._f1(features, tk))
        ntr = len(a)

        self.A_ = []
        for m in self.MS:
            P = self._path(a, m)
            for hl in self.HLS:
                A = self._ew(a, P, ntr, hl)
                self.A_.append(self._lead_mode(0.5 * (A - A.T)))

        # tail of the ranked Feature.1 cross-section needed by the trailing path
        self.tail_ = a[-(max(self.MS) - 1):].copy() if ntr else np.zeros((0, len(tk)))

    # ---------------------------------------------------------------- predict
    def predict(self, features):
        tk = self.tk_ if set(self.tk_) == set(self._tickers(features)) \
            else self._tickers(features)
        a = self._rank_u(self._f1(features, tk))
        nv = len(a)

        h = len(self.tail_)
        av = np.vstack([self.tail_, a]) if h else a

        sig = np.zeros((nv, len(tk)))
        i = 0
        for m in self.MS:
            P = self._path(av, m)[h:]
            for _ in self.HLS:
                s, e, f = self.A_[i]
                # P @ A_hat^T with A_hat = s (f e^T - e f^T)  ->  s[(P.e) f - (P.f) e].
                # Two row-wise contractions over the 20 names: no cross-row arithmetic,
                # so the result of a row never depends on how many rows were passed in.
                pe = (P * e[None, :]).sum(1)
                pf = (P * f[None, :]).sum(1)
                sig += self._zclip(s * (pe[:, None] * f[None, :]
                                        - pf[:, None] * e[None, :]), self.CLIP)
                i += 1

        sig = sig - sig.mean(1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
