"""Competition 5 — t29 "second lead-lag leg" (Track A).

WHAT THIS IS
    A structurally independent lead-lag estimator, built to PAIR with the incumbent
    lead-lag object (submission_ticket2.py), not to replace it.  It is the ONLY signal
    in this file so that it can be priced cleanly.

    LEG 1 "ALL" — antisymmetric multi-lag Levy-area lead-lag, leading mode.
        The signed area between the trailing m-bar path of the ranked Feature.1
        cross-section and its increments,
            A = sum_t w_t ( P_{t-1} q_t^T  -  q_t P_{t-1}^T ),
        i.e. the level-2 path-signature (Levy area) term.  ANTISYMMETRIC BY
        CONSTRUCTION, so its diagonal is exactly zero and it carries no own-name
        reversal or momentum at all; and because P aggregates m bars it sums lead-lag
        information over m lags rather than lag 1 alone.  It is then reduced to its
        LEADING 2-PLANE (every real antisymmetric matrix is a sum of plane rotations):
        A_hat = s (f e^T - e f^T) with e the top eigenvector of -A@A.  That is
        parameter-free and cuts 190 free numbers to 40; the reading is explicit —
        portfolio e LEADS portfolio f.

    LEG 2 "XF" — cross-feature directed marginal Granger graph, diagonal zeroed.
        G_c = sum_t w_t a_t u^c_{t-1}^T for each source feature c in {2,3,4} (a = ranked
        Feature.1).  MARGINAL (bivariate Granger) coefficients: no S = sum b b^T, no
        matrix inverse, no ridge.  All three sources are averaged, not selected.

    Both graphs are estimated on TRAINING rows only and held FIXED across the validation
    block (no runtime refitting).  Neither leg uses `target` at all — both are estimated
    purely from the feature block, so no label/target channel exists.

WHY IT IS STRUCTURALLY DIFFERENT FROM THE INCUMBENT
    The incumbent is a grid-averaged, correlation-normalised, ridge-shrunk VAR(1) on the
    z-scored Feature.1 cross-section, refit on rolling windows and re-ranked.  This file
    changes the estimator (antisymmetric path signature / marginal Granger, no inverse,
    no ridge), the state (a trailing m-bar path, and Features 2-4, instead of the
    instantaneous Feature.1 shock), the reduction (a rank-1 leader/follower plane), and
    the fitting regime (frozen, not rolling).

MEASURED (fast harness, 61,243 validation rows, identical PnL convention to runner.py)
        LEG 1 alone : full +0.0157   corr vs ticket2_pnl +0.099
        LEG 2 alone : full +0.0191   corr vs ticket2_pnl +0.283
        1:1 sum     : full +0.0223 | h1 +0.0247  h2 +0.0198  rec20 +0.0106  hold +0.0178
                      corr vs ticket2_pnl +0.257 ; corr between the two legs +0.055

    HONEST READING: this is a WEAK leg.  At Sharpe 0.0223 and correlation 0.257 against
    an incumbent of 0.0922, the best achievable two-asset combination is 0.0922 — i.e.
    it adds essentially nothing.  See research/agents/t29/ for the full negative result
    and the mechanism (in a cross-sectionally de-meaned universe every row of the lag-1
    cross-moment sums to zero, so own-name reversal and cross-name lead-lag are
    algebraically entangled; only the antisymmetric part is genuinely independent, and
    the antisymmetric part measures ~0.02 on every feature and every horizon tried).

PARAMETERS
    Path horizons m in {5, 20} and halflives {1000, 8000} — both members of each pair are
    INCLUDED and averaged, never selected.  Clip at 3 sd.  The rank-1 reduction and the
    zeroed diagonal are parameter-free structural constraints.  No fitted blend weight:
    the two legs are each scaled to unit cross-sectional dispersion per row and summed 1:1.

CAUSALITY
    Every graph is frozen at the train/validation boundary.  The prediction at row t is a
    fixed linear map of row t and its 19 predecessors (through the trailing path), taken
    from the stored training tail when they precede the block.  No statistic is ever
    computed over the prediction block; every operation is row-wise (axis=1) or a
    strictly-trailing window.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class T29LeadLagPredictor(Predictor):
    MS = (5, 20)                 # trailing path horizons (averaged)
    HLS = (1000.0, 8000.0)       # graph halflives (averaged)
    SRCS = (1, 2, 3)             # 0-based -> Feature.2, Feature.3, Feature.4
    CLIP = 3.0
    TAILMULT = 6.0               # exponential-weight truncation (6 halflives)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _stack(features, tk):
        return np.stack(
            [np.nan_to_num(features["Feature.%d" % (i + 1)][tk].to_numpy(np.float64))
             for i in range(6)], 1)          # (n, 6, k)

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
    def _unit_row(a):
        return a / np.maximum(a.std(1, keepdims=True), 1e-12)

    @staticmethod
    def _lead_mode(A):
        """Leading antisymmetric 2-plane: A_hat = s (f e^T - e f^T)."""
        B = -(A @ A)
        B = 0.5 * (B + B.T)
        w, V = np.linalg.eigh(B)
        e = V[:, -1]
        if e[int(np.argmax(np.abs(e)))] < 0.0:
            e = -e
        f = A @ e
        s = float(np.linalg.norm(f))
        if s < 1e-14:
            return np.zeros_like(A)
        f = f / s
        return s * (np.outer(f, e) - np.outer(e, f))

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
        F = self._stack(features, tk)
        ntr = len(F)
        a = self._rank_u(F[:, 0, :])

        # LEG 1: antisymmetric Levy-area graphs, reduced to the leading plane
        self.A_ = []
        for m in self.MS:
            P = self._path(a, m)
            for hl in self.HLS:
                A = self._ew(a, P, ntr, hl)
                self.A_.append(self._lead_mode(0.5 * (A - A.T)))

        # LEG 2: cross-feature marginal Granger graphs, diagonal zeroed
        self.G_ = []
        for c in self.SRCS:
            u = self._rank_u(F[:, c, :])
            for hl in self.HLS:
                G = self._ew(a, u, ntr, hl)
                self.G_.append(G - np.diag(np.diag(G)))

        # tail of the ranked Feature.1 cross-section needed by the trailing path
        self.tail_ = a[-(max(self.MS) - 1):].copy() if ntr else np.zeros((0, len(tk)))

    # ---------------------------------------------------------------- predict
    def predict(self, features):
        tk = self.tk_ if set(self.tk_) == set(self._tickers(features)) \
            else self._tickers(features)
        F = self._stack(features, tk)
        nv = len(F)
        a = self._rank_u(F[:, 0, :])

        h = len(self.tail_)
        av = np.vstack([self.tail_, a]) if h else a

        leg1 = np.zeros((nv, len(tk)))
        i = 0
        for m in self.MS:
            P = self._path(av, m)[h:]
            for _ in self.HLS:
                leg1 += self._zclip(P @ self.A_[i].T, self.CLIP)
                i += 1

        leg2 = np.zeros((nv, len(tk)))
        i = 0
        for c in self.SRCS:
            u = self._rank_u(F[:, c, :])
            for _ in self.HLS:
                leg2 += self._zclip(u @ self.G_[i].T, self.CLIP)
                i += 1

        sig = self._unit_row(leg1) + self._unit_row(leg2)
        sig = sig - sig.mean(1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
