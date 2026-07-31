"""Competition 5 — "levy": a completely INDEPENDENT signal.

Signed Lévy-area lead-lag network-momentum graph (distilled from Li & Ferreira 2025,
"Follow the Leader", arXiv 2501.07135 — the parameter-free eq.1 primitive ONLY; the
paper's learned graph / DTW / (alpha,beta) grid are dropped as high-selection).

MECHANISM (a genuinely different graph than ticket2's linear VAR(1)):
  For each trailing window, the signed discrete Lévy area of two return paths,
    L_ij = 0.5 * (P_i·dP_j - P_j·dP_i)  summed over the window   (P = demeaned cumulative
  return path, dP = the per-bar return), is an antisymmetric matrix whose sign says which
  name LEADS the other (path/ordering information, not linear covariance). Averaged over a
  pre-declared set of lookbacks, sparsified to each lagger's top-K leaders, row-normalised
  to A, the spillover signal is s_i = sum_j A_ij * momentum_j — a laggard inherits its
  leaders' recent trend (network MOMENTUM). Cross-sectionally ranked and de-meaned.

WHY IT IS INDEPENDENT: it is graph-*momentum* (trend spillover), orthogonal to the
crowded reversal axis AND to ticket2's graph-*reversion*. Measured corr: reversal 0.06,
apex6 0.09, sparse 0.07, unique 0.09, ticket2 0.01 — the lowest of any signal built this
project (implied blocker 0.007). It is WEAK standalone (full ~0.015); its role is an
independent CORE to club with a strength-carrying leg, not a standalone flagship.

CONSTANTS: all pre-declared, no fit, no argmax, no training.
  LOOKS = (22,44,66,88,110,132) lookbacks (averaged over, never selected from);
  TOPK = 5 leaders per laggard;  MOM_W = 22 momentum window.

CAUSALITY: the signal at bar t uses only bars < t (trailing windows). train() stores the
tail of the training Feature.1 cross-section so the first validation bars have a real
window (warm-up), never a look-ahead.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class LevyPredictor(Predictor):
    LOOKS = (22, 44, 66, 88, 110, 132)
    TOPK = 5
    MOM_W = 22

    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _f1(features, tk):
        return np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _csrank(a, j):
        r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        # warm-up buffer: the tail of the standardised-free F1 cross-section, enough for
        # the largest lookback plus the momentum window.
        f1 = self._f1(features, tk)
        self.buf_ = f1[-(max(self.LOOKS) + self.MOM_W + 2):].copy()

    def _levy_signal(self, rv):
        """rv: (nv, J) validation F1 returns. Returns (nv, J) raw spillover signal.
        Warm-started with self.buf_ so early rows have a full trailing window."""
        buf = getattr(self, "buf_", np.zeros((0, rv.shape[1])))
        x = np.vstack([buf, rv])
        t0 = len(buf)
        nv, J = rv.shape
        Wmax = max(self.LOOKS)
        out = np.zeros((nv, J))
        for u in range(nv):
            t = t0 + u                       # absolute index in x of the bar being predicted
            if t < Wmax:                     # not enough history even with the buffer
                continue
            L = np.zeros((J, J))
            for W in self.LOOKS:
                seg = x[t - W:t]             # strictly-prior window (rows < t)
                P = np.cumsum(seg, axis=0)
                P = P - P.mean(axis=0)
                L += 0.5 * (P.T @ seg - seg.T @ P)
            A = (L / len(self.LOOKS)).T      # A[i,j] = how much j leads i
            M = np.zeros_like(A)
            for i in range(J):
                idx = np.argsort(-np.abs(A[i]))[:self.TOPK]
                M[i, idx] = A[i, idx]
            M = M / (np.abs(M).sum(axis=1, keepdims=True) + 1e-12)
            mom = x[t - self.MOM_W:t].sum(axis=0)
            out[u] = M @ mom
        return out

    def predict(self, features):
        tk = self._tickers(features)
        rv = self._f1(features, tk)
        s = self._levy_signal(rv)
        j = len(tk)
        sig = self._csrank(s, j)
        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
