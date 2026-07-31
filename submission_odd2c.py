"""Competition 5 candidate: ODD-2c "WFMVO" — mean-variance weighting of a
reversal-orthogonalised odd basis.

WHAT THIS IS
    Twelve ODD, own-name basis columns, each exactly projected orthogonal to
    cross-sectional reversal AT EVERY BAR, combined by an in-period
    mean-variance (max-Sharpe) weight vector fitted on the period's TRAIN rows
    only.

      L1..L4  csrank of the own-name return at lags 1..4        (degree-1 odd)
      X3..X6  csrank of Feature.3, .4, .5, .6                   (the non-return views)
      D1..D4  csrank of r*vol60, r/vol60, r*csdev^2, r^2*r[t-1] (degree-3 odd)

    Feature.2 is deliberately EXCLUDED: it equals the lag-1 return at corr 0.9934,
    i.e. it is column L1 already.
    No lead-lag/graph column is used: the obvious `peer` construction needs a
    full-period correlation matrix and is in-period lookahead.

    Orthogonalisation: at each bar t the column is cross-sectionally demeaned and
    the component along unit(demean(csrank(Feature.1[t]))) is removed exactly.
    This is a per-bar, time-t-only operation, so the signal is reversal-orthogonal
    BY CONSTRUCTION rather than by fitting.

PARAMETERS (all pre-declared, none tuned)
    LAM  = 1.0   ridge shrinkage of the covariance diagonal in the MVO solve
    P    = 60    rolling window for vol60
    TAIL = 64    rows of train features carried into predict() to warm the
                 rolling window and the lags (causal; nothing after the boundary)
    Basis membership and the exclusion of Feature.2 and of the graph column were
    fixed in writing before any number was measured.

CAUSALITY
    train() sees only training rows. The MVO weights use bar t-1 features against
    the bar-t realised cross-section, with the bar-t return taken as Feature.1
    (Feature.1 IS the contemporaneous return, corr 0.9985) — a contemporaneous
    quantity used only to score a signal formed at t-1, so no lookahead.
    predict() recomputes the same twelve columns on the validation block, warmed
    by the stored TAIL of training rows, and applies the frozen weights.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Odd2cPredictor(Predictor):
    LAM = 1.0
    P = 60
    TAIL = 64
    MINP = 10

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _mat(features, name, tk):
        return np.nan_to_num(features[name][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _csrank(a):
        j = a.shape[1]
        r = pd.DataFrame(a).rank(axis=1, method="average").to_numpy()
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _dm(a):
        return a - a.mean(1, keepdims=True)

    @classmethod
    def _lag(cls, a, k):
        s = np.zeros_like(a)
        if len(a) > k:
            s[k:] = a[:-k]
        return s

    @classmethod
    def _basis(cls, Fd):
        """Fd: dict name -> (T, J) array. Returns list of 12 orthogonalised columns."""
        r = Fd["Feature.1"]
        v = np.nan_to_num(pd.DataFrame(r).rolling(cls.P, min_periods=cls.MINP).std().to_numpy())
        csd = r - r.mean(1, keepdims=True)
        raw = ([cls._csrank(cls._lag(r, k)) for k in (1, 2, 3, 4)]
               + [cls._csrank(Fd[f"Feature.{i}"]) for i in (3, 4, 5, 6)]
               + [cls._csrank(r * v), cls._csrank(r / (v + 1e-6)),
                  cls._csrank(r * csd ** 2), cls._csrank((r ** 2) * cls._lag(r, 1))])
        u = cls._dm(cls._csrank(r))
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        out = []
        for c in raw:
            cd = cls._dm(c)
            out.append(cd - (cd * u).sum(1, keepdims=True) * u)
        return out

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self._tk = tk
        Fd = {f"Feature.{i}": self._mat(features, f"Feature.{i}", tk) for i in range(1, 7)}
        cols = self._basis(Fd)
        k = len(cols)

        # per-bar PnL of each leg: g[t, m] = <col_m[t-1], demean(R[t])>, R := Feature.1
        Rd = self._dm(Fd["Feature.1"])
        n = Rd.shape[0]
        g = np.zeros((n - 1, k))
        for m, c in enumerate(cols):
            g[:, m] = (c[:-1] * Rd[1:]).sum(1)
        g = g[self.P:]                      # drop the vol60 warm-up region

        mu = g.mean(0)
        S = np.cov(g, rowvar=False)
        w = np.linalg.solve(S + self.LAM * np.diag(np.diag(S)), mu)
        nw = np.linalg.norm(w)
        self._w = w / nw if nw > 0 else w

        # causal warm-up tail for predict()
        self._tailF = {kk: v[-self.TAIL:].copy() for kk, v in Fd.items()}

    # ---------------------------------------------------------------- predict
    def predict(self, features):
        tk = self._tickers(features)
        Fd = {f"Feature.{i}": self._mat(features, f"Feature.{i}", tk) for i in range(1, 7)}
        nv = Fd["Feature.1"].shape[0]
        tail = getattr(self, "_tailF", None)
        if tail is not None and tail["Feature.1"].shape[1] == Fd["Feature.1"].shape[1]:
            nt = tail["Feature.1"].shape[0]
            Fd = {kk: np.vstack([tail[kk], Fd[kk]]) for kk in Fd}
        else:
            nt = 0
        cols = self._basis(Fd)
        sig = sum(self._w[m] * cols[m] for m in range(len(cols)))[nt:]
        sig = self._dm(np.nan_to_num(sig))
        return pd.DataFrame(sig, index=features.index[:nv], columns=tk)
