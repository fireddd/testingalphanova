"""VERIFY t13: resid_allfeat carrier + apexP Ledoit-Wolf position map (LAM=0.50).

Independent re-implementation from the PRE-DECLARATION text, not by importing
research/agents/t13/riskmap.py. Toggle MODE to compare BASE vs LW.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class V13LWPredictor(Predictor):
    COV_WIN = 250
    COV_MINP = 60
    LAM = 0.50
    TAIL = 600
    MODE = "LW"          # "LW" or "BASE"

    def __init__(self):
        self._tail = None

    # ------------------------------------------------------------------
    def train(self, features, target):
        tk = sorted(features.columns.get_level_values(1).unique(),
                    key=lambda c: int(c.split(".")[-1]))
        f1 = features["Feature.1"][tk].to_numpy(dtype=np.float64)
        self._tail = np.nan_to_num(f1[-self.TAIL:])

    # ------------------------------------------------------------------
    def predict(self, features):
        tk = sorted(features.columns.get_level_values(1).unique(),
                    key=lambda c: int(c.split(".")[-1]))
        F = [np.nan_to_num(features[f"Feature.{k}"][tk].to_numpy(dtype=np.float64))
             for k in range(1, 7)]
        T, J = F[0].shape

        # --- carrier: -csrank(cross-sectional OLS residual of F1 on [1,F2..F6])
        y = F[0] - F[0].mean(1, keepdims=True)
        X = np.stack(F[1:], axis=2)
        X = X - X.mean(1, keepdims=True)
        A = np.einsum("tjk,tjl->tkl", X, X) + 1e-12 * np.eye(5)
        b = np.einsum("tjk,tj->tk", X, y)
        beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        e = y - np.einsum("tjk,tk->tj", X, beta)
        r = pd.DataFrame(e).rank(axis=1, method="average").to_numpy()
        P = -((r - 0.5 * (J + 1)) / (0.5 * (J - 1)))
        P = P - P.mean(1, keepdims=True)

        if self.MODE == "BASE":
            return pd.DataFrame(P, index=features.index, columns=tk)

        # --- causal rolling Ledoit-Wolf shrunk covariance of Feature.1
        tail = self._tail if self._tail is not None else np.zeros((0, J))
        Z = np.vstack([tail, F[0]])
        Sig = self._lw_cov(Z)[len(tail):]
        prec = np.linalg.inv(Sig)

        q = np.einsum("tij,tj->ti", prec, P)
        nq = np.sqrt((q * q).sum(1, keepdims=True)) + 1e-18
        npn = np.sqrt((P * P).sum(1, keepdims=True))
        q = q * (npn / nq)
        out = (1.0 - self.LAM) * P + self.LAM * q
        out = out - out.mean(1, keepdims=True)
        return pd.DataFrame(out, index=features.index, columns=tk)

    # ------------------------------------------------------------------
    def _lw_cov(self, X):
        """Rolling window Ledoit-Wolf shrunk covariance, window ends at row t inclusive."""
        win, minp = self.COV_WIN, self.COV_MINP
        T, J = X.shape
        eye = np.eye(J)
        out = np.empty((T, J, J))
        for t in range(T):
            lo = max(0, t - win + 1)
            W = X[lo:t + 1]
            n = W.shape[0]
            if n < minp:
                out[t] = eye
                continue
            mu = W.mean(0)
            Wc = W - mu
            S = (Wc.T @ Wc) / n
            m = np.trace(S) / J
            d2 = ((S - m * eye) ** 2).sum() / J
            # bbar2 = mean over t of ||x_t x_t' - S||_F^2 / J
            q = (W * W).sum(1)
            bbar2 = ((q * q).sum() / n - (S * S).sum()) / (n * J)
            b2 = min(max(bbar2, 0.0), d2)
            w = b2 / d2 if d2 > 1e-30 else 1.0
            Sig = (1.0 - w) * S + (w * m) * eye
            out[t] = Sig + 1e-6 * max(np.trace(Sig) / J, 1e-18) * eye
        return out
