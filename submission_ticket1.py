"""Competition 5 submission: "slow-graph" diversification ticket.

PURPOSE
    This is not an attempt to beat our flagship on raw Sharpe. It is a
    deliberately DECORRELATED entry: a second, partly independent draw on the
    live-window ranking. Its whole design constraint is to carry as little of
    the crowded short-term-reversal factor as possible while still clearing
    the quality bar.

MECHANISM (four causal blocks, fixed weights, nothing fitted at run time)

  1. ll_rank   — expanding LEAD-LAG graph.  A_ij = E[u_{i,t-1} u_{j,t}] over
                 all pairs seen so far (diagonal removed) on market-removed,
                 per-ticker-standardised, clipped F1.  Today's score is
                 (A^T u_t)_j, cross-sectionally ranked: "who does yesterday's
                 mover lead?".  This block is the diversifier — its stand-alone
                 PnL correlation with the plain reversal factor is NEGATIVE.

  2. f6_dev_z  — fade the deviation of F6 from its own expanding per-ticker
                 mean, scaled by the expanding per-ticker std.  A slow state
                 variable, orthogonal to price reversal.

  3. lagrev_x3 — reversal built from LAGS 1-2 of F1 ONLY (lag 0 deliberately
                 excluded, which is what removes most of the crowded-cluster
                 exposure) plus the surviving F3 innovation e3, where e3 is the
                 per-row OLS residual of F3 on [1, F1, F2].

  4. f3_f1     — cross-sectional rank of (F3 - F1): the slow F3/F1 spread.

  signal = 1.00*f3_f1 + 2.00*ll_rank + 0.75*f6_dev_z + 0.60*lagrev_x3

  The weights are inverse-PnL-volatility (equal-risk) weights across the three
  blocks {f3_f1}, {ll_rank}, {ll+f6z+0.8*lagrev}, rounded to two decimals.
  They are FLAT: every weight can move +-25% and the full-sample Sharpe stays
  inside 0.069-0.073 (structural, not tuned).

train() is a no-op — nothing is fit at run time.  Every operation is causal
(shift / expanding / cumulative / ewm only); no row ever sees its own future.
Predictions are cross-sectionally de-meaned at every timestamp.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket1Predictor(Predictor):
    """Fixed-weight lead-lag / slow-state / lagged-reversal diversification signal."""

    W_F3F1 = 1.00
    W_LL = 2.00
    W_F6Z = 0.75
    W_LAGREV = 0.60

    LL_WARM = 40        # rows before the lead-lag graph is trusted
    CLIP = 3.0          # clip on standardised F1
    EWM_E3 = 2          # span of the F3-innovation smoother
    LAG1_W = 0.90       # weight on F1 lagged 1
    LAG2_W = 0.35       # weight on F1 lagged 2
    E3_W = 1.40         # weight on the smoothed F3 innovation

    # ------------------------------------------------------------- helpers
    def _tickers(self, features):
        return sorted(features.columns.get_level_values(1).unique(),
                      key=lambda c: int(c.split(".")[-1]))

    def _rowdm(self, a):
        return a - a.mean(axis=1, keepdims=True)

    def _csrank(self, a, J):
        """Cross-sectional rank mapped to [-1, 1]. Row-wise, no time mixing."""
        if J < 2:
            return np.zeros_like(a)
        r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
        return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

    def _ewm(self, a, span):
        """Causal EWM with bias correction (only rows <= t enter row t)."""
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(a)
        acc = np.zeros(a.shape[1])
        wsum = 0.0
        for t in range(len(a)):
            acc = (1 - alpha) * acc + a[t]
            wsum = (1 - alpha) * wsum + 1.0
            out[t] = acc / wsum
        return out

    def _lag(self, a, k):
        out = np.zeros_like(a)
        out[k:] = a[:-k]
        return out

    def _exp_std(self, a, min_n=10):
        """Strictly causal expanding per-ticker std (row t uses rows 0..t only)."""
        T = a.shape[0]
        n = (np.arange(T) + 1.0)[:, None]
        s1 = np.cumsum(a, axis=0)
        s2 = np.cumsum(a * a, axis=0)
        v = s2 / n - (s1 / n) ** 2
        sd = np.sqrt(np.maximum(v, 0.0))
        # early rows: no backfill (that would peek forward). Fall back to the
        # running mean-absolute deviation, then to a hard floor.
        mad = np.cumsum(np.abs(a), axis=0) / n
        early = np.arange(T)[:, None] < min_n
        sd = np.where(early & (sd <= 1e-12), mad, sd)
        return np.maximum(sd, 1e-12)

    def _cs_resid2(self, y, x1, x2):
        """Per-row OLS residual of y on [1, x1, x2] across tickers. Causal."""
        ym, x1m, x2m = self._rowdm(y), self._rowdm(x1), self._rowdm(x2)
        a11 = (x1m * x1m).sum(1)
        a12 = (x1m * x2m).sum(1)
        a22 = (x2m * x2m).sum(1)
        b1 = (x1m * ym).sum(1)
        b2 = (x2m * ym).sum(1)
        det = a11 * a22 - a12 * a12 + 1e-18
        be1 = (a22 * b1 - a12 * b2) / det
        be2 = (a11 * b2 - a12 * b1) / det
        return ym - x1m * be1[:, None] - x2m * be2[:, None]

    def _leadlag_rank(self, u, J):
        """Expanding lead-lag adjacency, off-diagonal only, then ranked.

        A at time t is built ONLY from pairs (u_{s-1}, u_s) with s <= t, so the
        score for row t uses no information from t+1 onward.
        """
        T = u.shape[0]
        C1 = np.zeros((J, J))
        sig = np.zeros((T, J))
        for t in range(T):
            if t >= 1:
                C1 += np.outer(u[t - 1], u[t])
            if t >= self.LL_WARM:
                A = C1 / t
                np.fill_diagonal(A, 0.0)
                sig[t] = A.T @ u[t]
        out = self._csrank(sig, J)
        out[:self.LL_WARM + 1] = 0.0
        return out

    # ----------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """No-op: all weights are fixed offline; nothing is fit to the target."""
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        J = len(tk)
        index = features.index
        F = {}
        for i in range(1, 7):
            F[i] = np.nan_to_num(
                features["Feature.%d" % i][tk].values.astype(np.float64))
        f1, f2, f3, f6 = F[1], F[2], F[3], F[6]

        # --- block 1: expanding lead-lag graph on standardised, de-marketed F1
        f1d = self._rowdm(f1)
        u = np.clip(f1d / (self._exp_std(f1d) + 1e-12), -self.CLIP, self.CLIP)
        ll = self._leadlag_rank(u, J)

        # --- block 2: F6 deviation from its own expanding mean, faded
        T = f1.shape[0]
        n_t = (np.arange(T) + 1.0)[:, None]
        f6em = np.cumsum(f6, axis=0) / n_t
        f6z = -self._csrank((f6 - f6em) / (self._exp_std(f6) + 1e-12), J)

        # --- block 3: lag-1/2 reversal + surviving F3 innovation
        e3 = self._cs_resid2(f3, f1, f2)
        lagrev = -self._csrank(
            self.LAG1_W * self._lag(f1, 1) + self.LAG2_W * self._lag(f1, 2)
            - self.E3_W * self._ewm(e3, self.EWM_E3), J)

        # --- block 4: slow F3/F1 spread
        f3f1 = self._csrank(f3 - f1, J)

        out = (self.W_F3F1 * self._rowdm(f3f1)
               + self.W_LL * self._rowdm(ll)
               + self.W_F6Z * self._rowdm(f6z)
               + self.W_LAGREV * self._rowdm(lagrev))
        out = np.nan_to_num(out)
        out = out - out.mean(axis=1, keepdims=True)
        return pd.DataFrame(out, index=index, columns=tk)
