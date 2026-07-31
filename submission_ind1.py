"""Competition 5 submission: IND1 — AXIS 1, under-used raw features F2/F5/F6.

Mechanism (one construction, no fitted weights):

  For each under-used feature f in {Feature.2, Feature.5, Feature.6}:
      z_t   = clip((f_t - mean_120(f)) / std_120(f), -4, +4)      per ticker, causal
      leg_t = -( z_t - mean(z_{t-1}, z_{t-2}) )                   row de-meaned
  signal = sum_f leg_f / SCALE_f,  SCALE_f frozen offline (median per-period
           mean |leg| over TRAINING rows only), then row de-meaned.

Why this shape.  Step-A measurement on training rows only (label = next-bar
de-meaned realised return) showed every one of F2/F5/F6 carries a *negative*
cross-sectional level IC with 100% per-period sign stability
(F2 -0.0256, F6 -0.0118, F5 -0.0055).  That level exposure is, however,
exactly what the tree lanes already trade (they consume raw cs-ranks r2/r5/r6),
so a plain level short is ~0.45 PnL-correlated with them.  Subtracting each
name's own two-bar trailing z-level removes the slow, persistent part of the
level - the part the tree lanes span - and keeps the fast innovation, which is
essentially orthogonal to every existing lane (max |corr| 0.13).

Feature.1 is NOT used: not as a driver, not as a neutraliser.  F2/F5/F6 are
already near-orthogonal to F1 cross-sectionally (mean cs-corr of their ranks
vs rank(F1): -0.067 / +0.092 / -0.007), so no beta hedge is needed.

Local walk-forward (61,243 validation rows): full 0.0240, h1 0.0305,
h2 0.0160, rec20 0.0272, hold 0.0116.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ind1Predictor(Predictor):
    """Short the fast innovation in the cross-sectional level of F2, F5, F6."""

    FEATS = (2, 5, 6)
    WIN = 120           # trailing window for the per-ticker z-score
    MINP = 20           # min periods for that window
    CLIP = 4.0          # z clip
    K = 2               # bars of trailing z averaged to define the innovation
    TAIL = 300          # rows of training features carried into predict()
    # frozen: median per-period mean|leg| over TRAINING rows, computed offline
    SCALE = {2: 0.668332, 5: 0.667242, 6: 0.172467}

    def __init__(self):
        self._tail = None

    def _tickers(self, features):
        return sorted(features.columns.get_level_values(1).unique(),
                      key=lambda c: int(str(c).split(".")[-1]))

    def _leg(self, f):
        """Causal z-innovation for one (T, J) feature block."""
        d = pd.DataFrame(f)
        r = d.rolling(self.WIN, min_periods=self.MINP)
        z = ((d - r.mean()) / (r.std() + 1e-12)).clip(-self.CLIP, self.CLIP)
        z = np.nan_to_num(z.values)
        base = np.nan_to_num(
            pd.DataFrame(z).shift(1).rolling(self.K, min_periods=1).mean().values)
        s = -(z - base)
        return s - s.mean(axis=1, keepdims=True)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        tk = self._tickers(features)
        cut = max(0, len(features) - self.TAIL)
        self._tail = {i: np.nan_to_num(
            features["Feature.%d" % i][tk].values.astype(np.float64)[cut:])
            for i in self.FEATS}
        self._tail_tk = list(tk)
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        n = len(features)
        use_tail = (self._tail is not None
                    and list(getattr(self, "_tail_tk", [])) == list(tk))
        out = None
        for i in self.FEATS:
            f = np.nan_to_num(
                features["Feature.%d" % i][tk].values.astype(np.float64))
            if use_tail and len(self._tail[i]):
                f = np.concatenate([self._tail[i], f], axis=0)
            s = self._leg(f)[-n:] / self.SCALE[i]
            out = s if out is None else out + s
        out = out - out.mean(axis=1, keepdims=True)
        pred = pd.DataFrame(out, index=features.index, columns=tk)
        return pred.fillna(0.0)
