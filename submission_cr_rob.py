"""Clean-room arm "robust": non-Gaussian (quadrant / sign) cross-sectional lead-lag.

Economic content
----------------
A name's next-bar *relative* return is predictable from the recent relative returns of
the whole cross-section through a time-varying linear map.  Everything here is an
estimator of that map that never touches a second moment.

Instead of a Pearson covariance we accumulate the lag-1 **cross-quadrant association**

        B_ij  =  E[ sgn(c_{t+1,i}) * sgn(c^(k)_{t,j}) ]

where c is the row-centred contemporaneous return and c^(k) its trailing k-bar sum.
B is a directed Blomqvist beta: it depends on the joint *orthant* distribution only, so
a handful of huge joint moves cannot dominate it the way they dominate a sample
covariance.  Fat tails are, by construction, invisible to it.

B is accumulated with an exponentially weighted first-order recursion (two half-lives)
and used directly as a forecasting map -- it is never inverted, never shrunk toward an
identity, and the row it is applied to is a sign vector, not a rank vector.

Everything is strictly causal: the association used at bar t is the EW state through
bar t-1, and the sign row it multiplies is bar t's own (observable at t; the backtest
applies its own one-bar trade lag).  Predictions are therefore bit-identical when a
block is replaced by a prefix of itself.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from predictor import Predictor


class QuadrantLeadLagPredictor(Predictor):
    """Sign/quadrant association map for cross-sectional relative-value reversion."""

    RETURN_FIELD = "Feature.1"
    HORIZONS = (1, 5)
    HALFLIVES = (250.0, 2500.0)
    WARMUP_ROWS = 12000
    STRIDE = 4096
    SEED = 20260731

    class _EwPipe:
        """Carried state of one exponentially weighted 20x20 association accumulator."""

        def __init__(self, width, halflife):
            decay = 0.5 ** (1.0 / float(halflife))
            self.num = np.array([1.0 - decay], dtype=np.float64)
            self.den = np.array([1.0, -decay], dtype=np.float64)
            self.zi = np.zeros((1, width), dtype=np.float64)
            self.edge = np.zeros(width, dtype=np.float64)

        def push(self, batch):
            """Filter `batch` (m, width); return the *lagged, unit-norm* states (m, width)."""
            state, self.zi = lfilter(self.num, self.den, batch, axis=0, zi=self.zi)
            scale = np.sqrt((state * state).sum(axis=1))
            scale = np.where(scale > 0.0, scale, 1.0)
            lagged = np.empty_like(state)
            lagged[0] = self.edge
            lagged[1:] = state[:-1] / scale[:-1, None]
            self.edge = state[-1] / scale[-1]
            return lagged

    def __init__(self):
        np.random.seed(self.SEED)
        self._names = None
        self._warmup = None

    # ------------------------------------------------------------------ helpers

    def _field(self, features):
        """Pull the return field out of the MultiIndex frame in a fixed column order."""
        panel = features[self.RETURN_FIELD]
        if self._names is not None and list(panel.columns) != self._names:
            keep = [c for c in self._names if c in panel.columns]
            if len(keep) == len(self._names):
                panel = panel.loc[:, keep]
        return panel

    def _centred(self, raw):
        clean = np.nan_to_num(np.asarray(raw, dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
        return clean - clean.mean(axis=1, keepdims=True)

    def _quadrants(self, centred, horizon):
        """Sign of the trailing `horizon`-bar cumulative relative return."""
        if horizon <= 1:
            return np.sign(centred)
        run = np.cumsum(centred, axis=0)
        agg = np.empty_like(centred)
        agg[:horizon] = run[:horizon]
        agg[horizon:] = run[horizon:] - run[:-horizon]
        return np.sign(agg)

    def _responses(self, centred):
        """Next-bar relative-return sign; the final row has no successor and is left flat."""
        nxt = np.zeros_like(centred)
        nxt[:-1] = np.sign(centred[1:])
        return nxt

    def _sweep(self, centred):
        """Stream the whole history once and emit the raw (undemeaned) score."""
        rows, wide = centred.shape
        cell = wide * wide
        answer = np.zeros((rows, wide), dtype=np.float64)
        replies = self._responses(centred)
        for horizon in self.HORIZONS:
            marks = self._quadrants(centred, horizon)
            pipes = [self._EwPipe(cell, hl) for hl in self.HALFLIVES]
            for lo in range(0, rows, self.STRIDE):
                hi = min(lo + self.STRIDE, rows)
                span = hi - lo
                pairs = (replies[lo:hi, :, None] * marks[lo:hi, None, :]).reshape(span, cell)
                here = marks[lo:hi, None, :]
                for pipe in pipes:
                    lagged = pipe.push(pairs).reshape(span, wide, wide)
                    answer[lo:hi] += (lagged * here).sum(axis=2)
        return answer

    # ------------------------------------------------------------------- api

    def train(self, features, target):
        panel = features[self.RETURN_FIELD]
        self._names = list(panel.columns)
        block = np.asarray(panel.to_numpy(), dtype=np.float64)
        if block.shape[0] > self.WARMUP_ROWS:
            block = block[-self.WARMUP_ROWS:]
        self._warmup = np.ascontiguousarray(block)

    def predict(self, features):
        panel = self._field(features)
        live = np.asarray(panel.to_numpy(), dtype=np.float64)
        tail = live.shape[0]
        if self._warmup is not None and self._warmup.shape[1] == live.shape[1]:
            live = np.vstack([self._warmup, live])
        centred = self._centred(live)
        score = self._sweep(centred)[-tail:]
        score = score - score.mean(axis=1, keepdims=True)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = score - score.mean(axis=1, keepdims=True)
        return pd.DataFrame(score, index=features.index, columns=panel.columns)
