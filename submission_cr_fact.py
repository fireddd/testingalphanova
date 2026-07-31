"""Reduced-rank (factor-space) cross-sectional lead-lag predictor.

Hypothesis
----------
The predictable part of a name's next-bar *relative* return lives in a handful of
common cross-sectional modes, not in a dense pairwise web.  So: never form an
N x N map at all.  Take an eigen-projection of the trailing relative-return
second moment onto its top ``N_MODES`` directions, estimate the whole lead-lag
dynamic *inside* that low-dimensional space as a factor VAR, then carry the
forecast back to the names through the same loadings.  Reduced rank IS the
regulariser -- there is no shrinkage constant and no matrix is inverted in
name space.

Determinism / causality
-----------------------
* Every statistic is either a single-row (axis=1) reduction or a strictly
  trailing accumulation.  No block-wide statistic is ever taken.
* The composite map ``V A V^T`` is invariant to the sign of, and to any
  orthogonal rotation within, the retained eigenspace, because the same ``V``
  projects in and out.  Chunk-invariance is therefore structural.  Signs are
  additionally pinned for readability.
* The map is refreshed on a schedule counted from the first row of the block
  being predicted, always from rows strictly earlier than the refresh row, so a
  block and any prefix of it produce bit-identical output.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class SpectralLeadLagPredictor(Predictor):
    """Rank-limited factor VAR on trailing cross-sectional relative returns."""

    RELATIVE_FEATURE = "Feature.1"
    N_MODES = 8            # retained eigen-directions (of 20)
    MEMORY = 2000          # trailing rows entering each moment estimate
    ORDER = 1              # factor-VAR lag order
    REFRESH_EVERY = 100    # rows between map re-estimations
    WINSOR = 4.0           # cross-sectional z clip
    RIDGE_REL = 1e-3       # ridge relative to mean retained eigenvalue
    MIN_SAMPLE = 400       # below this the map is not estimated
    TINY = 1e-12

    def __init__(self):
        np.random.seed(0)
        self._carry = None      # encoded trailing rows kept from train()
        self._carry_cols = None

    # ---------------------------------------------------------------- helpers

    def _relative(self, features):
        """Pull the relative-return panel out of the MultiIndex frame."""
        panel = features[self.RELATIVE_FEATURE]
        return list(panel.columns), panel.to_numpy(dtype=np.float64, copy=True)

    def _shape_rows(self, raw):
        """Per-row: centre, scale by own cross-sectional dispersion, winsorise.

        Uses axis=1 reductions only -- row t never sees row s != t.
        """
        centred = raw - raw.mean(axis=1, keepdims=True)
        spread = np.sqrt(np.mean(centred * centred, axis=1, keepdims=True))
        return np.clip(centred / (spread + self.TINY), -self.WINSOR, self.WINSOR)

    def _lag_moments(self, window):
        """Second moments of the window at lags 0 .. ORDER."""
        rows = window.shape[0]
        out = [window.T @ window / rows]
        for gap in range(1, self.ORDER + 1):
            out.append(window[gap:].T @ window[:rows - gap] / (rows - gap))
        return out

    def _basis(self, gram):
        """Top-N_MODES eigenvectors of a symmetric gram matrix, signs pinned."""
        vals, vecs = np.linalg.eigh(gram)
        keep = np.argsort(vals)[::-1][:self.N_MODES]
        basis = np.ascontiguousarray(vecs[:, keep])
        for c in range(basis.shape[1]):
            if basis[int(np.argmax(np.abs(basis[:, c]))), c] < 0.0:
                basis[:, c] = -basis[:, c]
        return basis

    def _spectral_map(self, window):
        """Estimate the composite maps C_1 .. C_ORDER from a trailing window.

        Returns None when the window is too short to be trusted.
        """
        if window.shape[0] < self.MIN_SAMPLE:
            return None
        moments = self._lag_moments(window)
        basis = self._basis(moments[0])
        k = basis.shape[1]
        proj = [basis.T @ m @ basis for m in moments]
        ridge = self.RIDGE_REL * float(np.trace(proj[0])) / max(k, 1)
        span = k * self.ORDER
        gram = np.empty((span, span), dtype=np.float64)
        for a in range(self.ORDER):
            for b in range(self.ORDER):
                gap = b - a
                blk = proj[gap] if gap >= 0 else proj[-gap].T
                gram[a * k:(a + 1) * k, b * k:(b + 1) * k] = blk
        gram[np.diag_indices(span)] += ridge
        rhs = np.hstack([proj[g] for g in range(1, self.ORDER + 1)])
        coef = np.linalg.solve(gram.T, rhs.T).T
        return [basis @ coef[:, g * k:(g + 1) * k] @ basis.T
                for g in range(self.ORDER)]

    def _align_carry(self, cols):
        """Re-order the stored warm-up tail onto the columns of this block."""
        if self._carry is None or self._carry_cols is None:
            return np.zeros((0, len(cols)), dtype=np.float64)
        if self._carry_cols == cols:
            return self._carry
        lookup = {c: i for i, c in enumerate(self._carry_cols)}
        if not all(c in lookup for c in cols):
            return np.zeros((0, len(cols)), dtype=np.float64)
        return np.ascontiguousarray(self._carry[:, [lookup[c] for c in cols]])

    # ------------------------------------------------------------------- api

    def train(self, features, target):
        cols, raw = self._relative(features)
        shaped = self._shape_rows(raw)
        keep = self.MEMORY + self.ORDER + 1
        self._carry = np.ascontiguousarray(shaped[-keep:])
        self._carry_cols = cols

    def predict(self, features):
        cols, raw = self._relative(features)
        block = self._shape_rows(raw)
        n_rows = block.shape[0]
        n_names = block.shape[1]
        warm = self._align_carry(cols)
        history = np.concatenate([warm, block], axis=0) if warm.shape[0] else block
        base = warm.shape[0]

        scores = np.zeros((n_rows, n_names), dtype=np.float64)
        maps = None
        for start in range(0, n_rows, self.REFRESH_EVERY):
            anchor = base + start
            fresh = self._spectral_map(history[max(0, anchor - self.MEMORY):anchor])
            if fresh is not None:
                maps = fresh
            if maps is None:
                continue
            stop = min(start + self.REFRESH_EVERY, n_rows)
            for gap, mat in enumerate(maps, start=1):
                lo = anchor + 1 - gap
                hi = base + stop + 1 - gap
                if lo < 0:
                    continue
                src = history[lo:hi]
                dst = scores[start:stop]
                # Explicit fixed-order accumulation over source names instead of
                # a gemm: BLAS selects different kernels for different row
                # counts, which perturbs the last ULP and breaks bit-exact
                # chunk-invariance between a block and its prefix.
                for j in range(n_names):
                    dst += src[:, j:j + 1] * mat[:, j]

        scores -= scores.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.mean(scores * scores, axis=1, keepdims=True))
        scores /= (norm + self.TINY)
        scores -= scores.mean(axis=1, keepdims=True)
        return pd.DataFrame(scores, index=features.index, columns=cols)
