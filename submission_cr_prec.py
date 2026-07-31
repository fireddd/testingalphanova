"""Clean-room arm "precision": sparse partial-correlation graph propagation.

Economic content
----------------
A name's next-bar *relative* return is predictable from the recent relative returns of the
other names through a time-varying linear map. Here that map is derived from a **sparse
precision matrix**: the trailing exponentially-weighted second moment of the vol-scaled,
row-centred returns is inverted to a precision matrix, converted to **partial correlations**
(the association between two names after conditioning on all eighteen others), and the partial
correlations are **soft-thresholded** — an L1 / graphical-lasso surrogate that sets weak
conditional edges exactly to zero. The surviving edges define a node-wise conditional
expectation for each name; the traded score is minus that name's conditional residual, i.e.
the part of its move that its conditional neighbours did not account for.

Everything is strictly causal: every trailing statistic used at row t is an exponentially
weighted accumulation over rows <= t only, carried forward by an IIR recursion, so a prefix of
a block produces bit-identical scores on that prefix.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from predictor import Predictor


class PartialGraphPredictor(Predictor):
    """Sparse partial-correlation (precision) propagation over the 20-name panel."""

    DRIVER = "Feature.1"          # the contemporaneous return column group
    VOL_HALFLIFE = 128.0          # trailing scale, applied with a one-bar lag
    SCORE_CAP = 4.0               # winsorisation of the vol-scaled move
    GRAPH_HALFLIVES = (256.0, 1024.0, 4096.0)
    RIDGE = 0.05                  # correlation-matrix ridge, invertibility only
    L1 = 0.05                     # soft-threshold on partial correlations
    WARMUP_ROWS = 8000            # trailing training rows carried into predict()
    BLOCK = 8192                  # internal chunking; boundaries fixed from block start
    SEED = 20260731

    def __init__(self) -> None:
        np.random.seed(self.SEED)
        self._warmup = None
        self._panel = None

    # ------------------------------------------------------------------ train
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        np.random.seed(self.SEED)
        driver = features[self.DRIVER]
        panel = sorted(driver.columns.tolist(), key=lambda c: int(str(c).split(".")[-1]))
        tail = driver[panel].to_numpy(dtype=np.float64)
        if len(tail) > self.WARMUP_ROWS:
            tail = tail[-self.WARMUP_ROWS:]
        self._panel = panel
        self._warmup = np.ascontiguousarray(tail)

    # ---------------------------------------------------------------- predict
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        driver = features[self.DRIVER]
        panel = sorted(driver.columns.tolist(), key=lambda c: int(str(c).split(".")[-1]))
        live = driver[panel].to_numpy(dtype=np.float64)
        n_out, width = live.shape

        if self._warmup is not None and self._panel == panel and len(self._warmup):
            lead = self._warmup
        else:
            lead = np.zeros((0, width), dtype=np.float64)
        n_lead = lead.shape[0]
        stream = np.concatenate((lead, live), axis=0) if n_lead else live

        def decay(halflife):
            return 0.5 ** (1.0 / halflife)

        def ew_state(width_):
            return np.zeros((1, width_), dtype=np.float64)

        def ew_step(values, alpha, state, first_index):
            """One chunk of a bias-corrected exponentially weighted mean along axis 0."""
            filtered, carry = lfilter(
                [1.0 - alpha], [1.0, -alpha], values, axis=0, zi=state
            )
            steps = np.arange(
                first_index, first_index + values.shape[0], dtype=np.float64
            )
            mass = 1.0 - alpha ** (steps + 1.0)
            return filtered / mass.reshape(-1, 1), carry

        def centre(block):
            return block - block.mean(axis=1, keepdims=True)

        # --- pass 1: vol scaling, one bar lagged, over the whole stream -----
        centred = centre(stream)
        vol_alpha = decay(self.VOL_HALFLIFE)
        power, _ = ew_step(centred * centred, vol_alpha, ew_state(width), 0.0)
        lagged = np.empty_like(power)
        lagged[0] = power[0]
        lagged[1:] = power[:-1]
        scaled = np.clip(
            centred / np.sqrt(lagged + 1e-300), -self.SCORE_CAP, self.SCORE_CAP
        )

        # --- pass 2: chunked precision -> partial correlation -> propagation -
        alphas = [decay(h) for h in self.GRAPH_HALFLIVES]
        states = [ew_state(width * width) for _ in alphas]
        n_graphs = len(alphas)
        identity = np.eye(width)
        diag = np.arange(width)
        scores = np.empty((n_out, width), dtype=np.float64)

        edges = [(0, n_lead)] if n_lead else []
        cursor = n_lead
        while cursor < n_lead + n_out:
            edges.append((cursor, min(cursor + self.BLOCK, n_lead + n_out)))
            cursor += self.BLOCK

        for lo, hi in edges:
            piece = scaled[lo:hi]
            outer = (piece[:, :, None] * piece[:, None, :]).reshape(hi - lo, width * width)
            emit = lo >= n_lead
            acc = np.zeros((hi - lo, width), dtype=np.float64) if emit else None
            for g in range(n_graphs):
                moment, states[g] = ew_step(outer, alphas[g], states[g], float(lo))
                if not emit:
                    continue
                gram = moment.reshape(-1, width, width)
                root = np.sqrt(gram[:, diag, diag])
                corr = gram / (root[:, :, None] * root[:, None, :])
                precision = np.linalg.inv(corr + self.RIDGE * identity)
                spread = np.sqrt(precision[:, diag, diag])
                partial = -precision / (spread[:, :, None] * spread[:, None, :])
                partial[:, diag, diag] = 0.0
                kept = np.sign(partial) * np.maximum(np.abs(partial) - self.L1, 0.0)
                weights = kept * (spread[:, None, :] / spread[:, :, None])
                # explicit multiply-reduce, NOT einsum/matmul: BLAS dispatch is
                # batch-size dependent and breaks bit-exact chunk invariance.
                acc += (weights * piece[:, None, :]).sum(axis=2) - piece
            if emit:
                scores[lo - n_lead:hi - n_lead] = acc / n_graphs

        scores = centre(scores)
        scores[~np.isfinite(scores)] = 0.0
        scores = centre(scores)
        return pd.DataFrame(scores, index=features.index, columns=panel)
