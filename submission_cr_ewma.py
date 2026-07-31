"""Clean-room arm "ewma": recursive exponentially-weighted lead-lag map.

Economic content
----------------
A name's next-bar relative return is partly explained by the current relative
returns of the rest of the panel through a slowly-drifting linear map.  The map
is estimated by a *recursive* exponentially-weighted regression: two cross-moment
accumulators per decay rate are advanced one bar at a time, and the map is the
ridge solution built from them.  There is no trailing window anywhere, so there
are no window-exit discontinuities; an old observation fades instead of dropping
out.

Three pre-declared decay rates (half-lives 120 / 600 / 3000 bars) are blended at
equal weight after each leg is put on equal risk by its own recursive
exponentially-weighted scale.

Everything is causal by construction: accumulator state at row t is a function of
rows <= t only, the state produced by train() is copied (never mutated) before
predict() runs, and the chunking used to bound memory happens at fixed offsets
from the start of the block, so predicting a prefix is bit-identical to
predicting the first rows of the whole block.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from predictor import Predictor


class RecursiveMomentPredictor(Predictor):
    """EW-recursive lag-1 cross-sectional map over three decay rates."""

    # ---- pre-declared constants; no grid, no sweep --------------------------
    DECAY_HALFLIVES = (120.0, 600.0, 3000.0)
    TAIL_CLIP = 4.0
    STREAM_CHUNK = 8192
    TINY = 1e-12
    RETURN_BLOCK = "Feature.1"
    SEED = 20260731

    def __init__(self):
        np.random.seed(self.SEED)
        self._decays = tuple(
            float(0.5 ** (1.0 / hl)) for hl in self.DECAY_HALFLIVES
        )
        self._carry = None
        self._width = None

    # ------------------------------------------------------------------ input
    def _pull_relatives(self, frame):
        """Contemporaneous relative-return panel as a dense (T, J) array.

        De-meaned Feature.1 is the contemporaneous return; the de-mean and the
        row scaling below are per-row (axis=1) operations, so no information
        crosses row boundaries.
        """
        block = frame[self.RETURN_BLOCK]
        names = list(block.columns)
        raw = np.ascontiguousarray(block.to_numpy(dtype=np.float64))
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        raw -= raw.mean(axis=1, keepdims=True)
        rms = np.sqrt((raw * raw).mean(axis=1, keepdims=True))
        raw /= np.maximum(rms, self.TINY)
        np.clip(raw, -self.TAIL_CLIP, self.TAIL_CLIP, out=raw)
        raw -= raw.mean(axis=1, keepdims=True)
        return raw, names

    # ------------------------------------------------------------------ state
    def _fresh_state(self, width):
        """Zeroed accumulators: one bundle per decay plus the shared last row."""
        cell = width * width
        bundle = []
        for _ in self._decays:
            bundle.append(
                {
                    "cross": np.zeros(cell, dtype=np.float64),
                    "gram": np.zeros(cell, dtype=np.float64),
                    "mass": 0.0,
                    "power": 0.0,
                }
            )
        return {"legs": bundle, "last": None}

    def _clone_state(self, state):
        return {
            "legs": [
                {
                    "cross": leg["cross"].copy(),
                    "gram": leg["gram"].copy(),
                    "mass": float(leg["mass"]),
                    "power": float(leg["power"]),
                }
                for leg in state["legs"]
            ],
            "last": None if state["last"] is None else state["last"].copy(),
        }

    # --------------------------------------------------------------- streaming
    def _roll_forward(self, panel, state, emit):
        """Advance the recursive accumulators over `panel`, optionally scoring.

        Returns (scores or None, state).  `state` is advanced in place and must
        therefore be a private copy whenever the caller cares about reuse.
        """
        n_rows, width = panel.shape
        cell = width * width
        eye = np.eye(width, dtype=np.float64)
        out = np.zeros((n_rows, width), dtype=np.float64) if emit else None

        # ESS of an EW average with decay d is (1+d)/(1-d); shrink by J/ESS.
        ridges = [
            width * (1.0 - d) / (1.0 + d) for d in self._decays
        ]

        def run_iir(decay, stream, seed_value):
            """One-pole causal filter y = (1-d) x + d y_prev, carrying y_prev."""
            taps_b = np.array([1.0 - decay], dtype=np.float64)
            taps_a = np.array([1.0, -decay], dtype=np.float64)
            zi = np.full((1,) + stream.shape[1:], 0.0, dtype=np.float64)
            zi[0] = decay * seed_value
            filtered, zf = lfilter(taps_b, taps_a, stream, axis=0, zi=zi)
            return filtered, zf[0] / decay

        start = 0
        while start < n_rows:
            stop = min(start + self.STREAM_CHUNK, n_rows)
            here = panel[start:stop]
            m_rows = stop - start

            shifted = np.empty_like(here)
            if state["last"] is None:
                shifted[0] = 0.0
                gate = np.ones((m_rows, 1), dtype=np.float64)
                gate[0] = 0.0
            else:
                shifted[0] = state["last"]
                gate = np.ones((m_rows, 1), dtype=np.float64)
            shifted[1:] = here[:-1]

            pair = (here[:, :, None] * shifted[:, None, :]).reshape(m_rows, cell)
            self_prod = (
                shifted[:, :, None] * shifted[:, None, :]
            ).reshape(m_rows, cell)

            for slot, leg in enumerate(state["legs"]):
                decay = self._decays[slot]
                cross_seq, leg["cross"] = run_iir(decay, pair, leg["cross"])
                gram_seq, leg["gram"] = run_iir(decay, self_prod, leg["gram"])
                mass_seq, mass_end = run_iir(
                    decay, gate, np.array([leg["mass"]], dtype=np.float64)
                )
                leg["mass"] = float(mass_end[0])

                # The leg is scored even when `emit` is False: its recursive
                # risk scale depends on its own past output, so the score
                # stream must advance during train() as well.
                gram_mat = gram_seq.reshape(m_rows, width, width)
                cross_mat = cross_seq.reshape(m_rows, width, width)
                mass = np.maximum(mass_seq, self.TINY)
                scale = (
                    np.trace(gram_mat, axis1=1, axis2=2)[:, None]
                    / (width * mass[:, 0][:, None])
                )
                scale = np.maximum(scale, self.TINY)
                damped = gram_mat + (ridges[slot] * scale)[:, :, None] * eye

                lifted = np.linalg.solve(damped, here[:, :, None])
                raw_leg = np.matmul(cross_mat, lifted)[:, :, 0]

                energy = (raw_leg * raw_leg).mean(axis=1)[:, None]
                power_seq, power_end = run_iir(
                    decay, energy, np.array([leg["power"]], dtype=np.float64)
                )
                leg["power"] = float(power_end[0])

                if emit:
                    unit = raw_leg / np.sqrt(
                        np.maximum(power_seq, self.TINY)
                    )
                    out[start:stop] += (mass * unit) / len(self._decays)

            state["last"] = here[-1].copy()
            start = stop

        if emit:
            out -= out.mean(axis=1, keepdims=True)
        return out, state

    # ------------------------------------------------------------------- API
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        panel, _ = self._pull_relatives(features)
        self._width = panel.shape[1]
        _, self._carry = self._roll_forward(
            panel, self._fresh_state(self._width), emit=False
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        panel, names = self._pull_relatives(features)
        width = panel.shape[1]
        if self._carry is None or self._width != width:
            base = self._fresh_state(width)
        else:
            base = self._clone_state(self._carry)
        scores, _ = self._roll_forward(panel, base, emit=True)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores -= scores.mean(axis=1, keepdims=True)
        return pd.DataFrame(scores, index=features.index, columns=names)
