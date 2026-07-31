"""Competition 5 submission: ticket3 — ticket2 with the fusion dial moved once.

WHAT THIS IS
    ticket3 is submission_ticket2 with EXACTLY ONE scalar changed: the relative
    weight of leg B (the decorrelator) in the final signal-space fusion. Nothing
    else about the construction — the two legs, the graphs, the ridge shrinkages,
    the window grid, the causality contract — is touched.

      LEG A  price space  — grid-averaged VAR(1) lead-lag coefficient matrix on the
                            standardised, de-marketed return cross-section.
      LEG B  rank space   — ridge VAR on cross-sectional RANKS with own-name
                            coefficients zeroed; orthogonal to own-name reversal by
                            construction. It carries the DECORRELATION.

    ticket2 fuses the two legs at equal risk:  sig = unit(A) + unit(B).
    ticket3 fuses them:                        sig = unit(A) + GB * unit(B),
    with GB = 1.5. GB = 1.0 reproduces ticket2 BIT-FOR-BIT (the fidelity gate).

WHY MOVE THE DIAL
    ticket2 scored platform Sharpe 0.0709 (would rank #2-#3) but was NOT admitted
    to the prize-eligible set: its global novelty read 57.48 deg, i.e. correlation
    cos(57.48) = 0.5376 with the nearest better-ranked signal, over the 0.50
    admission line. Admission is a correlation gate, and the correlation lives in
    PnL/return space (global novelty = arccos of a return correlation). Up-weighting
    leg B moves the signal in exactly that space.

    GB = 1.5 is the same direction as a leg-B fusion weight w_B = 0.60
    (unit(A) + 1.5*unit(B)  is proportional to  0.4*unit(A) + 0.6*unit(B)). It lowers
    the local proxy corr-vs-plain-reversal from 0.2825 to 0.1989 — a reduction of
    0.084, inside the 0.06-0.12 band targeted to pull the blocker correlation from
    0.5376 down to roughly 0.43-0.47 with margin under the plausible blocker anchors
    (neutral 0.432, reversal-like 0.379), while the pessimistic anchor (blocker hits
    both legs equally) is the one it does not clear. It is the SMALLEST pre-declared
    grid point that lands in that band, so it gives away the least Sharpe and least
    enlarges the better-ranked set it is measured against.

    This is a single dial at a pre-declared value — no argmax, no new machinery, no
    new fitted parameter. Selection intensity is minimal by design, which is why
    ticket2 (2 legs, equal risk, not an argmax) transferred to the platform at
    ~0.769 of local Sharpe while heavily-searched signals transferred at ~0.42-0.56.
    ticket3 is the same construction with one dial moved and should transfer alike.

MEASURED VALUES (harness runner.py --full --gauge-fix; see the deliverable report)
    Sharpe +0.0842, IC +0.0233, concentration 0.0238, city novelty ~11.4 deg.
    windows : full 0.0842 | h1 0.1064 | h2 0.0574 | rec20 0.0481 | hold 0.0754
    corr vs plain reversal -csrank(Feature.1): 0.1989 (was 0.2825 at GB=1.0).
    EXPECTATION on the platform, using ticket2's own transfer ratio 0.769:
      0.0842 * 0.769 = ~0.065; conservative 0.48 -> ~0.040. Podium threshold ~0.0551.
    The slot is bought for ADMISSION (decorrelation), not for Sharpe. Even a mid-0.05
    platform Sharpe is podium-grade if it clears the 0.50 correlation gate.

PARAMETERS
    One numeric choice beyond ticket2's: GB = 1.5, pre-declared (= w_B 0.60). The two
    ridge shrinkages and window grid are unchanged from ticket2 and remain flat /
    averaged, not selected.

CAUSALITY
    Identical to ticket2. train() sees only training rows; leg A stores a tail of the
    standardised cross-section and recurses forward one validation row at a time; leg
    B stores ridge coefficient matrices fitted on training rows only. No statistic at
    row t uses any row > t; GB is a compile-time constant applied per row. The stateful
    train() contract (bufA_, WB_) is preserved exactly.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class Ticket3Predictor(Predictor):
    # ---- the ONE moved dial --------------------------------------------------
    GB = 1.5      # leg-B gain in the fusion; GB == 1.0 reproduces ticket2 exactly
    # ---- leg A: price-space VAR(1) lead-lag graph ----------------------------
    WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
    LAM_A = 0.10
    MINP = 250
    # ---- leg B: rank-space own-name-zeroed ridge VAR -------------------------
    TAILS = (6000, 25000)
    LAM_B = 0.10
    KLAG = 2      # lags 0 and 1 in the design
    HM = 5        # rows held back at the tail so the design never runs off the end

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _f1(features, tk):
        return np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _zrow(a):
        """Cross-sectional z-score of each row."""
        m = a.mean(1, keepdims=True)
        s = a.std(1, keepdims=True)
        return (a - m) / (s + 1e-9)

    @staticmethod
    def _rank_u(a):
        """Cross-sectional rank of each row mapped to [-1, +1]."""
        j = a.shape[1]
        r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _csrank_c(a):
        """Cross-sectional rank of each row mapped to [-0.5, +0.5]."""
        j = a.shape[1]
        r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
        return r / (j - 1.0) - 0.5

    @staticmethod
    def _design(u, k):
        """[u_t, u_{t-1}, ...] with NaN where the lag does not exist."""
        t, j = u.shape
        x = np.full((t, k * j), np.nan)
        x[:, :j] = u
        for i in range(1, k):
            x[i:, i * j:(i + 1) * j] = u[:-i]
        return x

    @staticmethod
    def _tozc(s):
        """Row-demean, row-standardise, clip; non-finite rows -> 0."""
        bad = ~np.isfinite(s)
        x = np.where(bad, 0.0, s)
        x = x - x.mean(1, keepdims=True)
        x = x / np.maximum(x.std(1, keepdims=True), 1e-12)
        x[bad.any(1)] = 0.0
        return np.clip(x, -3.0, 3.0)

    @staticmethod
    def _unit_row(a):
        """Scale each row to unit cross-sectional dispersion (causal, row-local)."""
        s = a.std(1, keepdims=True)
        return a / np.maximum(s, 1e-12)

    # ------------------------------------------------------------------ train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        f1 = self._f1(features, tk)
        j = f1.shape[1]
        self.j_ = j

        # --- leg A state: tail of the standardised cross-section -------------
        self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()

        # --- leg B state: one ridge coefficient matrix per tail --------------
        tgt = np.nan_to_num(target[tk].to_numpy(dtype=np.float64))
        ntr = len(f1)
        kj = self.KLAG * j
        own = np.zeros((kj, j), dtype=bool)
        for k in range(self.KLAG):
            own[k * j:(k + 1) * j] = np.eye(j, dtype=bool)

        self.WB_ = []
        for tail in self.TAILS:
            lo = max(0, ntr - tail)
            u = self._rank_u(f1[lo:])
            y = tgt[lo:]
            y = y - y.mean(1, keepdims=True)
            y = y / np.maximum(y.std(1, keepdims=True), 1e-12)
            t = len(u)
            if t < kj + self.HM + 8:
                self.WB_.append(np.zeros((kj, j)))
                continue
            # design rows: block k is u[K-1-k : T-1-HM-k]; here K = 3 blocks are
            # laid out but only the first KLAG are kept, so the row window is
            # [2 : T-1-HM] and the target is aligned to block 0.
            base = 3 - 1
            x = np.concatenate(
                [u[base - k:t - 1 - self.HM - k] for k in range(self.KLAG)], axis=1)
            yy = y[base:t - 1 - self.HM]
            a = x.T @ x / len(x)
            b = x.T @ yy / len(x)
            ridge = self.LAM_B * float(np.diag(a).mean())
            w = np.linalg.solve(a + ridge * np.eye(kj), b)
            self.WB_.append(np.where(own, 0.0, w))

    # ---------------------------------------------------------------- predict
    def _legA(self, xv):
        """Grid-averaged VAR(1) lead-lag propagation. Recurses causally."""
        nv, k = xv.shape
        x = np.vstack([self.bufA_, xv])
        t0 = len(self.bufA_)
        a = x[1:]
        b = x[:-1]
        eye = np.eye(k)
        acc = np.zeros((nv, k))
        for wlen in self.WGRID:
            lo0 = max(0, t0 - wlen)
            m = a[lo0:t0].T @ b[lo0:t0]
            s = b[lo0:t0].T @ b[lo0:t0]
            vc = (a[lo0:t0] ** 2).sum(0)
            c = float(t0 - lo0)
            for u in range(nv):
                if u > 0:
                    p = t0 + u - 1
                    m += np.outer(a[p], b[p])
                    s += np.outer(b[p], b[p])
                    vc += a[p] ** 2
                    c += 1
                    q = p - wlen
                    if q >= 0:
                        m -= np.outer(a[q], b[q])
                        s -= np.outer(b[q], b[q])
                        vc -= a[q] ** 2
                        c -= 1
                if c < self.MINP:
                    continue
                ds = np.sqrt(np.maximum(np.diag(s), 1e-12))
                cm = m / np.sqrt(np.outer(np.maximum(vc, 1e-12),
                                          np.maximum(np.diag(s), 1e-12)))
                sc = s / np.outer(ds, ds)
                bm = cm @ np.linalg.inv((1.0 - self.LAM_A) * sc + self.LAM_A * eye)
                r = bm @ xv[u]
                o = np.argsort(np.argsort(r)).astype(np.float64)
                acc[u] += o / (k - 1.0) - 0.5
        return acc

    def _legB(self, f1v):
        """Own-name-zeroed rank-space VAR, averaged over the two tails."""
        u = self._rank_u(f1v)
        x = self._design(u, self.KLAG)
        xn = np.nan_to_num(x)
        badrow = np.isnan(x).any(1)
        out = np.zeros((len(u), self.j_))
        for w in self.WB_:
            s = xn @ w
            s[badrow] = np.nan
            out += self._tozc(s)
        return out / len(self.WB_)

    def predict(self, features):
        tk = self._tickers(features)
        f1 = self._f1(features, tk)
        xv = self._zrow(f1)

        a = self._legA(xv)
        b = self._legB(f1)

        # ticket2 fuses at equal risk: unit(a) + unit(b). ticket3 up-weights the
        # decorrelator leg B by the single pre-declared gain GB (= 1.5, i.e. leg-B
        # fusion weight w_B 0.60). GB == 1.0 reproduces ticket2 bit-for-bit.
        sig = self._unit_row(a) + self.GB * self._unit_row(b)
        sig -= sig.mean(1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
