"""t32 BASELINE: graph leg standalone, EXACT extraction of _GraphLeg from
submission_final.py with the shipped 5-bar `target` label.  Reference only --
used to verify the standalone extraction reproduces research/cache/ticket2_pnl.npy.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class GraphBasePredictor(Predictor):
    """Leg A (unsupervised VAR(1) grid) + leg B (ridge VAR on the 5-bar target)."""

    # ---- leg A: price-space VAR(1) lead-lag graph -----------------------
    WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
    LAM_A = 0.10
    MINP = 250
    # ---- leg B: rank-space own-name-zeroed ridge VAR --------------------
    TAILS = (6000, 25000)
    LAM_B = 0.10
    KLAG = 2      # lags 0 and 1 in the design
    HM = 5        # rows held back at the tail so the design never runs off the end

    # ------------------------------------------------------------- utils
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

    # ------------------------------------------------------------- train
    def train(self, features, target):
        tk = self._tickers(features)
        self.tk_ = tk
        f1 = self._f1(features, tk)
        j = f1.shape[1]
        self.j_ = j

        # --- leg A state: tail of the standardised cross-section --------
        self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()

        # --- leg B state: one ridge coefficient matrix per tail ---------
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
            # design rows: block k is u[K-1-k : T-1-HM-k]; here K = 3 blocks
            # are laid out but only the first KLAG are kept, so the row window
            # is [2 : T-1-HM] and the target is aligned to block 0.
            base = 3 - 1
            x = np.concatenate(
                [u[base - k:t - 1 - self.HM - k] for k in range(self.KLAG)], axis=1)
            yy = y[base:t - 1 - self.HM]
            a = x.T @ x / len(x)
            b = x.T @ yy / len(x)
            ridge = self.LAM_B * float(np.diag(a).mean())
            w = np.linalg.solve(a + ridge * np.eye(kj), b)
            self.WB_.append(np.where(own, 0.0, w))

    # ----------------------------------------------------------- predict
    # rows per chunk in the vectorised _legA; pure memory/perf knob, it does
    # NOT touch the arithmetic (the running sums are carried across chunks).
    CHUNK = 2048

    def _legA(self, xv):
        """Grid-averaged VAR(1) lead-lag propagation. Recurses causally.

        BIT-IDENTICAL vectorisation of the original row-at-a-time loop:

          m_u = ((m_{u-1} + outer(a[p],b[p])) - outer(a[q],b[q]))

        is a strictly sequential chain of float64 additions, so it is
        reproduced with `np.cumsum` over the INTERLEAVED delta stream
        [m_0, +X_1, -Y_1, +X_2, -Y_2, ...]; np.add.accumulate is defined as
        out[i] = out[i-1] + d[i], which is exactly the original association
        (including the intermediate rounding between the add and the
        subtract).  Rows whose window has not opened yet (q < 0) get the
        additive identity -0.0, which is exact for every float64 including
        +/-0.0.  Everything downstream (diag, outer normalisations, inv,
        matmul, argsort) is elementwise or per-slice and was verified to be
        bit-identical in batched form on this BLAS.
        """
        nv, k = xv.shape
        x = np.vstack([self.bufA_, xv])
        t0 = len(self.bufA_)
        a = x[1:]
        b = x[:-1]
        eye = np.eye(k)
        acc = np.zeros((nv, k))
        xv3 = xv[:, :, None]
        ch = self.CHUNK
        for wlen in self.WGRID:
            lo0 = max(0, t0 - wlen)
            m0 = a[lo0:t0].T @ b[lo0:t0]
            s0 = b[lo0:t0].T @ b[lo0:t0]
            v0 = (a[lo0:t0] ** 2).sum(0)

            # exact integer replica of the running float counter `c`
            cnt = np.empty(nv, dtype=np.int64)
            cnt[0] = t0 - lo0
            if nv > 1:
                pa = t0 + np.arange(1, nv, dtype=np.int64) - 1
                cnt[1:] = cnt[0] + np.cumsum(1 - (pa >= wlen).astype(np.int64))
            keep = cnt >= self.MINP

            mc, sc_, vc_ = m0, s0, v0
            for u0 in range(0, nv, ch):
                u1 = min(u0 + ch, nv)
                first = u0 == 0
                ustart = 1 if first else u0
                npair = u1 - ustart
                n = 1 + 2 * npair
                dm = np.empty((n, k, k))
                dsq = np.empty((n, k, k))
                dv = np.empty((n, k))
                dm[0] = mc
                dsq[0] = sc_
                dv[0] = vc_
                if npair:
                    p = t0 + np.arange(ustart, u1) - 1
                    ap = a[p]
                    bp = b[p]
                    dm[1::2] = ap[:, :, None] * bp[:, None, :]
                    dsq[1::2] = bp[:, :, None] * bp[:, None, :]
                    dv[1::2] = ap ** 2
                    # removals: q = p - wlen; q < 0 is a contiguous prefix
                    nb = int(np.searchsorted(p, wlen))
                    if nb:
                        dm[2:2 + 2 * nb:2] = -0.0
                        dsq[2:2 + 2 * nb:2] = -0.0
                        dv[2:2 + 2 * nb:2] = -0.0
                    if nb < npair:
                        q = p[nb:] - wlen
                        aq = a[q]
                        bq = b[q]
                        dm[2 + 2 * nb::2] = -(aq[:, :, None] * bq[:, None, :])
                        dsq[2 + 2 * nb::2] = -(bq[:, :, None] * bq[:, None, :])
                        dv[2 + 2 * nb::2] = -(aq ** 2)
                np.cumsum(dm, axis=0, out=dm)
                np.cumsum(dsq, axis=0, out=dsq)
                np.cumsum(dv, axis=0, out=dv)
                mc = dm[-1].copy()
                sc_ = dsq[-1].copy()
                vc_ = dv[-1].copy()
                off = 0 if first else 2
                mv = dm[off::2]
                sv = dsq[off::2]
                vv = dv[off::2]

                idx = np.flatnonzero(keep[u0:u1])
                if idx.size == 0:
                    continue
                mm = mv[idx]
                ss = sv[idx]
                vg = np.maximum(vv[idx], 1e-12)
                dg = np.maximum(np.diagonal(ss, axis1=1, axis2=2), 1e-12)
                ds = np.sqrt(dg)
                cm = mm / np.sqrt(vg[:, :, None] * dg[:, None, :])
                scm = ss / (ds[:, :, None] * ds[:, None, :])
                bm = cm @ np.linalg.inv(
                    (1.0 - self.LAM_A) * scm + self.LAM_A * eye)
                r = (bm @ xv3[u0:u1][idx])[:, :, 0]
                o = np.argsort(np.argsort(r, axis=1), axis=1).astype(np.float64)
                acc[u0 + idx] += o / (k - 1.0) - 0.5
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

        # equal risk in signal space: each leg to unit cross-sectional
        # dispersion at every timestamp, then summed 1:1. No fitted weight.
        sig = self._unit_row(a) + self._unit_row(b)
        sig -= sig.mean(1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk)
