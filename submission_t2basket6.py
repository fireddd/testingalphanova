"""Competition 5 submission: "t2 + graph_basket-6" — the clean admissible ticket2.

WHAT THIS IS
    ticket2 (the transfer carrier, 0.769) blended at FROZEN EQUAL RISK with
    graph_basket-6: an equal-risk mix of the SIX strongest INDEPENDENT graph
    primitives (levy, centrality, second-order, partial-correlation, corr-cluster,
    flow-imbalance).  The basket is a pure GRAPH-DECORRELATION basis: it injects its
    own orthogonal Sharpe (~0.033) so it decorrelates ticket2 off the crowded axis
    WITHOUT diluting with noise.  Result: the safe realization of the ticket2 thesis
    — full ~0.089, corr vs apex6 0.34 (vs ticket8's razor-thin 0.49), no fitted weights.

      L_t2     ticket2's full signal (price-space grid VAR(1) lead-lag leg + rank-space
               own-name-zeroed ridge VAR leg, equal risk).  Inlined VERBATIM as the
               non-inheriting nested helper `_Ticket2` (byte-for-byte submission_ticket8
               `_Ticket2`).  The Sharpe + transfer carrier.
      L_basket sum_k BW_k * sig_k over the six primitives, where sig_k =
               demean(csrank(demean(primitive_k(F1_val)))) computed COLD-START on the
               validation Feature.1 block (window strictly < t; the first W bars are
               zero, exactly as the research basket was built), and BW_k = sign_k / vol_k
               are FROZEN per-primitive equal-risk weights (sel75 PnL orientation / vol).

BLEND — FROZEN EQUAL RISK, NO FITTED WEIGHT, NO RUNTIME SCALE MEASUREMENT
    sig = L_t2 / SCALE_T2 + L_basket / SCALE_BASKET, then per-row cross-sectional demean.
    SCALE_T2 and SCALE_BASKET are FROZEN CLASS CONSTANTS (per-leg sel75 PnL vols,
    measured once offline on periods 001-075).  train() prepares ONLY the ticket2
    sleeve's warm-up state (leg-A tail buffer + leg-B ridge matrices); it measures no
    scale.  The six graph primitives are stateless / cold-started and need no train().

CAUSALITY
    Every primitive signal at bar t uses only validation rows < t (trailing window
    [t-W, t-1]); the first W bars are zero.  L_t2's two legs recurse causally.  csrank
    and demean are per-timestamp cross-sectional ops.  No backward fill, no whole-block
    statistic.  (Verified: element-wise probe before-t == 0 exactly.)

HARDCODED NUMERIC LITERALS (all labelled):
    frozen-scale : SCALE_T2, SCALE_BASKET, and BW (six sign/vol basket weights).
    structural   : W=60, TOPK=5 (graph window / sparsity); LOOKS/TOPK/MOM_W (levy);
                   ticket2's WGRID, LAM_A, MINP, TAILS, LAM_B, KLAG, HM; rank-centring
                   0.5 constants; 1e-3/1e-9/1e-12 numerical epsilons.
    fitted       : NONE.  No fitted blend weight, no runtime-measured scale.
"""

import numpy as np
import pandas as pd

from predictor import Predictor


class T2Basket6Predictor(Predictor):
    """ticket2 carrier + graph_basket-6 decorrelator, frozen equal-risk blend."""

    # ---- FROZEN equal-risk scales (per-leg sel75 PnL vol; NOT re-measured) ----
    SCALE_T2 = 0.04933074223312276      # frozen-scale: ticket2 full-signal PnL vol
    SCALE_BASKET = 2.893643802859039    # frozen-scale: basket leg (sum BW_k*sig_k) PnL vol

    # ---- FROZEN basket-internal equal-risk weights: sign_k / sel75-PnL-vol_k -----
    BW = {
        "levy":            52.38434994132883,
        "centrality_rev":  41.39598831657011,
        "secondorder":    -48.32167643097667,
        "partial_rev":     54.71724552268681,
        "corr_rev":        47.957177386804155,
        "flow_imb":        60.81299994309143,
    }

    # ---- graph-primitive structural constants (pre-declared, no argmax) --------
    W = 60            # trailing correlation/lead-lag window
    TOPK = 5          # sparsity: top-K neighbours per node
    LOOKS = (22, 44, 66, 88, 110, 132)   # levy lookback set (averaged over)
    MOM_W = 22        # levy momentum window

    # ====================================================================
    #  ticket2 sleeve — VERBATIM from submission_ticket8._Ticket2
    #  (non-inheriting helper; identical to submission_ticket2 logic).
    # ====================================================================
    class _Ticket2:
        """price-space VAR(1) lead-lag leg + rank-space own-name-zeroed ridge VAR
        leg, each normalised to unit cross-sectional dispersion per row, summed 1:1."""

        WGRID = (750, 1500, 2000, 2500, 3500, 5000, 7500)
        LAM_A = 0.10
        MINP = 250
        TAILS = (6000, 25000)
        LAM_B = 0.10
        KLAG = 2
        HM = 5

        @staticmethod
        def _tickers(features):
            cols = [c for c in features.columns if c[0] == "Feature.1"]
            return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

        @staticmethod
        def _f1(features, tk):
            return np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))

        @staticmethod
        def _zrow(a):
            m = a.mean(1, keepdims=True)
            s = a.std(1, keepdims=True)
            return (a - m) / (s + 1e-9)

        @staticmethod
        def _rank_u(a):
            j = a.shape[1]
            r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
            return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

        @staticmethod
        def _design(u, k):
            t, j = u.shape
            x = np.full((t, k * j), np.nan)
            x[:, :j] = u
            for i in range(1, k):
                x[i:, i * j:(i + 1) * j] = u[:-i]
            return x

        @staticmethod
        def _tozc(s):
            bad = ~np.isfinite(s)
            x = np.where(bad, 0.0, s)
            x = x - x.mean(1, keepdims=True)
            x = x / np.maximum(x.std(1, keepdims=True), 1e-12)
            x[bad.any(1)] = 0.0
            return np.clip(x, -3.0, 3.0)

        @staticmethod
        def _unit_row(a):
            s = a.std(1, keepdims=True)
            return a / np.maximum(s, 1e-12)

        def train(self, features, target):
            tk = self._tickers(features)
            self.tk_ = tk
            f1 = self._f1(features, tk)
            j = f1.shape[1]
            self.j_ = j
            self.bufA_ = self._zrow(f1)[-(max(self.WGRID) + 2):].copy()

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
                base = 3 - 1
                x = np.concatenate(
                    [u[base - k:t - 1 - self.HM - k] for k in range(self.KLAG)], axis=1)
                yy = y[base:t - 1 - self.HM]
                a = x.T @ x / len(x)
                b = x.T @ yy / len(x)
                ridge = self.LAM_B * float(np.diag(a).mean())
                w = np.linalg.solve(a + ridge * np.eye(kj), b)
                self.WB_.append(np.where(own, 0.0, w))

        def _legA(self, xv):
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
            sig = self._unit_row(a) + self._unit_row(b)
            sig -= sig.mean(1, keepdims=True)
            return pd.DataFrame(sig, index=features.index, columns=tk)

    # ====================================================================
    #  graph_basket-6 primitives — cold-start on the validation F1 block.
    #  Verbatim mechanisms from research/agents/t10/graph_gen.py + the levy
    #  primitive (research/agents/t9_levy.npy), which the packaged basket
    #  reproduces at PnL corr 1.0000 vs the cached graph_basket.npy.
    # ====================================================================
    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    @staticmethod
    def _f1(features, tk):
        return np.nan_to_num(features["Feature.1"][tk].to_numpy(dtype=np.float64))

    @staticmethod
    def _csrank(a, j):
        """graph_gen csrank: cross-sectional average rank -> [-1, +1]."""
        r = pd.DataFrame(np.asarray(a)).rank(axis=1, method="average").to_numpy()
        return (r - 0.5 * (j + 1)) / (0.5 * (j - 1))

    @staticmethod
    def _dm(a):
        a = np.nan_to_num(np.asarray(a, float))
        return a - a.mean(1, keepdims=True)

    @staticmethod
    def _zc(win):
        return (win - win.mean(0)) / (win.std(0) + 1e-9)

    @classmethod
    def _topk_row(cls, A, k):
        M = np.zeros_like(A)
        for i in range(A.shape[0]):
            idx = np.argsort(-np.abs(A[i]))[:k]
            M[i, idx] = A[i, idx]
        return M / (np.abs(M).sum(1, keepdims=True) + 1e-12)

    # ---- the six primitives (each returns raw (T, J); cold-start) -------------
    @classmethod
    def _corr_rev(cls, Rv):
        T, J = Rv.shape; s = np.zeros((T, J)); W, K = cls.W, cls.TOPK
        for t in range(W, T):
            C = np.nan_to_num(np.corrcoef(cls._zc(Rv[t - W:t]).T)); np.fill_diagonal(C, 0)
            A = cls._topk_row(np.abs(C), K); s[t] = -(A @ Rv[t - 1])
        return s

    @classmethod
    def _partial_rev(cls, Rv):
        T, J = Rv.shape; s = np.zeros((T, J)); W, K = cls.W, cls.TOPK
        for t in range(W, T):
            Cv = np.cov(cls._zc(Rv[t - W:t]).T) + 1e-3 * np.eye(J); P = np.linalg.inv(Cv)
            d = np.sqrt(np.diag(P)); Pc = -P / np.outer(d, d); np.fill_diagonal(Pc, 0)
            A = cls._topk_row(Pc, K); s[t] = -(A @ Rv[t - 1])
        return s

    @classmethod
    def _centrality_rev(cls, Rv):
        T, J = Rv.shape; s = np.zeros((T, J)); W = cls.W
        for t in range(W, T):
            C = np.abs(np.nan_to_num(np.corrcoef(cls._zc(Rv[t - W:t]).T))); np.fill_diagonal(C, 0)
            ev, V = np.linalg.eigh(C); cent = np.abs(V[:, -1]); s[t] = -cent * Rv[t - 1]
        return s

    @classmethod
    def _flow_imb(cls, Rv):
        T, J = Rv.shape; s = np.zeros((T, J)); W = cls.W
        for t in range(W, T):
            a = cls._zc(Rv[t - W + 1:t]); b = cls._zc(Rv[t - W:t - 1]); L = (b.T @ a) / len(a)
            out_deg = np.abs(L).sum(1); in_deg = np.abs(L).sum(0)
            s[t] = (out_deg - in_deg) * np.sign(Rv[t - 1])
        return s

    @classmethod
    def _secondorder(cls, Rv):
        T, J = Rv.shape; s = np.zeros((T, J)); W, K = cls.W, cls.TOPK
        for t in range(W, T):
            C = np.abs(np.nan_to_num(np.corrcoef(cls._zc(Rv[t - W:t]).T))); np.fill_diagonal(C, 0)
            A = cls._topk_row(C, K); A2 = A @ A; np.fill_diagonal(A2, 0); s[t] = -(A2 @ Rv[t - 1])
        return s

    @classmethod
    def _levy(cls, Rv):
        T, J = Rv.shape; out = np.zeros((T, J)); Wmax = max(cls.LOOKS)
        for t in range(Wmax, T):
            L = np.zeros((J, J))
            for Wl in cls.LOOKS:
                seg = Rv[t - Wl:t]; P = np.cumsum(seg, 0); P = P - P.mean(0)
                L += 0.5 * (P.T @ seg - seg.T @ P)
            A = (L / len(cls.LOOKS)).T; M = np.zeros_like(A)
            for i in range(J):
                idx = np.argsort(-np.abs(A[i]))[:cls.TOPK]; M[i, idx] = A[i, idx]
            M = M / (np.abs(M).sum(1, keepdims=True) + 1e-12)
            out[t] = M @ Rv[t - cls.MOM_W:t].sum(0)
        return out

    # ------------------------------------------------------------------ API
    def __init__(self):
        self._t2 = self._Ticket2()
        self._prims = {
            "levy": self._levy, "centrality_rev": self._centrality_rev,
            "secondorder": self._secondorder, "partial_rev": self._partial_rev,
            "corr_rev": self._corr_rev, "flow_imb": self._flow_imb,
        }

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # WARM-UP STATE ONLY: the ticket2 sleeve trains (bufA_, WB_). The six graph
        # primitives are cold-started on the val block and need no training state.
        self._t2.train(features, target)
        self.tk_ = self._tickers(features)
        self.j_ = len(self.tk_)
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        j = len(tk)
        Rv = self._f1(features, tk)                 # validation Feature.1 block

        # L_basket = sum_k BW_k * demean(csrank(demean(primitive_k)))  (cold-start)
        L_basket = np.zeros((len(Rv), j))
        for name, fn in self._prims.items():
            sig_k = self._dm(self._csrank(self._dm(fn(Rv)), j))
            L_basket += self.BW[name] * sig_k

        # L_t2 carrier
        pt2 = self._t2.predict(features)[tk]
        L_t2 = pt2.to_numpy(dtype=np.float64)

        # FROZEN equal-risk blend, then per-row cross-sectional demean.
        sig = L_t2 / self.SCALE_T2 + L_basket / self.SCALE_BASKET
        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk).fillna(0.0)
