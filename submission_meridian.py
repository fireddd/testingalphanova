import numpy as np
import pandas as pd
import lightgbm as lgb

from predictor import Predictor


class MeridianPredictor(Predictor):

    SCALE_GRAPH = 0.04933074223312276
    SCALE_APEX = 0.024778769297110750

    class _GraphLeg:

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
        def _csrank_c(a):
            j = a.shape[1]
            r = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
            return r / (j - 1.0) - 0.5

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

        CHUNK = 2048

        def _legA(self, xv):
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

    class _Apex7:

        W_A = 0.60
        W_G = 0.20
        W_T = 0.20
        SCALE_A = 0.34507836
        SCALE_G = 0.86575845
        SCALE_T = 0.60800000
        COMBINED_BETANEUT = True
        COV_WIN = 250
        COV_MINP = 60
        LAM = 0.50

        class _Apex4:

            W = {"f4_minus_f1": 0.3229, "mix_llf6_lagrev": 0.2656,
                 "ps_exp_top3_10": 0.1556, "x3r": 0.1493, "ps_exp_full40": 0.1065}
            REV_NEUT = 0.25
            BETANEUT = True
            TAIL = 600
            PRIOR_RECENT = None
            PRIOR_CAP = 500
            WARM = 30
            EPS = 1e-9

            def __init__(self):
                self._tail = None
                self._prior = None

            def _tickers(self, features):
                return sorted(features.columns.get_level_values(1).unique(),
                              key=lambda c: int(c.split(".")[-1]))

            def _rowdm(self, a):
                return a - np.nanmean(a, axis=1, keepdims=True)

            def _csrank_np(self, a, J):
                r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
                if J < 2:
                    return r * 0.0
                return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

            def _csrank_df(self, df, J):
                r = df.rank(axis=1, method="average")
                if J < 2:
                    return r * 0.0
                return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

            def _ewm_np(self, a, span):
                alpha = 2.0 / (span + 1.0)
                out = np.empty_like(a)
                acc = np.zeros(a.shape[1])
                wsum = 0.0
                for t in range(len(a)):
                    acc = (1 - alpha) * acc + a[t]
                    wsum = (1 - alpha) * wsum + 1.0
                    out[t] = acc / wsum
                return out

            def _tsrank(self, a, d, minp=None):
                if minp is None:
                    minp = max(5, d // 3)
                r = pd.DataFrame(a).rolling(d, min_periods=minp).rank(pct=True).values
                return np.nan_to_num(2.0 * r - 1.0)

            def _decay_linear(self, a, d):
                w = np.arange(d, 0, -1, dtype=float)
                w /= w.sum()
                out = np.zeros_like(a)
                for k in range(d):
                    if k == 0:
                        out += w[k] * a
                    else:
                        out[k:] += w[k] * a[:-k]
                return out

            def _lag_np(self, a, k):
                out = np.zeros_like(a)
                out[k:] = a[:-k]
                return out

            def _cs_resid2(self, y, x1, x2):
                ym, x1m, x2m = self._rowdm(y), self._rowdm(x1), self._rowdm(x2)
                a11 = (x1m * x1m).sum(1); a12 = (x1m * x2m).sum(1); a22 = (x2m * x2m).sum(1)
                b1 = (x1m * ym).sum(1); b2 = (x2m * ym).sum(1)
                det = a11 * a22 - a12 * a12 + 1e-18
                be1 = (a22 * b1 - a12 * b2) / det
                be2 = (a11 * b2 - a12 * b1) / det
                return ym - x1m * be1[:, None] - x2m * be2[:, None]

            def _exp_std_prior(self, a, n0, s10, s20, min_n=10):
                T = a.shape[0]
                s1 = np.cumsum(a, axis=0) + s10
                s2 = np.cumsum(a * a, axis=0) + s20
                n = (np.arange(T) + 1.0)[:, None] + n0
                v = s2 / n - (s1 / n) ** 2
                sd = np.sqrt(np.maximum(v, 1e-18))
                if n0 == 0 and T > min_n:
                    sd[:min_n] = sd[min_n]
                return sd

            def _extract(self, features, tk):
                f = {}
                for i in (1, 2, 3, 4, 6):
                    f[i] = np.nan_to_num(
                        features[f"Feature.{i}"][tk].values.astype(np.float64))
                return f

            def _stats_pass(self, f):
                f1, f6 = f[1], f[6]
                T, J = f1.shape
                st = {}
                f1dm = self._rowdm(f1)
                sd = self._exp_std_prior(f1dm, 0.0, 0.0, 0.0)
                u_ll = np.clip(f1dm / (sd + 1e-12), -3, 3)
                C1 = np.zeros((J, J))
                for t in range(1, T):
                    C1 += np.outer(u_ll[t - 1], u_ll[t])
                st["ll_C1"], st["ll_n"] = C1, float(T)
                st["ac_num"] = (u_ll[1:] * u_ll[:-1]).sum(0, keepdims=True)
                st["ac_den"] = (u_ll * u_ll).sum(0, keepdims=True)
                st["u_sd_n"] = float(T)
                st["u_sd_s1"] = f1dm.sum(0, keepdims=True)
                st["u_sd_s2"] = (f1dm * f1dm).sum(0, keepdims=True)
                st["f6_n"] = float(T)
                st["f6_s1"] = f6.sum(0, keepdims=True)
                st["f6_s2"] = (f6 * f6).sum(0, keepdims=True)
                u0 = f1dm
                sd60 = pd.DataFrame(u0).rolling(60, min_periods=10).std().values
                u_p = np.clip(np.nan_to_num(u0 / (sd60 + self.EPS)), -5.0, 5.0)
                S = np.einsum("ti,tj->ij", u_p, u_p)
                st["pair_S"], st["pair_n"] = S, float(T)
                P = np.cumsum(u_p, axis=0)
                for w in (10, 40):
                    g = P.copy()
                    g[w:] = P[w:] - P[:-w]
                    st[f"gap{w}_n"] = float(T)
                    st[f"gap{w}_s1"] = g.sum(0, keepdims=True)
                    st[f"gap{w}_s2"] = (g * g).sum(0, keepdims=True)
                e3 = self._cs_resid2(f[3], f1, f[2])
                st["e3_n"] = float(T)
                st["e3_s1"] = e3.sum(0, keepdims=True)
                st["e3_s2"] = (e3 * e3).sum(0, keepdims=True)
                return st

            def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
                tk = self._tickers(features)
                n = len(features)
                cut = max(0, n - self.TAIL)
                self._tail = features.iloc[cut:]
                pre = features.iloc[:cut]
                if self.PRIOR_RECENT:
                    pre = pre.iloc[-self.PRIOR_RECENT:]
                if len(pre) > 300:
                    st = self._stats_pass(self._extract(pre, tk))
                    if self.PRIOR_CAP:
                        fac = min(1.0, self.PRIOR_CAP / max(st["ll_n"], 1.0))
                        if fac < 1.0:
                            st = {k: (v * fac if isinstance(v, (int, float, np.ndarray)) else v)
                                  for k, v in st.items()}
                    self._prior = st
                else:
                    self._prior = None
                self.trained = True

            def predict(self, features: pd.DataFrame) -> pd.DataFrame:
                tk = self._tickers(features)
                J = len(tk)
                n_val = len(features)
                if self._tail is not None and len(self._tail) > 0:
                    full = pd.concat([self._tail, features], axis=0)
                else:
                    full = features
                off = len(full) - n_val
                f = self._extract(full, tk)
                f1d = full["Feature.1"][tk]
                f4d = full["Feature.4"][tk]
                f1, f2, f3, f6 = f[1], f[2], f[3], f[6]
                T = f1.shape[0]
                pr = self._prior or {}
                z = lambda shape: np.zeros((1, shape))
                p_n = pr.get("u_sd_n", 0.0)

                leg1 = self._csrank_np(self._tsrank((f4d - f1d).values, 60), J)

                f1dm = self._rowdm(f1)
                u_sd = self._exp_std_prior(f1dm, p_n, pr.get("u_sd_s1", z(J)), pr.get("u_sd_s2", z(J)))
                u = np.clip(f1dm / (u_sd + 1e-12), -3, 3)
                C1 = pr.get("ll_C1", np.zeros((J, J))).copy()
                n_ll = pr.get("ll_n", 0.0)
                sig = np.zeros((T, J))
                warm_ll = 1 if n_ll > 0 else 40
                for t in range(T):
                    if t >= 1:
                        C1 += np.outer(u[t - 1], u[t])
                    if t >= warm_ll:
                        A = C1 / (n_ll + t)
                        np.fill_diagonal(A, 0.0)
                        sig[t] = A.T @ u[t]
                ll = self._csrank_np(sig, J)
                if n_ll == 0:
                    ll[:41] = 0.0
                f6_n = pr.get("f6_n", 0.0)
                s1 = np.cumsum(f6, axis=0) + pr.get("f6_s1", z(J))
                nn = (np.arange(T) + 1.0)[:, None] + f6_n
                f6em = s1 / nn
                f6sd = self._exp_std_prior(f6, f6_n, pr.get("f6_s1", z(J)), pr.get("f6_s2", z(J)))
                f6z = -self._csrank_np((f6 - f6em) / (f6sd + 1e-12), J)
                if not pr:
                    f6z[:self.WARM] = 0.0
                acn = np.cumsum(np.vstack([np.zeros((1, J)), u[1:] * u[:-1]]), axis=0) \
                    + pr.get("ac_num", z(J))
                acd = np.cumsum(u * u, axis=0) + pr.get("ac_den", z(J))
                ac1 = acn / (acd + self.EPS)
                tf_raw = self._csrank_np(np.tanh(5.0 * ac1) * u, J)
                if not pr:
                    tf_raw[:self.WARM] = 0.0
                rv_tf = -self._csrank_np(f1, J)
                rv_tf = rv_tf - rv_tf.mean(1, keepdims=True)
                num_t = (tf_raw * rv_tf).sum(1, keepdims=True)
                den_t = (rv_tf * rv_tf).sum(1, keepdims=True) + 1e-18
                tf_o = self._csrank_np(
                    self._rowdm(tf_raw - (num_t / den_t) * rv_tf), J)

                e3 = self._cs_resid2(f3, f1, f2)
                lagrev = -self._csrank_np(
                    0.9 * self._lag_np(f1, 1) + 0.35 * self._lag_np(f1, 2)
                    - 1.4 * self._ewm_np(e3, 2), J)
                leg2 = self._rowdm(ll + f6z + 0.8 * lagrev)

                e3sd = self._exp_std_prior(e3, pr.get("e3_n", 0.0), pr.get("e3_s1", z(J)),
                                           pr.get("e3_s2", z(J)), min_n=20)
                s3 = np.clip(e3 / (e3sd + self.EPS), -5.0, 5.0)
                x3r = self._csrank_np(self._decay_linear(s3, 5), J)
                if not pr:
                    x3r[:self.WARM] = 0.0

                sd60 = pd.DataFrame(f1dm).rolling(60, min_periods=10).std().values
                u_p = np.clip(np.nan_to_num(f1dm / (sd60 + self.EPS)), -5.0, 5.0)
                outer = u_p[:, :, None] * u_p[:, None, :]
                S_exp = np.cumsum(outer, axis=0) + pr.get("pair_S", np.zeros((J, J)))[None]
                d_exp = np.einsum("tii->ti", S_exp)
                C_exp = S_exp / (np.sqrt(d_exp[:, :, None] * d_exp[:, None, :]) + self.EPS)
                warm_p = 1 if pr else self.WARM
                C_exp[:warm_p] = 0.0
                idx = np.arange(J)
                C_exp[:, idx, idx] = 0.0
                Wp = np.clip(C_exp, 0.0, None)
                rs = Wp.sum(axis=2, keepdims=True)
                Wu = np.full((J, J), 1.0 / (J - 1)); np.fill_diagonal(Wu, 0.0)
                W_full = np.where(rs > self.EPS, Wp / (rs + self.EPS), Wu[None])
                order = np.argsort(C_exp, axis=2)
                W_top3 = np.zeros_like(C_exp)
                rows = np.arange(J)
                for t in range(warm_p, T):
                    top = order[t, :, -3:]
                    W_top3[t, rows[:, None], top] = 1.0 / 3
                P = np.cumsum(u_p, axis=0)

                def gap_raw(w):
                    g = P.copy()
                    g[w:] = P[w:] - P[:-w]
                    return g

                def gap_z(w):
                    g = gap_raw(w)
                    gn = pr.get(f"gap{w}_n", 0.0)
                    gsd = self._exp_std_prior(g, gn, pr.get(f"gap{w}_s1", z(J)),
                                              pr.get(f"gap{w}_s2", z(J)), min_n=20)
                    return g / (gsd + self.EPS)

                def rel_fade(Wmat, zg):
                    avg = np.einsum("tji,ti->tj", Wmat, zg)
                    sig = -self._csrank_np(zg - avg, J)
                    sig[:warm_p] = 0.0
                    return sig

                leg3 = rel_fade(W_top3, gap_z(10))
                leg4 = rel_fade(W_full, self._tsrank(gap_raw(40), 120, minp=40))

                out = (self.W["f4_minus_f1"] * np.nan_to_num(leg1)
                       + self.W["mix_llf6_lagrev"] * np.nan_to_num(leg2)
                       + self.W["ps_exp_top3_10"] * np.nan_to_num(leg3)
                       + self.W["x3r"] * np.nan_to_num(x3r)
                       + self.W["ps_exp_full40"] * np.nan_to_num(leg4))

                if self.BETANEUT:
                    mkt = f1d.mean(axis=1)
                    f1p = pd.DataFrame(f1, index=full.index)
                    mk = pd.Series(mkt.values, index=full.index)
                    f1m = f1p.rolling(250, min_periods=20).mean()
                    mm = mk.rolling(250, min_periods=20).mean()
                    cov = f1p.mul(mk, axis=0).rolling(250, min_periods=20).mean() - f1m.mul(mm, axis=0)
                    varm = (mk * mk).rolling(250, min_periods=20).mean() - mm ** 2
                    B = np.nan_to_num(cov.div(varm + 1e-12, axis=0).fillna(1.0).values, nan=1.0)
                    Bd = B - B.mean(1, keepdims=True)
                    num = (out * Bd).sum(1, keepdims=True)
                    den = (Bd * Bd).sum(1, keepdims=True) + 1e-18
                    out = out - (num / den) * Bd

                if self.REV_NEUT:
                    rv = -self._csrank_np(f1, J)
                    rv = rv - rv.mean(1, keepdims=True)
                    num = (out * rv).sum(1, keepdims=True)
                    den = (rv * rv).sum(1, keepdims=True) + 1e-18
                    out = out - self.REV_NEUT * (num / den) * rv

                self.last_B = B[off:]
                self.last_tf = tf_o[off:]
                out = out[off:]
                pred = pd.DataFrame(out, index=features.index, columns=tk)
                pred = pred.sub(pred.mean(axis=1), axis=0)
                return pred.fillna(0.0)

        class _GBM:

            CONFIGS = [
                dict(hl=1500, cap=8000, target="rankf1", neut=False),
                dict(hl=800, cap=5000, target="rankres", neut=True),
                dict(hl=1500, cap=8000, target="rankres", neut=True),
                dict(hl=3000, cap=12000, target="rankres", neut=True),
            ]
            LGB_PARAMS = dict(
                objective="regression", num_leaves=7, max_depth=3, learning_rate=0.06,
                min_data_in_leaf=1500, feature_fraction=0.7, bagging_fraction=0.6,
                bagging_freq=1, lambda_l2=5.0, num_threads=2, verbosity=-1, seed=7,
                force_row_wise=True,
            )
            N_ROUNDS = 80

            def __init__(self):
                self.models = None
                self.feat_names = None

            def _csr(self, df, J):
                r = df.rank(axis=1, method="average")
                return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

            def _engineer(self, F, J):
                csr = lambda df: self._csr(df, J)
                f1, f2, f3, f4, f5, f6 = (F[i] for i in range(1, 7))
                C = {}
                for i, f in zip(range(1, 7), (f1, f2, f3, f4, f5, f6)):
                    C[f"r{i}"] = csr(f)
                m6 = f6.rolling(500, min_periods=30).mean()
                s6 = f6.rolling(500, min_periods=30).std()
                C["f6z"] = csr(((f6 - m6) / (s6 + 1e-9)).fillna(0.0))
                C["f6chg"] = csr(f6 - f6.rolling(60, min_periods=5).mean())
                s5 = f5.rolling(250, min_periods=20).std()
                t5 = np.tanh(f5 / (s5 + 1e-9)).fillna(0.0)
                C["t5s5"] = csr(t5.rolling(5, min_periods=1).sum())
                C["t5s20"] = csr(t5.rolling(20, min_periods=1).sum())
                adl = np.sign(f1) * np.log1p(f5.abs())
                C["adl60"] = csr(adl.ewm(span=60, min_periods=1).mean())
                C["adl100"] = csr(adl.ewm(span=100, min_periods=1).mean())
                a5 = f5.abs()
                C["vshock"] = csr(a5 / (a5.rolling(250, min_periods=10).mean() + 1e-9))
                mkt = f1.mean(axis=1)
                W = 250
                cov = f1.mul(mkt, axis=0).rolling(W, min_periods=20).mean() - \
                    f1.rolling(W, min_periods=20).mean().mul(
                        mkt.rolling(W, min_periods=20).mean(), axis=0)
                varm = (mkt * mkt).rolling(W, min_periods=20).mean() - \
                    mkt.rolling(W, min_periods=20).mean() ** 2
                beta = cov.div(varm + 1e-12, axis=0).fillna(1.0)
                resid = f1.sub(beta.mul(mkt, axis=0)).fillna(0.0)
                C["ir1"] = csr(resid)
                for w in (5, 20, 60):
                    C[f"irs{w}"] = csr(resid.rolling(w, min_periods=1).sum())
                ip = resid.cumsum()
                C["idd40"] = csr(ip.rolling(40, min_periods=1).max() - ip)
                price = f1.cumsum()
                C["dh10"] = csr(price.rolling(10, min_periods=1).max() - price)
                C["dh40"] = csr(price.rolling(40, min_periods=1).max() - price)
                C["dl20"] = csr(price - price.rolling(20, min_periods=1).min())
                C["macd_a"] = csr(f1.ewm(span=8, min_periods=1).mean()
                                  - f1.ewm(span=40, min_periods=1).mean())
                C["macd_b"] = csr(f1.ewm(span=20, min_periods=1).mean()
                                  - f1.ewm(span=100, min_periods=1).mean())
                for w in (5, 20, 60):
                    C[f"rs{w}"] = csr(f1.rolling(w, min_periods=1).sum())
                C["sd20"] = csr(f1.rolling(20, min_periods=2).std().fillna(0.0))
                C["sd60"] = csr(f1.rolling(60, min_periods=5).std().fillna(0.0))
                names = list(C.keys())
                X = np.stack([np.nan_to_num(C[k].values).astype(np.float32)
                              for k in names], axis=2)
                return X, names

            def _split_features(self, features):
                tickers = sorted(features["Feature.1"].columns,
                                 key=lambda c: int(str(c).split(".")[-1]))
                F = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
                return F, tickers

            def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
                F, tickers = self._split_features(features)
                J = len(tickers)
                X3, names = self._engineer(F, J)
                self.feat_names = names
                f1rank = self._csr(F[1], J).values
                T = X3.shape[0]
                self.models = []
                for cfg in self.CONFIGS:
                    t0 = max(0, T - cfg["cap"])
                    Xs = X3[t0:]
                    ynext = f1rank[t0 + 1:]
                    if cfg["target"] == "rankres":
                        rcur = f1rank[t0:-1]
                        num = (ynext * rcur).sum(axis=1, keepdims=True)
                        den = (rcur * rcur).sum(axis=1, keepdims=True) + 1e-12
                        y2 = ynext - (num / den) * rcur
                    else:
                        y2 = ynext
                    Xs = Xs[:-1]
                    Ts = Xs.shape[0]
                    wt = 0.5 ** ((Ts - 1 - np.arange(Ts)) / cfg["hl"])
                    Xf = Xs.reshape(-1, len(names))
                    yf = y2.reshape(-1)
                    wf = np.repeat(wt, J)
                    m = np.isfinite(yf)
                    ds = lgb.Dataset(Xf[m], label=yf[m], weight=wf[m])
                    mdl = lgb.train(self.LGB_PARAMS, ds, num_boost_round=self.N_ROUNDS)
                    self.models.append((cfg, mdl))

            def predict(self, features: pd.DataFrame) -> pd.DataFrame:
                assert self.models is not None, "Must call train() before predict()"
                F, tickers = self._split_features(features)
                J = len(tickers)
                X3, names = self._engineer(F, J)
                Tv = X3.shape[0]
                Xf = X3.reshape(-1, len(names)).astype(np.float64)
                r1 = X3[:, :, names.index("r1")].astype(np.float64)
                r1 = r1 - r1.mean(axis=1, keepdims=True)
                combo = np.zeros((Tv, J))
                for cfg, mdl in self.models:
                    pm = mdl.predict(Xf, num_threads=2).reshape(Tv, J)
                    pm = pm - pm.mean(axis=1, keepdims=True)
                    if cfg["neut"]:
                        den = (r1 * r1).sum(axis=1, keepdims=True) + 1e-12
                        proj = (pm * r1).sum(axis=1, keepdims=True) / den
                        pm = pm - proj * r1
                    pm = pm / (pm.std(axis=1, keepdims=True) + 1e-12)
                    combo += pm
                combo /= len(self.models)
                out = pd.DataFrame(combo, index=features.index, columns=tickers)
                out = out.sub(out.mean(axis=1), axis=0)
                return out.fillna(0.0)

        def _lw_precision(self, X):
            win, minp = self.COV_WIN, self.COV_MINP
            T, J = X.shape
            Xf = np.nan_to_num(X)
            cs = np.cumsum(np.vstack([np.zeros((1, J)), Xf]), axis=0)
            cso = np.cumsum(
                np.concatenate([np.zeros((1, J, J)), Xf[:, :, None] * Xf[:, None, :]],
                               axis=0), axis=0)
            q = (Xf * Xf).sum(1)
            csq = np.concatenate([[0.0], np.cumsum(q * q)])
            idx = np.arange(1, T + 1)
            lo = np.maximum(0, idx - win)
            n = (idx - lo).astype(float)
            s1 = cs[idx] - cs[lo]
            s2 = cso[idx] - cso[lo]
            sq = csq[idx] - csq[lo]
            mu = s1 / n[:, None]
            S = s2 / n[:, None, None] - mu[:, :, None] * mu[:, None, :]
            m = np.einsum("tii->t", S) / J
            normS2 = (S * S).sum((1, 2))
            d2 = (normS2 - J * m * m) / J
            bbar2 = (sq / n - normS2) / (n * J)
            b2 = np.minimum(np.clip(bbar2, 0.0, None), d2)
            w = np.where(d2 > 1e-30, b2 / np.maximum(d2, 1e-30), 1.0)
            eye = np.eye(J)
            Sig = (1.0 - w)[:, None, None] * S + (w * m)[:, None, None] * eye[None]
            tr = np.einsum("tii->t", Sig) / J
            Sig = Sig + (1e-6 * np.maximum(tr, 1e-18))[:, None, None] * eye[None]
            bad = n < minp
            if bad.any():
                Sig[bad] = eye * np.maximum(tr[bad], 1e-18)[:, None, None]
            return np.linalg.inv(Sig)

        def __init__(self):
            self._apex = self._Apex4()
            self._gbm = self._GBM()

        def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
            self._apex.train(features, target)
            self._gbm.train(features, target)
            self.trained = True

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            pa = self._apex.predict(features)
            pg = self._gbm.predict(features)[pa.columns]
            out = ((self.W_A / self.SCALE_A) * pa.values
                   + (self.W_G / self.SCALE_G) * pg.values
                   + (self.W_T / self.SCALE_T) * self._apex.last_tf)
            if self.COMBINED_BETANEUT:
                B = self._apex.last_B
                num = (out * B).sum(1, keepdims=True)
                den = (B * B).sum(1, keepdims=True) + 1e-18
                out = out - (num / den) * B

            tk = list(pa.columns)
            tail = self._apex._tail
            if tail is not None and len(tail) > 0:
                full = pd.concat([tail, features], axis=0)
            else:
                full = features
            off = len(full) - len(features)
            f1 = np.nan_to_num(full["Feature.1"][tk].values.astype(np.float64))
            prec = self._lw_precision(f1)[off:]
            qq = np.einsum("tij,tj->ti", prec, out)
            nq = np.sqrt((qq * qq).sum(1, keepdims=True)) + 1e-18
            np_out = np.sqrt((out * out).sum(1, keepdims=True))
            qq = qq * (np_out / nq)
            out = (1.0 - self.LAM) * out + self.LAM * qq

            out = out - out.mean(axis=1, keepdims=True)
            pred = pd.DataFrame(out, index=features.index, columns=pa.columns)
            return pred.fillna(0.0)

    @staticmethod
    def _tickers(features):
        cols = [c for c in features.columns if c[0] == "Feature.1"]
        return sorted({c[1] for c in cols}, key=lambda s: int(s.split(".")[-1]))

    def __init__(self):
        self._g = self._GraphLeg()
        self._a = self._Apex7()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self._g.train(features, target)
        self._a.train(features, target)
        tk = self._tickers(features)
        self.tk_ = tk
        self.j_ = len(tk)
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tk = self._tickers(features)
        pg = self._g.predict(features)[tk].to_numpy(dtype=np.float64)
        pa = self._a.predict(features)[tk].to_numpy(dtype=np.float64)

        sig = pg / self.SCALE_GRAPH + pa / self.SCALE_APEX
        sig = sig - sig.mean(axis=1, keepdims=True)
        return pd.DataFrame(sig, index=features.index, columns=tk).fillna(0.0)
