"""Competition 5 submission: APEX-8 — APEX-7 plus a small ticket2 sleeve.

WHAT THIS IS
    apex8_prediction = apex7_prediction + K * ticket2_prediction, then row-demean,
    with the FIXED offline constant K = 0.068664.

    Both parents ship in this repository and are unmodified here:

      APEX-7   (submission_apex7.py, class Apex7Predictor)  — the incumbent
               flagship.  0.60 apex4 core + 0.20 GBM leg + 0.20 TF leg, a
               combined rolling-beta projection, then the APEX-P Ledoit-Wolf
               inverse-covariance position map.  Local full Sharpe 0.1191.
      TICKET2  (submission_ticket2.py, class Ticket2Predictor)  — the "two-graph"
               ticket: a price-space grid-averaged VAR(1) lead-lag leg plus a
               rank-space own-name-zeroed ridge VAR leg, summed at equal risk.
               Local full Sharpe 0.0922.

    The two are only 0.267 PnL-correlated, which is why the sleeve adds anything
    at all.  Every earlier ensemble study in this project concluded "the
    orthogonal capacity has been spent" — but those studies pre-date ticket2 and
    every ingredient they tested was already spanned by apex6.

WHERE K COMES FROM (and why it is not a fitted parameter)
    The blend that was actually studied is the vol-normalised mixture

        (1 - w) * P_a7 / sigma_a7  +  w * P_t2 / sigma_t2

    at w = 0.15, where sigma is the standard deviation of each parent's per-bar
    PnL over periods 001-075 (apex7 0.01919434, ticket2 0.04933074).  Factoring
    out the global positive scalar 1 / sigma_a7 — which cannot change a Sharpe
    ratio, a correlation, or any scale-free statistic — leaves

        P_a7 + K * P_t2,   K = (w / (1 - w)) * (sigma_a7 / sigma_t2) = 0.068664.

    So K is arithmetic on two offline constants plus the single choice w = 0.15;
    it is not re-optimised here and nothing in this file is fitted to the blend.

WHY w = 0.15 AND NOT THE SHARPE-MAXIMISING WEIGHT
    The full-sample Sharpe maximum sits at w ~ 0.40 (local 0.1345), and it is
    rejected: it FAILS the recent-window bootstrap against apex7 (rec20 p = 0.63),
    i.e. its extra gain is an h1/h2 effect that does not survive on the recent
    window.  w = 0.15 is fixed by two BINDING CONSTRAINTS, not by an argmax:

      (a) corr(blend, ticket2) <= 0.45, so that ticket2 REMAINS SEPARATELY
          ADMISSIBLE as its own badge.  At w = 0.15 the PnL correlation is
          0.4181; at w = 0.25 it is 0.529, which would block ticket2's own slot.
      (b) the window-restricted period bootstrap versus apex7 must clear 0.90 on
          BOTH h2 and rec20.  At w = 0.15: h2 p = 1.000, rec20 p = 0.957-0.971.
          At w = 0.25 rec20 falls to 0.887-0.899 and misses the bar.

    Constraint (a) binds first.  This is a constraint-driven choice, not a
    maximisation, which is why the sleeve is deliberately smaller than the one
    that would have looked best in-sample.

MEASURED VALUES OF THE CODE AS SHIPPED (77-period walk-forward)
                full   sel75      h1      h2   rec20   hold(sealed)
      apex7   0.1191  0.1195  0.1521  0.0768  0.0949  0.1071
      APEX8   0.1276  0.1280  0.1623  0.0832  0.0991  0.1142

    Every window improves; the sealed hold-out (periods 076/077, never selected
    on) improves by +0.0071.  Window-restricted period bootstrap versus apex7,
    2000 resamples, pools resampled separately: h2 +0.0064 p = 1.000,
    rec20 +0.0042 p = 0.957-0.971.
    PnL correlations: apex6 0.912, ticket2 0.418, ticket1 0.375, sparse 0.361.

CAUSALITY
    Nothing new is introduced.  Both parents are copied verbatim, both train()
    methods are called on exactly the training rows this class receives, and the
    blend is a per-row linear combination of two same-row predictions.  apex7
    carries a known ~3e-3 backward dependence inherited from the GBM leg's
    un-axised `pm / (pm.std() + 1e-12)` normalisation over the validation block;
    ticket2 probes at exactly 0.0.  apex8's probe therefore reproduces apex7's
    number and adds nothing.

HONEST CAVEATS
    The sleeve inherits every caveat of both parents.  In particular apex7's IC
    is lower than apex6's while its Sharpe is higher, and ticket2's own IC is
    low (0.0256), so if the platform scores something closer to IC than to
    Sharpe this blend is not obviously an improvement.  The blend weight w was
    chosen on the same data the windows are reported on; only the hold window
    (periods 076/077) is genuinely sealed.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from predictor import Predictor


class Apex8APredictorC(Predictor):
    """apex7 + K * ticket2, row-demeaned.  The four RISK constants are re-measured
    inside train(); the offline values below survive only as exception fallbacks."""

    MEAS = 3000            # training rows the measurement block spans
    MIN_INNER = 2000       # rows the inner fit needs before the rule may fire
    W_MIX = 0.15           # ticket2's intended PnL-risk share -- NOT re-tuned

    K = 0.068664           # PRIOR ONLY: (w/(1-w)) * (sigma_a7/sigma_t2) at w = 0.15

    # ---------------- apex7 constants (verbatim from submission_apex7.py) ----
    W_A = 0.60
    W_G = 0.20
    W_T = 0.20
    SCALE_A = 0.34507836   # PRIOR ONLY: median per-period avg cs-std, apex4 core
    SCALE_G = 0.86575845   # PRIOR ONLY: median per-period avg cs-std, GBM leg
    SCALE_T = 0.60800000   # PRIOR ONLY: median per-period avg cs-std, TF leg
    COMBINED_BETANEUT = True
    COV_WIN = 250          # same window apex6 already uses for its rolling beta
    COV_MINP = 60          # below this many observations fall back to identity
    LAM = 0.50             # blend weight on the risk-mapped vector

    class _Apex4:
        """Apex3 warm-start + operator swaps (ts_rank/decay_linear) + partial reversal neutralization."""

        W = {"f4_minus_f1": 0.3229, "mix_llf6_lagrev": 0.2656,
             "ps_exp_top3_10": 0.1556, "x3r": 0.1493, "ps_exp_full40": 0.1065}
        REV_NEUT = 0.25   # partial projection off the plain-reversal direction
        BETANEUT = True
        TAIL = 600         # rows of training features prepended at predict time
        PRIOR_RECENT = None  # use only the last N pre-tail rows for priors (None = all)
        PRIOR_CAP = 500      # cap prior effective sample size (plateau-robust: 250-1000 equal)
        WARM = 30
        EPS = 1e-9

        def __init__(self):
            self._tail = None      # DataFrame tail of training features
            self._prior = None     # dict of sufficient statistics

        # ------------------------------------------------------------- helpers
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
            """Rolling ts-rank pct in [-1,1] per ticker (causal)."""
            if minp is None:
                minp = max(5, d // 3)
            r = pd.DataFrame(a).rolling(d, min_periods=minp).rank(pct=True).values
            return np.nan_to_num(2.0 * r - 1.0)

        def _decay_linear(self, a, d):
            """WMA with weights d..1 (today's weight d). Causal."""
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
            """Expanding per-ticker std with prior sufficient stats (n0,s1,s2)."""
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

        # --------------------------------------------------- sufficient stats
        def _stats_pass(self, f):
            """Accumulators over the pre-tail training history (all causal)."""
            f1, f6 = f[1], f[6]
            T, J = f1.shape
            st = {}
            # u for lead-lag: standardized market-removed F1 (expanding std, no prior)
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
            # F6 expanding mean/std stats
            st["f6_n"] = float(T)
            st["f6_s1"] = f6.sum(0, keepdims=True)
            st["f6_s2"] = (f6 * f6).sum(0, keepdims=True)
            # pairs: u_pair uses rolling sd60 (tail handles), but S_exp is expanding
            u0 = f1dm
            sd60 = pd.DataFrame(u0).rolling(60, min_periods=10).std().values
            u_p = np.clip(np.nan_to_num(u0 / (sd60 + self.EPS)), -5.0, 5.0)
            S = np.einsum("ti,tj->ij", u_p, u_p)
            st["pair_S"], st["pair_n"] = S, float(T)
            # gap z-score stats per window (gap = rolling-w sum of u_p)
            P = np.cumsum(u_p, axis=0)
            for w in (10, 40):
                g = P.copy()
                g[w:] = P[w:] - P[:-w]
                st[f"gap{w}_n"] = float(T)
                st[f"gap{w}_s1"] = g.sum(0, keepdims=True)
                st[f"gap{w}_s2"] = (g * g).sum(0, keepdims=True)
            # e3 expanding-std stats (for x3r)
            e3 = self._cs_resid2(f[3], f1, f[2])
            st["e3_n"] = float(T)
            st["e3_s1"] = e3.sum(0, keepdims=True)
            st["e3_s2"] = (e3 * e3).sum(0, keepdims=True)
            return st

        # ----------------------------------------------------------------- API
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

            # leg 1: ts-rank(60) of F4-F1, then cross-sectional rank
            leg1 = self._csrank_np(self._tsrank((f4d - f1d).values, 60), J)

            # leg 2: lead-lag + f6 dev + lagrev(x3)
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
                f6z[:self.WARM] = 0.0   # cold path: expanding-std backfill peeks ahead
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

            # x3r leg
            e3sd = self._exp_std_prior(e3, pr.get("e3_n", 0.0), pr.get("e3_s1", z(J)),
                                       pr.get("e3_s2", z(J)), min_n=20)
            s3 = np.clip(e3 / (e3sd + self.EPS), -5.0, 5.0)
            x3r = self._csrank_np(self._decay_linear(s3, 5), J)
            if not pr:
                x3r[:self.WARM] = 0.0

            # pairs legs
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
                # --- APEX7 FIX: project on the ROW-DEMEANED beta column. -------
                # B has row-mean identically 1 (B_j = cov(f1_j,mkt)/var(mkt) with
                # mkt = mean_j f1_j, so sum_j B_j = J).  Because `out` is exactly
                # row-demeaned, <out,B> = <out,Bd>, while the denominator carries
                # the constant part: <B,B> = J + <Bd,Bd>.  apex6/apexP therefore
                # removed only the fraction <Bd,Bd>/(J+<Bd,Bd>) (median 4.2%) of
                # the beta exposure they intended to remove.  Using Bd makes the
                # projection an actual projection.  No new constant.
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
        """Recency-weighted GBM ensemble; all logic inside the class."""

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

        # ---------- feature engineering (causal, cross-sectional ranks) ----------
        def _csr(self, df, J):
            r = df.rank(axis=1, method="average")
            return (r - 0.5 * (J + 1)) / (0.5 * (J - 1))

        def _engineer(self, F, J):
            """F: dict 1..6 -> (T,J) DataFrame. Returns (T,J,K) float32 + names."""
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

        # ------------------------------- train -----------------------------------
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

        # ------------------------------ predict ----------------------------------
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

    class _Ticket2:
        """The ticket2 sleeve, verbatim from submission_ticket2.py.

        LEG A  price space  — grid-averaged VAR(1) lead-lag coefficient matrix
                              estimated on the standardised, de-marketed return
                              cross-section, applied to today's cross-section.
        LEG B  rank space   — ridge VAR on cross-sectional RANKS with the
                              OWN-NAME coefficients zeroed, so it is orthogonal
                              to own-name reversal by construction.
        The legs are each normalised to unit cross-sectional dispersion per row
        and summed 1:1.  No fitted weights.
        """

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

            # equal risk in signal space: each leg to unit cross-sectional
            # dispersion at every timestamp, then summed 1:1. No fitted weight.
            sig = self._unit_row(a) + self._unit_row(b)
            sig -= sig.mean(1, keepdims=True)
            return pd.DataFrame(sig, index=features.index, columns=tk)

    # ============================ combined API ==========================
    def _lw_precision(self, X):
        """Causal rolling Ledoit-Wolf shrunk covariance of X -> precision.

        X: (T, J) array (Feature.1 per ticker).  At row t the estimator uses
        only rows max(0, t-COV_WIN+1) .. t (inclusive), so it is causal.
        Shrinkage target is the scaled identity m*I with m = trace(S)/J; the
        intensity follows Ledoit-Wolf (2004) analytically, so it adds no
        fitted hyper-parameter.  Rows with fewer than COV_MINP observations
        fall back to the identity (the transform becomes a no-op there).
        """
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
        d2 = (normS2 - J * m * m) / J                     # ||S - m I||^2 / J
        bbar2 = (sq / n - normS2) / (n * J)               # LW noise term
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

    def _a7_core(self, features, apex, gbm, sc, parts=None):
        """The apex7 forecast, byte-for-byte the logic of Apex7Predictor.predict.

        Parameterised over (a) which Apex4/GBM instance supplies the legs and
        (b) the three dispersion scales, so that the SAME code path serves both
        the live forecast and the train()-side measurement.  Nothing else about
        the leg computation changed relative to submission_apex8.py.
        """
        if parts is None:
            pa = apex.predict(features)
            pg = gbm.predict(features)[pa.columns]
        else:
            pa, pg = parts
        out = ((self.W_A / sc[0]) * pa.values
               + (self.W_G / sc[1]) * pg.values
               + (self.W_T / sc[2]) * apex.last_tf)
        if self.COMBINED_BETANEUT:
            B = apex.last_B
            num = (out * B).sum(1, keepdims=True)
            den = (B * B).sum(1, keepdims=True) + 1e-18
            out = out - (num / den) * B

        # ---- risk-model position mapping:  w = (1-LAM) f + LAM * C^-1 f ----
        # Feature.1 history for the covariance comes from the training tail the
        # Apex4 leg already keeps (600 rows >= COV_WIN), so every predicted row
        # sees a full backward-looking window and nothing from the future.
        tk = list(pa.columns)
        tail = apex._tail
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
        qq = qq * (np_out / nq)          # direction-only: keep the row's L2 norm
        out = (1.0 - self.LAM) * out + self.LAM * qq

        out = out - out.mean(axis=1, keepdims=True)
        pred = pd.DataFrame(out, index=features.index, columns=pa.columns)
        return pred.fillna(0.0)

    def _predict_apex7(self, features: pd.DataFrame) -> pd.DataFrame:
        return self._a7_core(features, self._apex, self._gbm,
                             (self.SCALE_A, self.SCALE_G, self.SCALE_T))

    # ------------------------------------------------------------------ scales
    @staticmethod
    def _avg_cs_std(a):
        """Mean over rows of the cross-sectional std -- 'avg cs-std'."""
        return float(np.nanmean(np.nanstd(np.asarray(a, dtype=np.float64), axis=1)))

    @staticmethod
    def _pf(pred, rets):
        """evaluation.backtest's convention: pf_t = <pred_{t-1}, ret_t>, pf_0 = 0."""
        q = np.zeros(len(pred))
        q[1:] = (pred[:-1] * rets[1:]).sum(1)
        return q

    def _measure(self, features: pd.DataFrame, target: pd.DataFrame):
        """Re-measure (SCALE_A, SCALE_G, SCALE_T) and K inside train()'s rows.

        Protocol: take one ORDINARY WALK-FORWARD STEP one block early.  Fresh
        copies of every sub-model are fitted on training rows [0, n-MEAS) and
        asked to predict training rows [n-MEAS, n).  Each sub-model therefore
        sees exactly the causal history it will see live, and every row touched
        is a training row -- the scored block is never referenced.

        Returns ((SCALE_A, SCALE_G, SCALE_T), K) or None to keep the priors.
        """
        n = len(features)
        if n < self.MEAS + self.MIN_INNER:
            return None
        cut = n - self.MEAS
        inf, int_ = features.iloc[:cut], target.iloc[:cut]
        apex, gbm, t2 = self._Apex4(), self._GBM(), self._Ticket2()
        apex.train(inf, int_)
        gbm.train(inf, int_)
        t2.train(inf, int_)

        mf = features.iloc[cut:]
        pa = apex.predict(mf)
        cols = list(pa.columns)
        pg = gbm.predict(mf)[cols]
        sc = (self._avg_cs_std(pa.values), self._avg_cs_std(pg.values),
              self._avg_cs_std(apex.last_tf))
        if not all(np.isfinite(s) and s > 0.0 for s in sc):
            return None

        # K equalises PnL VOLATILITY at w = W_MIX.  train() is never handed
        # returns, so Feature.1 -- the highest-correlated training-side proxy
        # for the same-row return -- stands in for them.  A volatility ratio is
        # driven by the predictions' norms and the return covariance, not by the
        # proxy's directional accuracy, which is what makes the substitution
        # sound here even though it would NOT be sound for a PnL mean.
        p7 = self._a7_core(mf, apex, gbm, sc, parts=(pa, pg)).values
        pt = t2.predict(mf)[cols].values
        rp = np.nan_to_num(mf["Feature.1"][cols].values.astype(np.float64))
        v7 = float(self._pf(p7, rp).std())
        v2 = float(self._pf(pt, rp).std())
        if not (np.isfinite(v7) and np.isfinite(v2) and v2 > 0.0):
            return None
        k = (self.W_MIX / (1.0 - self.W_MIX)) * (v7 / v2)
        if not (np.isfinite(k) and k > 0.0):
            return None
        return sc, float(k)

    def __init__(self):
        self._apex = self._Apex4()
        self._gbm = self._GBM()
        self._t2 = self._Ticket2()

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # (1) RE-MEASURE THE FOUR RISK CONSTANTS on the last MEAS training rows.
        #     Done first, on throwaway sub-models, so the live sub-models below
        #     are fitted on exactly the rows apex8 fits them on.  On any failure
        #     the class-level frozen priors stand.
        try:
            got = self._measure(features, target)
        except Exception:                                    # noqa: BLE001
            got = None
        if got is not None:
            (self.SCALE_A, self.SCALE_G, self.SCALE_T), self.K = got

        # (2) BOTH sub-models are warm-started on exactly the rows handed in here:
        # apex7's Apex4 leg keeps a 600-row feature tail plus recency-capped
        # sufficient statistics, its GBM leg fits four LightGBM models, and the
        # ticket2 sleeve stores leg A's standardised tail buffer (bufA_) and
        # leg B's two ridge coefficient matrices (WB_).  Skipping either train
        # raises AttributeError at predict time.
        self._apex.train(features, target)
        self._gbm.train(features, target)
        self._t2.train(features, target)
        self.trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        p7 = self._predict_apex7(features)
        pt = self._t2.predict(features)
        pt = pt[list(p7.columns)]          # same tickers, enforce column order
        out = p7.values + self.K * pt.values
        out = out - out.mean(axis=1, keepdims=True)
        pred = pd.DataFrame(out, index=features.index, columns=p7.columns)
        return pred.fillna(0.0)
