"""
CLEANED IN V106: removed 4 functions, 2 constants, the old run section and 7
imports that nothing in the current project uses. The full original is in
log/originals_v106/sweep_options_v85.py.

V85 - EVERY OPTION, AND THEIR COMBINATIONS, AGAINST THE SIMPLE RULE.

THE OPPONENT IS FIXED IN ADVANCE
--------------------------------
`low bb_position` - the single feature that ranked first in V80's early
selection (2008-2016), i.e. the simple rule a researcher would have deployed in
2017 using the same data V81's features came from. Not "the best single feature
in hindsight": that is a maximum over six tries and biased upward.

HOW A WINNER IS CHOSEN WITHOUT FOOLING ANYONE
---------------------------------------------
Run fifteen-plus variants and crown whichever looks best on 2017-2026, and the
"win" is a best-of-fifteen pick: some variant will lead by luck. So the held-out
decade is split:

    VALIDATION  2017-2021   every variant is scored; the winner is CHOSEN here
    TEST        2022-2026   the chosen winner is CONFIRMED here, once

The verdict is the test-period result of the variant chosen on validation.
Every variant's numbers on every period are still printed, so nothing is hidden;
but only one variant's test result is a claim. The full-decade comparison is
also shown with a Bonferroni-widened interval for the number of variants tried.

"BEATS" MEANS A PAIRED INTERVAL ABOVE ZERO
------------------------------------------
Each arm is measured against its own matched pools (same stock, same month,
only the day differs), and the comparison with the opponent is PAIRED: calendar
months are resampled with replacement, and both arms are re-averaged over the
same resampled months. Both trade the same markets, so the difference has a far
tighter interval than either arm alone.

THE OPTIONS (stage 1 - each one alone, on top of V81)
-----------------------------------------------------
  1  rank objective       XGBoost ranking within stock-month: trains on the
                          question the claim and the test ask (which day)
  2  + trend              px_vs_ema200, monotone "higher is better" - buy the
                          dip in an uptrend
  3  blend                V81 and bb_position percentiles averaged (no refit)
  4  + volatility         atr_pct, unconstrained
  5  more data            train on a bigger universe, test on the same 337
                          (only if RUN_BIG_CACHE is set; slow)
  6  classify objective   predict win/loss instead of R
  7  + short horizon      rsi_2, ret_1, ret_3, dist_low10 (all low is good)
  8  per-stock pct        each feature as a percentile of its own stock's year
  9  + breadth            full-universe market breadth, unconstrained. Caution:
                          the regime pattern was already seen on these years
 10  recency weighting    3-year half-life on training rows
 11  hyperparameters      a shallower and a deeper pre-set (most prone to fit
                          noise - reported, not favoured)

COMBINATIONS (stage 2) are built ONLY from options that improved on V81 in the
VALIDATION years: the cumulative top-2 and top-3, and all helpful options
together, with mutually exclusive choices (objective, transform, weighting,
hyperparameters) resolved in favour of the larger validation gain. A full
factorial of eleven options is 2,048 models - both infeasible and the surest
way to find a lucky one.

The winner is also refit under two more seeds; a lead that moves with the seed
is not a lead.

COST: about ten folds per variant (2017-2026 only - walk-forward folds are
independent, so the 2008-2016 folds add nothing here). With 5 seeds expect a
few minutes per variant; results are cached per variant, so an interrupted run
resumes where it stopped.
"""

import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import measure_v82 as V82
import measure_v83 as V83

warnings.filterwarnings("ignore", message="Mean of empty slice")

SEED = 81
VAL_YEARS = list(range(2017, 2022))
TEST_YEARS = list(range(2022, 2027))
HELD_YEARS = VAL_YEARS + TEST_YEARS
QUANTILE = V83.QUANTILE
MIN_ROWS = V83.MIN_ROWS
OPPONENT = "low bb_position"

BASE_FEATS = ["bb_position", "rsi_14", "ret_5", "px_vs_ema20",
              "range60_position", "px_vs_ema50"]
BASE_MONO = [-1, -1, -1, -1, -1, -1]
SHORT = [("rsi_2", -1), ("ret_1", -1), ("ret_3", -1), ("dist_low10", -1)]
HPARAMS = {"default": {},
           "shallow": dict(max_depth=3, min_child_weight=100,
                           n_estimators=500, learning_rate=0.03),
           "deep": dict(max_depth=6, min_child_weight=30, n_estimators=300,
                        learning_rate=0.04)}


# =============================================================================
# CONFIGS
# =============================================================================
def cfg(objective="reg", extra=(), transform="raw", weight=None,
        hp="default", pool="core", seed_offset=0):
    return {"objective": objective, "extra": [list(e) for e in extra],
            "transform": transform, "weight": weight, "hp": hp,
            "pool": pool, "seed_offset": seed_offset}


BASE = cfg()
STAGE1 = [
    ("1 rank objective", {"objective": "rank"}),
    ("2 + trend (px_vs_ema200)", {"extra": [["px_vs_ema200", 1]]}),
    ("4 + volatility (atr_pct)", {"extra": [["atr_pct", 0]]}),
    ("6 classify objective", {"objective": "clf"}),
    ("7 + short horizon", {"extra": [list(e) for e in SHORT]}),
    ("8 per-stock percentiles", {"transform": "stockpct"}),
    ("9 + breadth", {"extra": [["breadth_full", 0]]}),
    ("10 recency weighting", {"weight": 3.0}),
    ("11a hyperparams shallow", {"hp": "shallow"}),
    ("11b hyperparams deep", {"hp": "deep"}),
]
EXCLUSIVE = ("objective", "transform", "weight", "hp")


def apply_deltas(deltas):
    """Combine option deltas; an exclusive field already set is not replaced."""
    c, used, skipped = cfg(), [], []
    for name, delta in deltas:
        clash = [k for k in delta if k in EXCLUSIVE
                 and c[k] != BASE[k] and c[k] != delta[k]]
        if clash:
            skipped.append(name)
            continue
        for k, v in delta.items():
            if k == "extra":
                have = {e[0] for e in c["extra"]}
                c["extra"] += [list(e) for e in v if e[0] not in have]
            else:
                c[k] = v
        used.append(name)
    return c, used, skipped


def features_of(c):
    feats = ([f + "_spct" for f in BASE_FEATS] if c["transform"] == "stockpct"
             else list(BASE_FEATS))
    mono = list(BASE_MONO)
    for f, m in c["extra"]:
        feats.append(f)
        mono.append(int(m))
    return feats, mono


# =============================================================================
# EXTRA FEATURES
# =============================================================================
def price_features(price_cache, names):
    """One pass over the prices: full-universe breadth + short-horizon set."""
    above, rows = {}, []
    for t in names:
        df = E64.P2.load_prices(t, price_cache)
        if df is None or len(df) < 30:
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")]
        if len(c) >= 260:
            e = c.ewm(span=200, adjust=False).mean()
            a = (c > e).astype(float)
            a.iloc[:200] = np.nan
            above[t] = a
        dlt = c.diff()
        up = dlt.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
        dn = (-dlt).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi2 = np.where(dn > 0, 100 - 100 / (1 + up / dn),
                            np.where(up > 0, 100.0, 50.0))
        rows.append(pd.DataFrame({
            "ticker": t, "date_n": c.index, "rsi_2": rsi2,
            "ret_1": (c / c.shift(1) - 1).to_numpy(),
            "ret_3": (c / c.shift(3) - 1).to_numpy(),
            "dist_low10": (c / c.rolling(10).min() - 1).to_numpy()}))
    W = pd.DataFrame(above)
    n = W.notna().sum(axis=1)
    b = W.mean(axis=1, skipna=True)
    b[n < 50] = np.nan
    return b, pd.concat(rows, ignore_index=True)


# =============================================================================
# MODELS
# =============================================================================
def build(c, seed):
    from xgboost import XGBRegressor, XGBClassifier, XGBRanker
    params = dict(E64.XGB_PARAMS)
    params.update(HPARAMS[c["hp"]])
    _, mono = features_of(c)
    params["monotone_constraints"] = tuple(mono)
    if c["objective"] == "reg":       # identical to M81.build_model
        params.pop("objective", None)
        params.pop("eval_metric", None)
        return XGBRegressor(random_state=seed, objective="reg:squarederror",
                            **params)
    if c["objective"] == "clf":
        return XGBClassifier(random_state=seed, **params)
    params.pop("objective", None)
    params.pop("eval_metric", None)
    return XGBRanker(random_state=seed, objective="rank:pairwise", **params)


def fit_predict(c, tr, Xs, seed, start):
    feats, _ = features_of(c)
    m = build(c, seed)
    if c["objective"] == "rank":
        tr = tr.sort_values("_qid", kind="mergesort")
    w = None
    if c["weight"]:
        age = (start - pd.DatetimeIndex(tr["date"])).days.to_numpy(float)
        w = 0.5 ** (age / (365.25 * float(c["weight"])))
    if c["objective"] == "reg":
        m.fit(tr[feats], tr["r_multiple"], sample_weight=w, verbose=False)
        return [m.predict(X[feats]) for X in Xs]
    if c["objective"] == "clf":
        m.fit(tr[feats], tr["label"], sample_weight=w, verbose=False)
        return [m.predict_proba(X[feats])[:, 1] for X in Xs]
    q = tr["_qid"].to_numpy()
    gw = None
    if w is not None:
        first = np.r_[True, q[1:] != q[:-1]]
        gw = w[first]
    m.fit(tr[feats], tr["_grade"], qid=q, sample_weight=gw, verbose=False)
    return [m.predict(X[feats]) for X in Xs]


def model_fire(d, te, refs):
    fr = d.loc[te["row_id"].to_numpy(), ["date", "year"]].reset_index(
        drop=True)
    fr["p"] = te["p"].to_numpy(float)
    cut = V83.fold_cut(fr, refs)
    f = V83.apply_rule(fr["p"].to_numpy(float), cut, strict=True)
    fire = np.zeros(len(d), bool)
    fire[te["row_id"].to_numpy()[f]] = True
    return fire


def blend_fire(d, refs, good_pct, w_model=0.5):
    """
    Model percentile (against the fold's own pre-year window, so fold scale
    cannot leak in) averaged with the rule's trailing percentile, then the same
    strict rolling 1% cut on the blended score. Causal and warm.
    """
    win = V82._win()
    fire = np.zeros(len(d), bool)
    yr = d["year"].to_numpy()
    for y, ref in refs.items():
        start = pd.Timestamp(f"{y}-01-01").to_datetime64()
        rid = ref["row_id"].to_numpy()
        rp = ref["p"].to_numpy(float)
        rd = pd.DatetimeIndex(ref["date"]).values
        pre = np.sort(rp[(rd < start) & np.isfinite(rp)])
        if pre.size < MIN_ROWS:
            continue
        bl = w_model * (np.searchsorted(pre, rp, side="right") / pre.size) \
            + (1 - w_model) * good_pct[rid]
        o = np.argsort(rd, kind="mergesort")
        rds, bls = rd[o], bl[o]
        is_t = yr[rid] == y
        trid, tbl = rid[is_t], bl[is_t]
        tper = pd.PeriodIndex(pd.DatetimeIndex(rd[is_t]), freq="M")
        for m in pd.unique(tper):
            t = m.to_timestamp()
            lo = np.searchsorted(rds, (t - win).to_datetime64(), "left")
            hi = np.searchsorted(rds, t.to_datetime64(), "left")
            w = bls[lo:hi]
            w = w[np.isfinite(w)]
            if w.size < MIN_ROWS:
                continue
            sel = tper == m
            fire[trid[sel]] = tbl[sel] > float(np.quantile(w, 1 - QUANTILE))
    return fire


# =============================================================================
# EVALUATION - analytic matched control, paired month bootstrap
# =============================================================================
class Evaluator:
    """
    Excess per signal = outcome minus the mean outcome of its own stock-month
    pool: the exact expectation of the matched control (which samples from
    that pool), additive across any subset, so arms can be paired month by
    month. Resampling weights are drawn once per period and shared by every
    arm, which is what makes the comparisons paired.
    """

    def __init__(self, d, n_boot=4000, seed=SEED):
        per = pd.PeriodIndex(pd.DatetimeIndex(d["date"]), freq="M")
        key = d["ticker"].astype(str).to_numpy() + "|" + \
            per.astype(str).to_numpy()
        lab = d["label"].to_numpy(float)
        r = d["r_multiple"].to_numpy(float)
        pl = pd.Series(lab).groupby(key).transform("mean").to_numpy()
        pr = pd.Series(r).groupby(key).transform("mean").to_numpy()
        self.ex_w, self.ex_r = lab - pl, r - pr
        self.year = d["year"].to_numpy()
        self.mcode, self.months = pd.factorize(per)
        self.myear = np.array([m.year for m in self.months])
        self.n_boot, self.rng = n_boot, np.random.default_rng(seed)
        self._W = {}

    def _weights(self, years):
        k = tuple(years)
        if k not in self._W:
            idx = np.flatnonzero(np.isin(self.myear, years))
            W = self.rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)),
                                     size=self.n_boot).astype(float)
            self._W[k] = (idx, W)
        return self._W[k]

    def _sums(self, fire, years):
        idx, W = self._weights(years)
        m = fire & np.isin(self.year, years)
        k = len(self.months)
        sw = np.bincount(self.mcode[m], self.ex_w[m], k)[idx]
        sr = np.bincount(self.mcode[m], self.ex_r[m], k)[idx]
        cn = np.bincount(self.mcode[m], minlength=k)[idx].astype(float)
        return m, W, sw, sr, cn

    def summary(self, fire, years):
        m, W, sw, sr, cn = self._sums(fire, years)
        if cn.sum() < 30:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = (W @ sw) / (W @ cn) * 100
        yrs = [y for y in years]
        per_y = [self.ex_w[m & (self.year == y)] for y in yrs]
        ym = [v.mean() * 100 for v in per_y if v.size >= 15]
        cnt = [int((m & (self.year == y)).sum()) for y in yrs]
        return {"signals": int(cn.sum()),
                "lift": sw.sum() / cn.sum() * 100,
                "lo": np.nanpercentile(boot, 5), "hi": np.nanpercentile(boot, 95),
                "expR": sr.sum() / cn.sum(),
                "cond": f"{sum(v > 0 for v in ym)}/{len(ym)}",
                "worst": min(ym) if ym else np.nan,
                "max_year": max(cnt) if cnt else 0}

    def paired(self, fa, fb, years, k_tests=1):
        _, W, swa, _, cna = self._sums(fa, years)
        _, _, swb, _, cnb = self._sums(fb, years)
        if cna.sum() < 30 or cnb.sum() < 30:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            d = ((W @ swa) / (W @ cna) - (W @ swb) / (W @ cnb)) * 100
        a = 5.0 / k_tests
        return {"diff": (swa.sum() / cna.sum() - swb.sum() / cnb.sum()) * 100,
                "lo": np.nanpercentile(d, 5), "hi": np.nanpercentile(d, 95),
                "lo_bonf": np.nanpercentile(d, a),
                "hi_bonf": np.nanpercentile(d, 100 - a),
                "p_le0": float(np.nanmean(d <= 0))}
