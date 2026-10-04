"""
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

import os
import sys
import time
import json
import pickle
import hashlib
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import measure_v83 as V83

warnings.filterwarnings("ignore", message="Mean of empty slice")

SEED = 81
OUT_DIR = "thesis_tables_v85"
CACHE_DIR = "sweep_cache_v85"
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


def key_of(c, n_seeds):
    return hashlib.sha256(json.dumps([c, n_seeds, SEED, HELD_YEARS],
                                     sort_keys=True).encode()).hexdigest()[:16]


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


def stock_pct(d, cols, days=365, min_periods=10):
    """Each feature as a percentile of the same stock's trailing year. Causal."""
    out = {c + "_spct": np.full(len(d), np.nan) for c in cols}
    dd = d[["ticker", "date"] + cols].copy()
    dd["date"] = pd.DatetimeIndex(dd["date"])
    dd["_i"] = np.arange(len(d))
    for _, g in dd.groupby("ticker", sort=False):
        g = g.sort_values("date")
        gi = g.set_index("date")
        idx = g["_i"].to_numpy()
        for c in cols:
            r = gi[c].rolling(f"{days}D", min_periods=min_periods).apply(
                lambda w: (w <= w[-1]).mean(), raw=True)
            out[c + "_spct"][idx] = r.to_numpy()
    return pd.DataFrame(out, index=d.index)


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


def walk_forward(c, d, d_pool, n_seeds, horizon):
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    win = V82._win()
    dd = pd.DatetimeIndex(d["date"])
    dp = pd.DatetimeIndex(d_pool["date"])
    parts, refs = [], {}
    base_seed = SEED + int(c["seed_offset"])
    for y in HELD_YEARS:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d_pool[dp < start - embargo]
        te = d[d["year"] == y]
        ref = d[(dd >= start - win) & (dd < start)]
        if len(tr) < 2000 or len(te) < 200:
            continue
        pt, pr = [], []
        for s in range(n_seeds):
            a, b = fit_predict(c, tr, [te, ref], base_seed + s, start)
            pt.append(a)
            pr.append(b)
        pt, pr = np.mean(pt, axis=0), np.mean(pr, axis=0)
        parts.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(),
                                   "p": pt}))
        refs[y] = pd.DataFrame({
            "row_id": np.r_[ref["row_id"].to_numpy(), te["row_id"].to_numpy()],
            "date": np.r_[ref["date"].to_numpy(), te["date"].to_numpy()],
            "p": np.r_[pr, pt]})
    return pd.concat(parts, ignore_index=True), refs


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


# =============================================================================
# RUNNER
# =============================================================================
def run_v85(model_in="entry_model_v81.joblib", price_cache=None,
            big_cache=None, n_seeds=5, n_boot=4000, run_combos=True,
            run_seed_check=True, resume=True, only=None, out_dir=OUT_DIR,
            verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    m81 = M81.load_model(model_in)
    prov = m81["provenance"]
    price_cache = price_cache or prov["price_cache"]
    t0 = time.time()

    print("=" * 104)
    print("V85 - EVERY OPTION AGAINST THE SIMPLE RULE")
    print("=" * 104)
    print(f"  opponent fixed in advance: {OPPONENT} | choose on "
          f"{VAL_YEARS[0]}-{VAL_YEARS[-1]}, confirm on "
          f"{TEST_YEARS[0]}-{TEST_YEARS[-1]} | {n_seeds} seeds per model")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, prov["horizon"], prov["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year
    # the same ordering V82/V83 used, so the V81 base reproduces exactly
    d = d.sort_values("date").reset_index(drop=True)
    d["row_id"] = np.arange(len(d))

    print("  extra features: breadth + short horizon (one pass over prices), "
          "per-stock percentiles ...")
    b, sh = price_features(price_cache, names)
    d["date_n"] = pd.DatetimeIndex(d["date"]).normalize()
    d["breadth_full"] = d["date_n"].map(b).to_numpy(float)
    d = d.merge(sh, on=["ticker", "date_n"], how="left")
    d = d.sort_values("row_id").reset_index(drop=True)
    d = pd.concat([d, stock_pct(d, BASE_FEATS)], axis=1)
    per = pd.PeriodIndex(pd.DatetimeIndex(d["date"]), freq="M")
    d["_qid"] = pd.factorize(d["ticker"].astype(str) + "|"
                             + per.astype(str))[0]
    d["_grade"] = d["outcome"].map({"loss": 0, "flat": 1, "win": 2}) \
        .fillna(1).astype(int)

    d_big = None
    if big_cache:
        print(f"  building the training pool from {big_cache} (slow) ...")
        nb = sorted(os.path.splitext(f)[0]
                    for f in os.listdir(big_cache) if f.endswith(".pkl"))
        d_big = E64.build_dataset(nb, big_cache, prov["horizon"],
                                  prov["step"], verbose=verbose)
        d_big = E64.apply_label_mode(d_big, prov["label_mode"])
        d_big = d_big.sort_values("date").reset_index(drop=True)

    ev = Evaluator(d, n_boot=n_boot)

    # ---- simple rules (warm from 2005, strict) ------------------------------
    def rule_fire(score):
        cut = V83.rolling_cut(d["date"], score)
        return V83.apply_rule(score, cut, strict=True)
    rules = {OPPONENT: rule_fire(-d["bb_position"].to_numpy(float)),
             "low rsi_14": rule_fire(-d["rsi_14"].to_numpy(float)),
             "low range60_position": rule_fire(
                 -d["range60_position"].to_numpy(float)),
             "composite v2": rule_fire(V82.composite_v2(
                 d, list(zip(BASE_FEATS, BASE_MONO))))}
    opp = rules[OPPONENT]
    good_bb = 1.0 - V82.trailing_pct(d, "bb_position")

    # ---- run variants -------------------------------------------------------
    results, fires, runs = [], dict(rules), {}

    def run_variant(name, c):
        if only and name not in only and name != "V81 base":
            return None
        k = key_of(c, n_seeds)
        path = os.path.join(CACHE_DIR, f"{k}.pkl")
        t = time.time()
        if resume and os.path.exists(path):
            with open(path, "rb") as fh:
                te, refs = pickle.load(fh)
            how = "cached"
        else:
            pool = d if c["pool"] == "core" else d_big
            if pool is None:
                return None
            try:
                te, refs = walk_forward(c, d, pool, n_seeds, prov["horizon"])
            except Exception as e:
                print(f"    {name:<44} FAILED: {e}")
                return None
            with open(path, "wb") as fh:
                pickle.dump((te, refs), fh)
            how = f"{time.time() - t:5.0f}s"
        f = model_fire(d, te, refs)
        fires[name] = f
        runs[name] = (c, te, refs)
        s = ev.summary(f, VAL_YEARS)
        print(f"    {name:<44} {how:>7}  val lift "
              f"{s['lift'] if s else float('nan'):+6.2f} pp  "
              f"[{(time.time() - t0) / 60:5.1f} min total]")
        return f

    print(f"\n  [1] STAGE 1 - V81 and each option alone")
    run_variant("V81 base", BASE)
    for name, delta in STAGE1:
        c, _, _ = apply_deltas([(name, delta)])
        run_variant(name, c)
    if big_cache:
        run_variant("5 more data (big universe)", cfg(pool="big"))

    # blend of V81 with the rule (option 3) - no refit
    if "V81 base" in runs:
        _, _, refs = runs["V81 base"]
        fires["3 blend V81 + bb_position"] = blend_fire(d, refs, good_bb)

    # ---- integrity: V81 base must reproduce V83 ----------------------------
    ref_csv = os.path.join("thesis_tables_v83", "arms_held-out.csv")
    if "V81 base" in fires and os.path.exists(ref_csv):
        t = pd.read_csv(ref_csv)
        r = t[(t["Arm"] == "V81") & (t["Rule"] == ">")]
        if len(r):
            n_ref = int(r["Signals"].iloc[0])
            n_now = int((fires["V81 base"]
                         & np.isin(d["year"], HELD_YEARS)).sum())
            print(f"\n  INTEGRITY  V81 base fires {n_now:,} on 2017-2026; V83 "
                  f"recorded {n_ref:,}: "
                  f"{'REPRODUCED' if n_now == n_ref else '*** DIFFERENT ***'}")

    # ---- stage 2 combinations from VALIDATION gains --------------------------
    base_f = fires.get("V81 base")
    gains = []
    if base_f is not None:
        for name, delta in STAGE1:
            if name in fires:
                pg = ev.paired(fires[name], base_f, VAL_YEARS)
                if pg:
                    gains.append((name, delta, pg["diff"]))
    gains.sort(key=lambda x: -x[2])
    helpful = [(n, dl) for n, dl, g in gains if g > 0]
    if run_combos and len(helpful) >= 2:
        print(f"\n  [2] STAGE 2 - combinations of options that helped V81 in "
              f"the VALIDATION years")
        print("      " + ", ".join(f"{n} ({g:+.2f})" for n, _, g in gains
                                   if g > 0))
        seen = {json.dumps(BASE, sort_keys=True)}
        combos = []
        for k in (2, 3):
            if len(helpful) >= k:
                combos.append((f"combo top-{k}", helpful[:k]))
        if len(helpful) > 3:
            combos.append(("combo all helpful", helpful))
        for cname, items in combos:
            c, used, skipped = apply_deltas(items)
            sig = json.dumps(c, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            label = f"{cname}: " + " + ".join(u.split(" ", 1)[0]
                                               for u in used)
            if skipped:
                label += f" (skipped {', '.join(s.split(' ', 1)[0] for s in skipped)})"
            run_variant(label, c)
    elif run_combos:
        print(f"\n  [2] STAGE 2 skipped - fewer than two options improved on "
              f"V81 in the validation years")

    # blend of the best validation model too
    model_names = [n for n in runs]
    best_model = None
    if model_names:
        sc = {n: ev.paired(fires[n], opp, VAL_YEARS) for n in model_names}
        sc = {n: v["diff"] for n, v in sc.items() if v}
        if sc:
            best_model = max(sc, key=sc.get)
            if best_model != "V81 base":
                _, _, refs = runs[best_model]
                fires[f"3 blend [{best_model}] + bb_position"] = blend_fire(
                    d, refs, good_bb)

    # ---- tables ---------------------------------------------------------------
    arms = [a for a in fires]
    model_like = [a for a in arms if a not in rules]
    K = max(len(model_like), 1)
    rows = []
    for a in arms:
        row = {"Arm": a, "Kind": "rule" if a in rules else "model"}
        for tag, yrs in (("val", VAL_YEARS), ("test", TEST_YEARS),
                         ("held", HELD_YEARS)):
            s = ev.summary(fires[a], yrs)
            if s:
                row[f"{tag} lift"] = s["lift"]
                if tag == "held":
                    row.update({"held signals": s["signals"],
                                "held lo": s["lo"], "held hi": s["hi"],
                                "held ExpR": s["expR"],
                                "held cond yrs": s["cond"],
                                "held worst yr": s["worst"],
                                "max signals/yr": s["max_year"]})
            if a != OPPONENT:
                p = ev.paired(fires[a], opp, yrs, k_tests=K if tag == "held"
                              else 1)
                if p:
                    row[f"{tag} vs bb"] = p["diff"]
                    row[f"{tag} vs bb lo"] = p["lo"]
                    row[f"{tag} vs bb hi"] = p["hi"]
                    if tag == "held":
                        row["held vs bb lo (Bonf)"] = p["lo_bonf"]
                        row["held vs bb hi (Bonf)"] = p["hi_bonf"]
        if base_f is not None and a not in rules and a != "V81 base":
            p = ev.paired(fires[a], base_f, VAL_YEARS)
            if p:
                row["val vs V81"] = p["diff"]
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(f"{out_dir}/all_arms.csv", index=False)

    fmt = lambda v: f"{v:+.2f}" if isinstance(v, (float, np.floating)) else v
    print("\n" + "=" * 104)
    print("  [3] EVERY ARM - lift = precision above its own matched pools (pp); "
          "'vs bb' = paired difference")
    print("=" * 104)
    show = ["Arm", "val lift", "val vs V81", "val vs bb", "test lift",
            "test vs bb", "held lift", "held vs bb", "held ExpR",
            "held cond yrs", "held worst yr", "max signals/yr"]
    print(T[[c for c in show if c in T.columns]].to_string(
        index=False, float_format=lambda v: f"{v:+.2f}"))

    # ---- selection on validation, confirmation on test ----------------------
    M = T[T["Kind"] == "model"].dropna(subset=["val vs bb"])
    print("\n" + "=" * 104)
    print("  [4] THE VERDICT - chosen on 2017-2021, confirmed on 2022-2026")
    print("=" * 104)
    if M.empty:
        print("  no model arm produced a validation result")
        return T
    sel = M.loc[M["val vs bb"].idxmax()]
    print(f"  chosen on validation: {sel['Arm']}")
    print(f"    validation vs {OPPONENT}: {sel['val vs bb']:+.2f} pp "
          f"[{sel['val vs bb lo']:+.2f}, {sel['val vs bb hi']:+.2f}]  "
          f"(this is where it was picked - optimistic by construction)")
    print(f"    TEST       vs {OPPONENT}: {sel['test vs bb']:+.2f} pp "
          f"[{sel['test vs bb lo']:+.2f}, {sel['test vs bb hi']:+.2f}]  "
          f"<- the claim")
    print(f"    full decade vs {OPPONENT}: {sel['held vs bb']:+.2f} pp, "
          f"90% [{sel['held vs bb lo']:+.2f}, {sel['held vs bb hi']:+.2f}], "
          f"Bonferroni over {K} arms [{sel['held vs bb lo (Bonf)']:+.2f}, "
          f"{sel['held vs bb hi (Bonf)']:+.2f}]")
    if sel["test vs bb lo"] > 0:
        verdict = (f"it BEATS the simple rule on years it was not chosen on: "
                   f"{sel['test vs bb']:+.2f} pp with the whole interval above "
                   f"zero.")
    elif sel["test vs bb"] > 0:
        verdict = (f"it is ahead on the test years ({sel['test vs bb']:+.2f} "
                   f"pp) but the interval includes zero, so report it as "
                   f"matching the rule with a positive point estimate.")
    else:
        verdict = (f"it does NOT hold its validation lead on the test years "
                   f"({sel['test vs bb']:+.2f} pp), so the validation win was "
                   f"selection luck. Report the model as matching the rule.")
    print(f"\n  VERDICT for '{sel['Arm']}': {verdict}")
    base_row = T[T["Arm"] == "V81 base"]
    if len(base_row):
        br = base_row.iloc[0]
        print(f"\n  for reference, V81 base vs {OPPONENT}: validation "
              f"{br.get('val vs bb', np.nan):+.2f}, test "
              f"{br.get('test vs bb', np.nan):+.2f}, decade "
              f"{br.get('held vs bb', np.nan):+.2f} pp "
              f"[{br.get('held vs bb lo', np.nan):+.2f}, "
              f"{br.get('held vs bb hi', np.nan):+.2f}]")

    # ---- seed check on the winner -------------------------------------------
    if run_seed_check:
        name = sel["Arm"]
        blend = name.startswith("3 blend")
        src = best_model if blend and name != "3 blend V81 + bb_position" \
            else ("V81 base" if blend else name)
        if src in runs:
            c0 = runs[src][0]
            print(f"\n  [5] SEED CHECK on {name}")
            rows = [("seed +0", fires[name])]
            for off in (100, 200):
                c1 = dict(c0, seed_offset=off)
                f = run_variant(f"{src} [seed +{off}]", c1)
                if f is None:
                    continue
                if blend:
                    f = blend_fire(d, runs[f"{src} [seed +{off}]"][2], good_bb)
                rows.append((f"seed +{off}", f))
            sr = []
            for lab, f in rows:
                p = ev.paired(f, opp, HELD_YEARS)
                pt = ev.paired(f, opp, TEST_YEARS)
                sr.append({"Seed": lab, "decade vs bb": p["diff"] if p
                           else np.nan,
                           "test vs bb": pt["diff"] if pt else np.nan})
            sr = pd.DataFrame(sr)
            print(sr.to_string(index=False,
                               float_format=lambda v: f"{v:+.2f}"))
            sr.to_csv(f"{out_dir}/seed_check.csv", index=False)
            spread = sr["decade vs bb"].max() - sr["decade vs bb"].min()
            print(f"    spread across seeds: {spread:.2f} pp - a lead smaller "
                  f"than this is not a lead")

    print(f"\n  wrote {out_dir}/all_arms.csv | total "
          f"{(time.time() - t0) / 60:.1f} min")
    return T


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN    = "entry_model_v81.joblib"
    RUN_PRICE_CACHE = None          # None uses the cache V81 was trained on
    RUN_BIG_CACHE   = None          # e.g. "price_cache_v68" for option 5 (slow)
    RUN_N_SEEDS     = 5             # 3 is ~40% faster; 5 matches V81
    RUN_N_BOOT      = 4000
    RUN_COMBOS      = True
    RUN_SEED_CHECK  = True
    RUN_RESUME      = True          # reuse cached variants after a crash
    RUN_ONLY        = None          # or a list of option names to run
    RUN_OUT_DIR     = "thesis_tables_v85"
    # -------------------------------------------------------------------------

    run_v85(model_in=RUN_MODEL_IN, price_cache=RUN_PRICE_CACHE,
            big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS, n_boot=RUN_N_BOOT,
            run_combos=RUN_COMBOS, run_seed_check=RUN_SEED_CHECK,
            resume=RUN_RESUME, only=RUN_ONLY, out_dir=RUN_OUT_DIR)
