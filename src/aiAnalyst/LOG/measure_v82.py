"""
V82 - MEASURE THE RULE THE WAY PRODUCTION RUNS IT.

WHAT V81's INSPECTOR FOUND
--------------------------
V81 fired on wildly different shares of candidates from year to year:

    2008   178    2009      0    2010  7,735    2011  4,057    2012    68
    2013 4,335    2014  3,914    2015  1,952    2018    469    2019    43

That is fold-to-fold score drift. The walk-forward fits a NEW model each year,
and an expectancy regressor's output level tracks the average R of its own
training period, so each January the scale jumps. The rolling threshold was
taken from the trailing window - scored by LAST year's model - and applied to
THIS year's model. When the new model sat higher it flooded; when lower it
starved (2009: zero). The composite, whose scores are ranks, fired 93-186 every
year; V76 had the same disease (2 in 2009, 966 in 2022).

WHY THIS IS A MEASUREMENT BUG, NOT A PRODUCT BUG
------------------------------------------------
In production, score() rates the WHOLE trailing window with the single serving
model and takes the threshold from those scores. One model, one scale - the
fold mixing never happens. So the served V81 was fine; its measurement was not
measuring it. This file rescores the trailing window with EACH FOLD'S OWN model,
which is exactly what the served rule does. The served boosters are unchanged.

The trailing rows are mostly that fold's training data, so their scores are
in-sample. Production has the same property - the serving model scores its own
history to set today's cut - so the measurement now matches what is deployed.

ALSO FIXED HERE
---------------
MARKET BREADTH. V64 computes breadth over each DATE'S CANDIDATES - a median of
6 of ~300 stocks - which is why it read exactly 0.000 on every crash low. Here
breadth is rebuilt from EVERY ticker's daily close against its own 200-day EMA,
over all names alive that day. It is used for the regime split only; V81 does
not take it as a feature. The regime answer to "how does it do in normal and
bull markets" is only trustworthy on this series.

THE COMPOSITE. V80 ranked each feature within each date - among ~6 names - so
the "simple baseline" was handicapped. Composite v2 maps each feature to its
percentile in the trailing 252-day pool of all candidates, recomputed monthly:
causal, scale-free, and independent of how many names share a date. The old
version is kept beside it so the difference is visible.

INTEGRITY CHECK
---------------
The V81 arm is also scored under the OLD rule. The walk-forward here uses the
same builder, seeds and folds as V81, so that row must reproduce V81's 24,900
signals exactly. If it does not, something is nondeterministic and nothing
below can be compared with V81's output.
"""

import os
import sys
import copy

import numpy as np
import pandas as pd
import joblib

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76
import class_ai_entry_model_v81 as M81
import restricted_model_v80 as V80

SEED = 81                      # same as V81, so the folds and models reproduce
OUT_DIR = "thesis_tables_v82"
MODEL_IN = "entry_model_v81.joblib"
MODEL_OUT = "entry_model_v82.joblib"
QUANTILE = M81.QUANTILE
WINDOW_DAYS = M81.WINDOW_DAYS
MIN_ROWS = M76.MIN_WINDOW_ROWS
SPLIT_YEAR = M81.SPLIT_YEAR
BREADTH_CUTS = (0.40, 0.60)



def _win():
    return pd.Timedelta(days=int(WINDOW_DAYS * 365.25 / 252))


# =============================================================================
# 1. FULL-UNIVERSE MARKET BREADTH
# =============================================================================
def full_breadth(price_cache, names, min_names=50):
    """Share of ALL live tickers closing above their own 200-day EMA, daily."""
    cols = {}
    for t in names:
        df = E64.P2.load_prices(t, price_cache)
        if df is None or len(df) < 260:
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")]
        e = c.ewm(span=200, adjust=False).mean()
        a = (c > e).astype(float)
        a.iloc[:200] = np.nan                  # EMA not yet meaningful
        cols[t] = a
    W = pd.DataFrame(cols)
    n = W.notna().sum(axis=1)
    b = W.mean(axis=1, skipna=True)
    b[n < min_names] = np.nan
    return b, n


def breadth_sanity(b):
    checks = [("2008-11-20", "GFC low"), ("2009-03-09", "GFC final low"),
              ("2020-03-23", "COVID low"), ("2022-10-12", "2022 bear low"),
              ("2017-12-01", "2017 bull"), ("2021-06-15", "2021 bull")]
    rows = []
    for ds, name in checks:
        t = pd.Timestamp(ds)
        s = b.dropna()
        if s.empty or t < s.index.min() or t > s.index.max():
            continue
        i = s.index.get_indexer([t], method="nearest")[0]
        rows.append({"Date": s.index[i].date(), "Event": name,
                     "Full-universe breadth": float(s.iloc[i])})
    return pd.DataFrame(rows)


# =============================================================================
# 2. WALK-FORWARD THAT ALSO SCORES EACH FOLD'S TRAILING WINDOW
# =============================================================================
def walk_forward_ref(d, features, monotone, min_train_years, n_seeds, seed,
                     horizon):
    """
    As M81.walk_forward, plus: for each test year, the SAME fold models score
    every candidate in the trailing window before 1 January. Returns the test
    frame and {year: DataFrame(date, p)} covering trailing window + test year,
    all on that fold's scale.
    """
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    win = _win()
    d = d.sort_values("date").reset_index(drop=True)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    dd = pd.DatetimeIndex(d["date"])
    years = sorted(d["year"].unique())
    keep, preds, refs = [], [], {}
    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d[dd < start - embargo]
        te = d[d["year"] == y]
        if len(tr) < 2000 or len(te) < 200:
            continue
        ref = d[(dd >= start - win) & (dd < start)]
        ms = []
        for s in range(n_seeds):
            m, target, mode = M81.build_model(seed + s, monotone)
            m.fit(tr[features], tr[target], verbose=False)
            ms.append(m)
        p_te = M81.predict(ms, te[features], mode)
        p_ref = (M81.predict(ms, ref[features], mode) if len(ref)
                 else np.array([]))
        preds.append(p_te)
        keep.append(te)
        refs[y] = pd.DataFrame({
            "date": np.concatenate([ref["date"].to_numpy(),
                                    te["date"].to_numpy()]),
            "p": np.concatenate([p_ref, p_te])})
    out = pd.concat(keep, ignore_index=True)
    out["p"] = np.concatenate(preds)
    return out, refs


def fire_fold_consistent(te, refs, quantile=QUANTILE, min_rows=MIN_ROWS):
    """Monthly threshold from the trailing window scored by THIS year's model."""
    win = _win()
    fire = np.zeros(len(te), bool)
    di = pd.DatetimeIndex(te["date"])
    per = pd.PeriodIndex(di, freq="M")
    yrs = te["year"].to_numpy()
    p = te["p"].to_numpy(float)
    for y, ref in refs.items():
        rd = pd.DatetimeIndex(ref["date"]).values
        rp = ref["p"].to_numpy(float)
        o = np.argsort(rd, kind="mergesort")
        rd, rp = rd[o], rp[o]
        rows = np.flatnonzero(yrs == y)
        if not len(rows):
            continue
        pr = per[rows]
        for m in pd.unique(pr):
            t = m.to_timestamp()
            lo = np.searchsorted(rd, (t - win).to_datetime64(), "left")
            hi = np.searchsorted(rd, t.to_datetime64(), "left")
            w = rp[lo:hi]
            if w.size < min_rows:
                continue
            cut = float(np.quantile(w, 1.0 - quantile))
            sel = rows[pr == m]
            fire[sel] = p[sel] >= cut
    return fire


# =============================================================================
# 3. COMPOSITE v2 - trailing pooled percentiles, not per-date ranks
# =============================================================================
def trailing_pct(frame, col, min_rows=MIN_ROWS):
    win = _win()
    di = pd.DatetimeIndex(frame["date"])
    v = frame[col].to_numpy(float)
    o = np.argsort(di.values, kind="mergesort")
    ds, vs = di.values[o], v[o]
    per = pd.PeriodIndex(di, freq="M")
    codes, uniq = pd.factorize(per)
    out = np.full(len(frame), np.nan)
    for k, m in enumerate(uniq):
        t = m.to_timestamp()
        lo = np.searchsorted(ds, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(ds, t.to_datetime64(), "left")
        w = vs[lo:hi]
        w = np.sort(w[np.isfinite(w)])
        if w.size < min_rows:
            continue
        idx = np.flatnonzero(codes == k)
        out[idx] = np.searchsorted(w, v[idx], side="right") / w.size
    return out


def composite_v2(frame, feats_dirs):
    parts = []
    for f, sgn in feats_dirs:
        pc = trailing_pct(frame, f)
        parts.append(pc if sgn > 0 else 1.0 - pc)
    return np.nanmean(np.vstack(parts), axis=0)


# =============================================================================
# 4. MEASUREMENT HELPERS
# =============================================================================
def excess_regime(frame, fire, breadth_col="breadth_full"):
    pl, pr, _ = V80._pool_means(frame)
    f = np.asarray(fire, bool)
    b = frame[breadth_col].to_numpy(float)[f]
    lo, hi = BREADTH_CUTS
    reg = np.where(~np.isfinite(b), "unknown",
                   np.where(b < lo, "bear", np.where(b > hi, "bull",
                                                     "neutral")))
    return pd.DataFrame({
        "year": frame["year"].to_numpy()[f],
        "month": pd.PeriodIndex(pd.DatetimeIndex(frame["date"]),
                                freq="M").to_numpy()[f],
        "regime": reg,
        "ex_win": frame["label"].to_numpy(float)[f] - pl[f],
        "ex_R": frame["r_multiple"].to_numpy(float)[f] - pr[f]})


def coverage(frame, fire):
    f = pd.Series(np.asarray(fire, bool), index=frame.index)
    n = frame.groupby("year").size()
    k = f.groupby(frame["year"]).sum()
    rate = (k / n).fillna(0)
    nz = k[k > 0]
    return k.astype(int), rate, {
        "total": int(k.sum()), "min": int(k.min()), "max": int(k.max()),
        "cv": float(k.std() / k.mean()) if k.mean() else np.nan,
        "zero_years": int((k == 0).sum()),
        "max_rate": float(rate.max()), "min_rate": float(rate.min())}


# =============================================================================
# RUNNER
# =============================================================================
def run_v82(model_in=MODEL_IN, model_out=MODEL_OUT, price_cache=None,
            out_dir=OUT_DIR, n_draws=200, compare_deployed=True,
            seed=SEED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    m81 = M81.load_model(model_in)
    prov = m81["provenance"]
    price_cache = price_cache or prov["price_cache"]
    feats, dirs = m81["features"], m81["directions"]

    print("=" * 100)
    print("V82 - MEASURE THE RULE THE WAY PRODUCTION RUNS IT")
    print("=" * 100)

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, prov["horizon"], prov["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year

    # ---- breadth ----------------------------------------------------------
    print("\n  [1] FULL-UNIVERSE BREADTH (every live ticker, daily)")
    b, nb = full_breadth(price_cache, names)
    san = breadth_sanity(b)
    print(san.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    s = b.dropna()
    lo, hi = BREADTH_CUTS
    print(f"  range {s.min():.2f}..{s.max():.2f}, median names counted "
          f"{int(nb[nb > 0].median())}; share of days: bear "
          f"{(s < lo).mean():.0%}, neutral {((s >= lo) & (s <= hi)).mean():.0%}"
          f", bull {(s > hi).mean():.0%}")
    print("  Expect crash lows well under 0.30 but NOT exactly 0 - some "
          "names (energy in 2022)")
    print("  stay above their EMA even in a bear market. Bull dates should "
          "read 0.65+.")

    # ---- walk-forwards ------------------------------------------------------
    print(f"\n  [2] WALK-FORWARD, V81 model ({len(feats)} features, monotone) "
          f"+ each fold's trailing window")
    te, refs = walk_forward_ref(d, feats, dirs, prov["min_train_years"],
                                prov["n_seeds"], prov["seed"],
                                prov["horizon"])
    dn = pd.DatetimeIndex(te["date"]).normalize()
    te["breadth_full"] = dn.map(b).to_numpy(float)

    te28, refs28 = None, None
    if compare_deployed:
        print("      and V76 (28 features, unconstrained) the same way ...")
        f28 = E64.resolve_features("no_confirm")
        te28, refs28 = walk_forward_ref(d, f28, None, prov["min_train_years"],
                                        prov["n_seeds"], prov["seed"],
                                        prov["horizon"])
        te28["breadth_full"] = pd.DatetimeIndex(
            te28["date"]).normalize().map(b).to_numpy(float)

    # ---- arms ---------------------------------------------------------------
    arms = [
        ("V81, fold-consistent cut", "v81_fixed", te,
         fire_fold_consistent(te, refs)),
        ("V81, OLD cut (check)", "v81_old", te,
         V80._fire(te, te["p"].to_numpy(float), QUANTILE, WINDOW_DAYS)),
        ("composite v2 (trailing pct)", "comp2", te,
         V80._fire(te, composite_v2(te, list(zip(feats, dirs))), QUANTILE,
                   WINDOW_DAYS)),
        ("composite v1 (per-date rank)", "comp1", te,
         V80._fire(te, V80.composite_score(te, list(zip(feats, dirs))),
                   QUANTILE, WINDOW_DAYS)),
    ]
    for f, sgn in zip(feats, dirs):
        arms.append((f"{'low' if sgn < 0 else 'high'} {f}", "single", te,
                     V80._fire(te, sgn * te[f].to_numpy(float), QUANTILE,
                               WINDOW_DAYS)))
    for i in range(3):
        arms.append((f"random #{i+1}", "floor", te,
                     V80._fire(te, np.random.default_rng(4000 + i).random(
                         len(te)), QUANTILE, WINDOW_DAYS)))
    if te28 is not None:
        arms.append(("V76, fold-consistent cut", "v76_fixed", te28,
                     fire_fold_consistent(te28, refs28)))

    # ---- coverage -----------------------------------------------------------
    print("\n  [3] COVERAGE - signals per year (the rule should fire on a "
          "steady ~1-2%)")
    cov_rows, counts = [], {}
    for label, kind, fr, fire in arms:
        k, rate, st = coverage(fr, fire)
        counts[label] = k
        cov_rows.append({"Arm": label, "Total": st["total"], "Min yr":
                         st["min"], "Max yr": st["max"], "CV": st["cv"],
                         "Zero years": st["zero_years"],
                         "Rate range": f"{st['min_rate']:.2%} .. "
                                       f"{st['max_rate']:.2%}"})
    cov = pd.DataFrame(cov_rows)
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    pyc = pd.DataFrame(counts).fillna(0).astype(int)
    pyc.to_csv(f"{out_dir}/signals_per_year.csv")
    show = [c for c in ("V81, OLD cut (check)", "V81, fold-consistent cut",
                        "composite v2 (trailing pct)",
                        "V76, fold-consistent cut") if c in pyc.columns]
    print("\n" + pyc[show].to_string())

    old = int(cov.set_index("Arm").loc["V81, OLD cut (check)", "Total"])
    py81 = pd.DataFrame(m81["measured"].get("per_year", []))
    ref_total = (int(py81.loc[py81["Arm"].str.startswith("v81"),
                              "Signals"].sum()) if len(py81) else None)
    if ref_total is None:
        print(f"\n  INTEGRITY  no per-year record in {model_in} to compare "
              f"against (old cut fires {old:,})")
    else:
        same = old == ref_total
        print(f"\n  INTEGRITY  V81 under the old cut fires {old:,} here; "
              f"{model_in} recorded {ref_total:,}: "
              f"{'REPRODUCED' if same else '*** DIFFERENT - stop ***'}")
        if not same:
            print("  The folds, seeds or data differ from V81's run, so V82's "
                  "rows cannot be compared")
            print("  with V81's output. Check RUN_PRICE_CACHE and that "
                  "nothing in V64 changed.")

    # ---- matched-control tables ---------------------------------------------
    years = sorted(te["year"].unique())
    late = [y for y in years if y >= SPLIT_YEAR]
    tables, excess = {}, {}
    for scope, yrs in (("held-out 2017+", late), ("all years", years)):
        rows = []
        for label, kind, fr, fire in arms:
            if kind == "v81_old":
                continue
            r, ex = V80.evaluate(fr, fire, label, kind, yrs, n_draws, seed)
            if r is None:
                continue
            pc, kc, nc, worst, sd = M81.conditional_sign_test(ex)
            r.update({"Cond sign p": pc, "Cond yrs": f"{kc}/{nc}",
                      "Worst year pp": worst, "Year SD pp": sd})
            rows.append(r)
            if scope == "all years":
                excess[label] = excess_regime(fr, fire)
        tables[scope] = pd.DataFrame(rows)
        tables[scope].to_csv(f"{out_dir}/arms_{scope.split()[0]}.csv",
                             index=False)

    cols = ["Arm", "Signals", "Prec lift pp", "90% CI lo", "90% CI hi",
            "ExpR lift", "Cond yrs", "Cond sign p", "Sign yrs", "Sign p",
            "Worst year pp", "Year SD pp"]
    for scope in ("held-out 2017+", "all years"):
        print("\n" + "=" * 100)
        print(f"  [4] {scope.upper()} - matched control")
        print("=" * 100)
        t = tables[scope]
        print(t[[c for c in cols if c in t.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- regime -------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"  [5] REGIME AT ENTRY - FULL-UNIVERSE breadth (bear < {lo:.2f} < "
          f"neutral < {hi:.2f} < bull), all years")
    print("=" * 100)
    reg_rows = []
    single_lift = tables["all years"].set_index("Arm")["Prec lift pp"]
    best_single = (tables["all years"][tables["all years"]["Kind"] ==
                                       "single"]["Prec lift pp"].idxmax())
    best_single = tables["all years"].loc[best_single, "Arm"]
    for label in ("V81, fold-consistent cut", "composite v2 (trailing pct)",
                  best_single, "V76, fold-consistent cut"):
        ex = excess.get(label)
        if ex is None:
            continue
        for rg in ("bear", "neutral", "bull"):
            e = ex[ex["regime"] == rg]
            lo_, hi_ = V80.month_boot(e, "ex_win", seed=seed)
            reg_rows.append({"Arm": label, "Regime": rg, "Signals": len(e),
                             "Excess win pp": e["ex_win"].mean() * 100
                             if len(e) else np.nan,
                             "90% lo": lo_ * 100, "90% hi": hi_ * 100,
                             "Excess R": e["ex_R"].mean() if len(e)
                             else np.nan})
    regime = pd.DataFrame(reg_rows)
    regime.to_csv(f"{out_dir}/regime_full_breadth.csv", index=False)
    print(regime.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- per year, for the record --------------------------------------------
    yr_rows = []
    for label in ("V81, fold-consistent cut", "composite v2 (trailing pct)",
                  "V76, fold-consistent cut"):
        ex = excess.get(label)
        if ex is None:
            continue
        for y, e in ex.groupby("year"):
            yr_rows.append({"Arm": label, "Year": int(y), "Signals": len(e),
                            "Excess win pp": e["ex_win"].mean() * 100,
                            "Excess R": e["ex_R"].mean()})
    per_year = pd.DataFrame(yr_rows)
    per_year.to_csv(f"{out_dir}/per_year.csv", index=False)

    # ---- verdict ------------------------------------------------------------
    H = tables["held-out 2017+"].set_index("Arm")
    A = tables["all years"].set_index("Arm")
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    cv = cov.set_index("Arm")
    for a in ("V81, OLD cut (check)", "V81, fold-consistent cut"):
        if a in cv.index:
            print(f"  coverage {a:<30} CV {cv.loc[a, 'CV']:.2f}, "
                  f"{int(cv.loc[a, 'Zero years'])} zero years, "
                  f"{cv.loc[a, 'Rate range']}")
    print()
    for a in ("V81, fold-consistent cut", "composite v2 (trailing pct)",
              "composite v1 (per-date rank)", best_single,
              "V76, fold-consistent cut"):
        for nm, T in (("held-out", H), ("all yrs", A)):
            if a in T.index:
                r = T.loc[a]
                print(f"  {a:<30} {nm:<9} {r['Prec lift pp']:+6.2f} pp "
                      f"[{r['90% CI lo']:+.2f}, {r['90% CI hi']:+.2f}]  "
                      f"ExpR {r['ExpR lift']:+.3f}  cond {r['Cond yrs']}  "
                      f"worst {r['Worst year pp']:+.2f}")
    fl = A[A["Kind"] == "floor"]["Prec lift pp"]
    if len(fl):
        print(f"\n  floor (random, all years): {fl.min():+.2f} .. "
              f"{fl.max():+.2f} pp")

    # ---- save corrected measurement beside the unchanged served model -------
    m82 = copy.copy(m81)
    m82["measured"] = dict(m81["measured"])
    m82["measured"].update({
        "tables": {k: v.to_dict("records") for k, v in tables.items()},
        "regime": regime.to_dict("records"),
        "per_year": per_year.to_dict("records"),
        "coverage": cov.to_dict("records"),
        "breadth_source": "full universe, daily, close vs own EMA200"})
    m82["provenance"] = dict(m81["provenance"],
                             version="v82",
                             measurement="fold-consistent threshold: each "
                                         "fold's model scores its own "
                                         "trailing window, as score() does")
    joblib.dump(m82, model_out)
    print(f"\n  saved {model_out} - SAME served boosters as {model_in}, "
          f"corrected measurement")
    print(f"  wrote CSVs to {out_dir}/")
    return tables, regime, cov, per_year


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN         = "entry_model_v81.joblib"
    RUN_MODEL_OUT        = "entry_model_v82.joblib"
    RUN_PRICE_CACHE      = None    # None uses the cache V81 was trained on
    RUN_OUT_DIR          = "thesis_tables_v82"
    RUN_N_DRAWS          = 200
    RUN_COMPARE_DEPLOYED = True    # V76 under the fixed cut (extra walk-forward)
    RUN_SEED             = SEED
    # -------------------------------------------------------------------------

    run_v82(model_in=RUN_MODEL_IN, model_out=RUN_MODEL_OUT,
            price_cache=RUN_PRICE_CACHE, out_dir=RUN_OUT_DIR,
            n_draws=RUN_N_DRAWS, compare_deployed=RUN_COMPARE_DEPLOYED,
            seed=RUN_SEED)
