"""
V80 - A MODEL ON ONLY THE WINNING FEATURES, AND WHERE THE EFFECT LIVES.

TWO QUESTIONS
-------------
1. V79 found six single features that each match or beat the 28-feature model.
   Does a model restricted to those features beat the best single feature? If
   it does, combining them adds something. If not, the single feature is the
   answer and no model is needed.

2. How does the effect behave in normal and bull markets, as opposed to crisis
   years?

THE TRAP IN QUESTION 1, AND HOW THIS FILE AVOIDS IT
---------------------------------------------------
The six features were chosen BECAUSE they won on the 2008-2026 out-of-sample
record. Training a model on them and scoring it on that same record lets the
test set pick the features - the post-hoc selection trap this project hit
earlier with signal direction. The result would be optimistic by construction,
and an examiner would ask exactly that.

So features are selected on the EARLY period only (test years before
SPLIT_YEAR, purged by the holding window) and every arm is scored on the LATE
period, which played no part in the choice. That is what a researcher in 2017
could actually have done. The file also prints whether the early-selected set
matches the six V79 named: if it does, the post-hoc worry was moot; if not,
that difference is itself the finding.

THE ARMS (late period, identical rule, identical matched control)
-----------------------------------------------------------------
  random score x3              the floor
  each selected single feature
  rank composite               equal-weight average of per-date cross-sectional
                               ranks of the selected features. No fitting at
                               all. In factor research a composite like this
                               often beats a fitted combination out of sample,
                               so it is the bar a restricted model has to clear.
  XGB on selected features     walk-forward, same settings as the deployed model
  XGB on selected, monotone    same, with each feature's direction FIXED by the
                               mean-reversion hypothesis (low bb_position ->
                               higher score). Theory-imposed regularisation,
                               not tuning.
  XGB on all 28                the deployed model

THE TRAP IN QUESTION 2, AND HOW THIS FILE AVOIDS IT
---------------------------------------------------
Dropping years AFTER seeing they went badly is cherry-picking, and it shows up
the moment anyone reads the per-year table. Two honest alternatives are used:

  a) REGIME AT ENTRY. Market breadth (share of the universe above its 200-day
     EMA) is known on the day a signal fires. Signals are split into bear /
     neutral / bull regimes by FIXED cutoffs, so the split uses no hindsight and
     could be used as a live gate in the product.

  b) A PRINCIPLED FOLD EXCLUSION. 2008 is the only fold whose training window
     (2005 to late 2007) contains no bear market: its model is being asked to
     act in a regime it has never seen, by construction. That is a reason
     defined by the TRAINING data, not by the test outcome, so excluding it is
     defensible - and both numbers are always printed side by side.

PER-SIGNAL EXCESS
-----------------
Regime splits need an additive measure. For each fired signal, the expected
outcome of the matched control is the mean outcome of its own ticker-month pool,
so excess = outcome - pool mean is exactly the matched-control lift per signal
and averages cleanly within any subgroup. Intervals come from resampling
calendar MONTHS, because signals in one month share market conditions and are
not independent.
"""

import os
import sys
from math import comb

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76
from falsify_mechanism_v79 import MatchedControl

SEED = 80
OUT_DIR = "thesis_tables_v80"
SPLIT_YEAR = 2017                 # select features before, evaluate from here
TOP_K = 6
MIN_SELECT_SIGNALS = 300          # an early-period winner on fewer is noise

# V79's six, for comparison with the honest early selection.
V79_SIX = [("bb_position", -1), ("rsi_14", -1), ("ret_5", -1),
           ("px_vs_ema20", -1), ("ret_20", -1), ("mkt_ret_20", -1)]

# Folds whose TRAINING window holds no bear market. Defined by training data,
# never by test outcome. 2008's model trains on 2005 to late 2007.
OUT_OF_REGIME_FOLDS = [2008]

BREADTH_CUTS = (0.40, 0.60)       # fixed: bear < 0.40 <= neutral <= 0.60 < bull


# =============================================================================
# DATA SANITY - the regime split is meaningless on prices that do not behave
# like a real market, so it is checked before anything is reported
# =============================================================================
CRASH_LOWS = [("2008-11-20", "GFC low"), ("2009-03-09", "GFC final low"),
              ("2020-03-23", "COVID low"), ("2022-10-12", "2022 bear low")]


def market_sanity(d):
    """
    Breadth and the median gap to the 200-day EMA on dates every real US
    equity universe agrees were deep lows.

    Written after the 62-name development cache was found to read breadth 0.40
    and a median gap of -1.8% on 2020-03-23, when real stocks were typically
    25-40% below their 200-day EMA. Every result produced on that cache tested
    MECHANICS, never effect sizes. This makes the same failure impossible to
    miss on any other cache.
    """
    per = (d.groupby("date")
             .agg(breadth=("mkt_breadth_ema200", "first"),
                  med_gap=("px_vs_ema200", "median"))
             .sort_index())
    rows, failed = [], 0
    for ds, name in CRASH_LOWS:
        t = pd.Timestamp(ds)
        if t < per.index.min() or t > per.index.max():
            continue
        i = per.index.get_indexer([t], method="nearest")[0]
        if abs((per.index[i] - t).days) > 10:
            continue
        b, g = float(per["breadth"].iloc[i]), float(per["med_gap"].iloc[i])
        ok = b < 0.30 and g < -0.08
        failed += (not ok)
        rows.append({"Date": per.index[i].date(), "Event": name,
                     "Breadth": b, "Median gap to EMA200": g,
                     "Looks real": ok})
    lo, hi = BREADTH_CUTS
    b = per["breadth"]
    share = {"bear": float((b < lo).mean()),
             "neutral": float(((b >= lo) & (b <= hi)).mean()),
             "bull": float((b > hi).mean())}
    return pd.DataFrame(rows), failed, share, b.describe()


# =============================================================================
# HELPERS
# =============================================================================
def _fire(frame, score, quantile, window_days):
    t = frame.assign(p=np.asarray(score, float))
    m, _ = M76.fire_rolling(t, quantile, window_days, spaced=False)
    return m


def _pool_means(frame):
    di = pd.DatetimeIndex(frame["date"])
    key = frame["ticker"].astype(str) + "|" + \
        pd.PeriodIndex(di, freq="M").astype(str)
    g = frame.groupby(key)
    return (g["label"].transform("mean").to_numpy(float),
            g["r_multiple"].transform("mean").to_numpy(float),
            key.to_numpy())


def excess_frame(frame, fire):
    """Per fired signal: outcome minus its own ticker-month pool mean."""
    pl, pr, key = _pool_means(frame)
    f = np.asarray(fire, bool)
    out = pd.DataFrame({
        "year": frame["year"].to_numpy()[f],
        "month": pd.PeriodIndex(pd.DatetimeIndex(frame["date"]),
                                freq="M").to_numpy()[f],
        "breadth": frame["mkt_breadth_ema200"].to_numpy(float)[f],
        "ex_win": frame["label"].to_numpy(float)[f] - pl[f],
        "ex_R": frame["r_multiple"].to_numpy(float)[f] - pr[f]})
    lo, hi = BREADTH_CUTS
    out["regime"] = np.where(out["breadth"] < lo, "bear",
                             np.where(out["breadth"] > hi, "bull", "neutral"))
    return out


def month_boot(ex, col, n_boot=300, seed=SEED):
    """90% interval on the mean, resampling calendar months."""
    if len(ex) < 30:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    by = {m: g[col].to_numpy(float) for m, g in ex.groupby("month")}
    months = list(by)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(months), len(months), replace=True)
        v = np.concatenate([by[months[i]] for i in pick])
        stats.append(v.mean())
    return float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def sign_test(frame, fire, years, floor=15):
    f = frame[np.asarray(fire, bool)]
    rows = []
    for y in years:
        fy = f[f["year"] == y]
        if len(fy) >= floor:
            g = frame[frame["year"] == y]
            rows.append(float(fy["label"].mean()) > float(g["label"].mean()))
    n = len(rows)
    if n < 3:
        return np.nan, 0, n
    k = int(sum(rows))
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, k, n


def composite_score(frame, feats_dirs):
    """Equal-weight mean of signed per-date cross-sectional percentile ranks."""
    parts = []
    for f, sgn in feats_dirs:
        rk = frame.groupby("date")[f].rank(pct=True).to_numpy(float)
        parts.append(rk if sgn > 0 else 1.0 - rk)
    return np.nanmean(np.vstack(parts), axis=0)


def walk_forward_with(d, features, provenance, monotone=None):
    """E64.walk_forward, optionally with monotone constraints, params restored."""
    old = E64.XGB_PARAMS
    try:
        if monotone is not None:
            E64.XGB_PARAMS = {**old, "monotone_constraints": tuple(monotone)}
        return E64.walk_forward(d, features, provenance["min_train_years"],
                                provenance["n_seeds"], provenance["seed"],
                                provenance["horizon"],
                                verbose=False).reset_index(drop=True)
    finally:
        E64.XGB_PARAMS = old


# =============================================================================
# EVALUATE ONE ARM ON A PERIOD
# =============================================================================
def evaluate(frame, fire, label, kind, years, n_draws, seed,
             exclude_years=()):
    """Matched-control lift on `years` only, optionally minus excluded folds."""
    yrs = [y for y in years if y not in exclude_years]
    sub = frame["year"].isin(yrs).to_numpy()
    fr = frame[sub].reset_index(drop=True)
    fi = np.asarray(fire, bool)[sub]
    pos = np.flatnonzero(fi)
    if len(pos) < 50:
        return None, None
    mc = MatchedControl(fr)
    c = mc.compare(pos, n_draws, seed)
    p_sign, k, n = sign_test(fr, fi, yrs)
    ex = excess_frame(fr, fi)
    lo, hi = month_boot(ex, "ex_win", seed=seed)
    row = {"Arm": label, "Kind": kind, "Signals": int(len(pos)),
           "Prec lift pp": c["precision"]["lift"] * 100,
           "90% CI lo": lo * 100 if np.isfinite(lo) else np.nan,
           "90% CI hi": hi * 100 if np.isfinite(hi) else np.nan,
           "ExpR lift": c["expectancy_R"]["lift"],
           "Stop-out pp": c["loss_share"]["lift"] * 100,
           "Beat": f"{c['precision']['beat']}/{c['precision']['n']}",
           "Sign p": p_sign, "Sign yrs": f"{k}/{n}" if n else ""}
    return row, ex


# =============================================================================
# RUNNER
# =============================================================================
def run_v80(model_file="entry_model_v76.joblib", price_cache=None,
            out_dir=OUT_DIR, split_year=SPLIT_YEAR, top_k=TOP_K,
            feature_source="early_selection", n_draws=200, n_random=3,
            run_monotone=True, seed=SEED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model = M76.load_model(model_file)
    prov, rule = model["provenance"], model["rule"]
    price_cache = price_cache or prov["price_cache"]
    q, win = rule["quantile"], rule["window_days"]
    all_feats = model["features"]

    print("=" * 100)
    print("V80 - RESTRICTED MODEL, AND WHERE THE EFFECT LIVES")
    print("=" * 100)
    print(f"  select on test years < {split_year}, evaluate on >= {split_year}"
          f" | feature source: {feature_source} | top {top_k}")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, prov["horizon"], prov["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year

    print(f"\n  [0] DATA SANITY - does this cache behave like a real market?")
    sane, failed, share, bdesc = market_sanity(d)
    if len(sane):
        print(sane.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  breadth across dates: {bdesc['min']:.2f} .. {bdesc['max']:.2f} "
          f"(sd {bdesc['std']:.3f}); regime share of dates: "
          f"bear {share['bear']:.0%}, neutral {share['neutral']:.0%}, "
          f"bull {share['bull']:.0%}")
    regime_ok = failed == 0 and len(sane) > 0 and bdesc["std"] > 0.10
    if not regime_ok:
        print("  *** THIS CACHE DOES NOT BEHAVE LIKE A REAL MARKET. On a deep "
              "crash low real breadth")
        print("  *** falls below ~0.15 and the median stock sits 20-40% under "
              "its 200-day EMA. The")
        print("  *** REGIME and PER-YEAR sections below are printed for "
              "mechanics only and must not")
        print("  *** be read or quoted. Effect sizes from this cache are not "
              "evidence of anything.")
    else:
        print("  Crash lows read as crash lows. The regime split is "
              "meaningful on this cache.")

    print(f"\n  walk-forward: deployed model (all {len(all_feats)} features)")
    te = walk_forward_with(d, all_feats, prov)
    years = sorted(te["year"].unique())
    early = [y for y in years if y < split_year]
    late = [y for y in years if y >= split_year]
    print(f"  {len(te):,} OOS trades | early {early[0]}-{early[-1]} "
          f"({len(early)}y) | late {late[0]}-{late[-1]} ({len(late)}y)")

    # ---- 1. SELECTION on the early period only ---------------------------
    n_bars = E64.HORIZON_CONFIGS[prov["horizon"]]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    cutoff = pd.Timestamp(f"{split_year}-01-01") - embargo
    early_mask = (pd.DatetimeIndex(te["date"]) < cutoff)
    print(f"\n  [1] SELECTION on early signals dated before {cutoff.date()} "
          f"(embargo {embargo.days}d)")
    sel_rows = []
    for f in all_feats:
        best = None
        for sgn in (-1, 1):
            fire = _fire(te, sgn * te[f].to_numpy(float), q, win)
            fe = fire & early_mask
            if fe.sum() < MIN_SELECT_SIGNALS:
                continue
            ex = excess_frame(te, fe)
            v = float(ex["ex_win"].mean() * 100)
            if best is None or v > best[2]:
                best = (f, sgn, v, int(fe.sum()))
        if best:
            sel_rows.append({"Feature": best[0],
                             "Direction": "low" if best[1] < 0 else "high",
                             "sgn": best[1], "Early lift pp": best[2],
                             "Early signals": best[3]})
    sel = pd.DataFrame(sel_rows).sort_values("Early lift pp",
                                             ascending=False)
    sel.drop(columns="sgn").to_csv(f"{out_dir}/selection_early.csv",
                                   index=False)
    print(sel.drop(columns="sgn").head(12).to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    early_pick = [(r.Feature, r.sgn) for r in sel.head(top_k).itertuples()]
    v79 = set(f for f, _ in V79_SIX)
    overlap = set(f for f, _ in early_pick) & v79
    print(f"\n  early-selected top {top_k}: "
          + ", ".join(f"{'low' if s < 0 else 'high'} {f}"
                      for f, s in early_pick))
    print(f"  overlap with V79's six (chosen on ALL years): "
          f"{len(overlap)}/{top_k}  {sorted(overlap)}")
    if feature_source == "v79_six":
        picks = [x for x in V79_SIX if x[0] in te.columns][:top_k]
        print(f"  USING V79's SIX. Their selection saw the late period, so "
              f"every late-period number")
        print(f"  below is OPTIMISTIC. Use only to compare with the honest "
              f"early selection.")
    else:
        picks = early_pick

    # ---- 2. ARMS, scored on the late period ------------------------------
    print(f"\n  [2] ARMS on the late period ({late[0]}-{late[-1]}), "
          f"{n_draws} matched draws each")
    arms = []          # (label, kind, frame, fire)

    for i in range(n_random):
        rg = np.random.default_rng(2000 + i)
        arms.append((f"random #{i+1}", "floor", te,
                     _fire(te, rg.random(len(te)), q, win)))

    for f, sgn in picks:
        arms.append((f"{'low' if sgn < 0 else 'high'} {f}", "single", te,
                     _fire(te, sgn * te[f].to_numpy(float), q, win)))

    arms.append((f"rank composite ({len(picks)} feats)", "composite", te,
                 _fire(te, composite_score(te, picks), q, win)))

    sub_feats = [f for f, _ in picks]
    print(f"    walk-forward: XGB on {len(sub_feats)} selected features ...")
    te_k = walk_forward_with(d, sub_feats, prov)
    arms.append((f"XGB on {len(sub_feats)} selected", "xgb_k", te_k,
                 _fire(te_k, te_k["p"], q, win)))

    if run_monotone:
        print(f"    walk-forward: XGB on selected, monotone ...")
        mono = [int(s) for _, s in picks]    # low-is-good -> -1
        te_m = walk_forward_with(d, sub_feats, prov, monotone=mono)
        arms.append((f"XGB on {len(sub_feats)} selected, monotone",
                     "xgb_mono", te_m, _fire(te_m, te_m["p"], q, win)))

    arms.append(("XGB on all 28 (deployed)", "xgb_all", te,
                 _fire(te, te["p"], q, win)))

    results, excess = [], {}
    for scope, excl in (("late", ()), ("late excl. out-of-regime", tuple(
            y for y in OUT_OF_REGIME_FOLDS if y in late))):
        if scope != "late" and not excl:
            continue
        for label, kind, fr, fire in arms:
            row, ex = evaluate(fr, fire, label, kind, late, n_draws, seed,
                               exclude_years=excl)
            if row:
                row["Scope"] = scope
                results.append(row)
                if scope == "late":
                    excess[label] = ex

    # full-period rows for the regime question (no selection involved for
    # these three, so the full record is legitimate for them)
    full_rows, full_ex = [], {}
    for label, kind, fr, fire in arms:
        if kind not in ("xgb_all", "composite", "single"):
            continue
        for scope, excl in (("all years", ()),
                            ("excl. out-of-regime", tuple(OUT_OF_REGIME_FOLDS))):
            row, ex = evaluate(fr, fire, label, kind, years, n_draws, seed,
                               exclude_years=excl)
            if row:
                row["Scope"] = scope
                full_rows.append(row)
                if scope == "all years":
                    full_ex[label] = ex

    res = pd.DataFrame(results)
    res.to_csv(f"{out_dir}/arms_late.csv", index=False)
    cols = ["Arm", "Signals", "Prec lift pp", "90% CI lo", "90% CI hi",
            "ExpR lift", "Stop-out pp", "Beat", "Sign p", "Sign yrs"]
    print("\n" + "=" * 100)
    print(f"  LATE PERIOD {late[0]}-{late[-1]} - features chosen without "
          f"seeing it")
    print("=" * 100)
    print(res[res["Scope"] == "late"][cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 3. REGIME ---------------------------------------------------------
    print("\n" + "=" * 100)
    if not regime_ok:
        print("  [3] REGIME - *** NOT MEANINGFUL ON THIS CACHE, see [0] *** "
              "(printed for mechanics)")
    print(f"  [3] REGIME AT ENTRY - market breadth, fixed cutoffs "
          f"(bear < {BREADTH_CUTS[0]:.2f} < neutral < {BREADTH_CUTS[1]:.2f} "
          f"< bull)")
    print("      All years. Excess = outcome minus the signal's own "
          "ticker-month pool mean.")
    print("=" * 100)
    reg_rows = []
    key_arms = [a for a in full_ex if a.startswith("XGB on all")] + \
        [a for a in full_ex if a.startswith("rank composite")] + \
        [a for a in full_ex if a.startswith("low ") or a.startswith("high ")]
    for a in key_arms[:4]:
        ex = full_ex[a]
        for rg in ("bear", "neutral", "bull"):
            e = ex[ex["regime"] == rg]
            lo, hi = month_boot(e, "ex_win", seed=seed)
            reg_rows.append({"Arm": a, "Regime": rg, "Signals": len(e),
                             "Share": len(e) / max(len(ex), 1),
                             "Excess win pp": e["ex_win"].mean() * 100
                             if len(e) else np.nan,
                             "90% lo": lo * 100, "90% hi": hi * 100,
                             "Excess R": e["ex_R"].mean() if len(e)
                             else np.nan})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(f"{out_dir}/regime.csv", index=False)
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 4. PER YEAR -------------------------------------------------------
    print("\n" + "=" * 100)
    print("  [4] PER YEAR - excess over the matched pool, with each year's "
          "unconditional win rate")
    print("=" * 100)
    blind = te.groupby("year")["label"].mean()
    yr_rows = []
    for a in key_arms[:2]:
        ex = full_ex[a]
        for y, e in ex.groupby("year"):
            yr_rows.append({"Arm": a, "Year": int(y), "Signals": len(e),
                            "Excess win pp": e["ex_win"].mean() * 100,
                            "Excess R": e["ex_R"].mean(),
                            "Blind win": float(blind.get(y, np.nan)),
                            "Out-of-regime": y in OUT_OF_REGIME_FOLDS})
    yrs_df = pd.DataFrame(yr_rows)
    yrs_df.to_csv(f"{out_dir}/per_year.csv", index=False)
    print(yrs_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # correlation of the effect with how hard the year was
    for a in key_arms[:2]:
        s = yrs_df[(yrs_df["Arm"] == a) & (yrs_df["Signals"] >= 30)]
        if len(s) >= 5:
            rho = s["Blind win"].corr(s["Excess win pp"], method="spearman")
            print(f"\n  {a}: Spearman(year's blind win rate, excess) = "
                  f"{rho:+.3f} over {len(s)} years with >= 30 signals")
    print("  NEGATIVE means the effect is LARGER in hard years and SMALLER in "
          "easy (bull) years -")
    print("  the known profile of mean reversion, which momentum dominates in "
          "steady uptrends.")

    # ---- 5. VERDICT --------------------------------------------------------
    L = res[res["Scope"] == "late"].set_index("Arm")
    print("\n" + "=" * 100)
    print("  VERDICT (late period, features chosen without seeing it)")
    print("=" * 100)
    fl = L[L["Kind"] == "floor"]["Prec lift pp"]
    if len(fl):
        print(f"  floor (random):        {fl.min():+.2f} .. {fl.max():+.2f} pp")
    singles = L[L["Kind"] == "single"]
    if len(singles):
        b = singles["Prec lift pp"].idxmax()
        print(f"  best single feature:   {singles.loc[b, 'Prec lift pp']:+.2f}"
              f" pp  ({b})")
    for kind, name in (("composite", "rank composite"),
                       ("xgb_k", "XGB restricted"),
                       ("xgb_mono", "XGB restricted, monotone"),
                       ("xgb_all", "XGB all 28")):
        s = L[L["Kind"] == kind]
        if len(s):
            r = s.iloc[0]
            print(f"  {name:<22} {r['Prec lift pp']:+.2f} pp  "
                  f"[{r['90% CI lo']:+.2f}, {r['90% CI hi']:+.2f}]  "
                  f"ExpR {r['ExpR lift']:+.4f}  sign p {r['Sign p']:.3f}")
    if len(singles):
        best_single = float(singles["Prec lift pp"].max())
        for kind, name in (("composite", "The composite"),
                           ("xgb_k", "The restricted model"),
                           ("xgb_mono", "The monotone restricted model")):
            s = L[L["Kind"] == kind]
            if not len(s):
                continue
            v = float(s["Prec lift pp"].iloc[0])
            lo = float(s["90% CI lo"].iloc[0])
            verdict = ("BEATS" if v > best_single and lo > 0 else
                       "matches" if v >= 0.85 * best_single else
                       "LOSES TO")
            print(f"    {name} {verdict} the best single feature "
                  f"({v:+.2f} vs {best_single:+.2f} pp).")
        print(f"    Note the best single feature is a MAXIMUM over "
              f"{len(singles)} arms, so it is biased upward;")
        print(f"    a combination that merely matches it is doing well.")
    print(f"\n  wrote CSVs to {out_dir}/")
    return res, reg, yrs_df, sel


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE     = "entry_model_v76.joblib"
    RUN_PRICE_CACHE    = None           # None uses the model's training cache
    RUN_OUT_DIR        = "thesis_tables_v80"
    RUN_SPLIT_YEAR     = 2017           # select before, evaluate from here
    RUN_TOP_K          = 6
    RUN_FEATURE_SOURCE = "early_selection"   # or "v79_six" (optimistic - see doc)
    RUN_N_DRAWS        = 200
    RUN_N_RANDOM       = 3
    RUN_MONOTONE       = True           # one extra walk-forward
    RUN_SEED           = SEED
    # -------------------------------------------------------------------------

    run_v80(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
            out_dir=RUN_OUT_DIR, split_year=RUN_SPLIT_YEAR, top_k=RUN_TOP_K,
            feature_source=RUN_FEATURE_SOURCE, n_draws=RUN_N_DRAWS,
            n_random=RUN_N_RANDOM, run_monotone=RUN_MONOTONE, seed=RUN_SEED)
