"""
V74 - IS THE SCORE ON ONE SCALE? (and is the tail result load-bearing)

THE CONTRADICTION THIS EXISTS TO RESOLVE
----------------------------------------
V73 on 337 names at the 1% cut produced two findings that cannot both be taken
at face value.

  1. Against random entry in the SAME NAME and MONTH, the model wins on every
     metric that matches its claim, at the p floor: precision .5353 vs .4518,
     expectancy +.5648 vs +.3329, R per bar +.02181 vs +.01346, loss share
     .3737 vs .4596 - 200/200 draws on each.

  2. Across score deciles, the relationship is INVERTED. Decile 1 (mean score
     -0.24) has the highest win rate (.4459) and the highest mean R (+.3613);
     decile 10 (mean score +0.65) has .4256 and +.2726. Spearman -0.661 on win
     rate, -0.479 on R, and the decile 1 -> 10 spread is NEGATIVE on both.

A score that is globally anti-correlated with the outcome, yet whose extreme
1% tail beats a matched control on every metric, is either a real tail effect or
an artifact. There is one artifact that produces exactly this shape, and it has
a fingerprint already visible in the V73 output.

THE SUSPECT: FOLD SCALE DRIFT
-----------------------------
Purged walk-forward fits a NEW model per fold. With PREDICT_MODE = "expectancy"
each fold regresses the R multiple under that period's volatility regime, so
each fold's scores have their own location and spread. Pooling every fold's
scores and cutting GLOBAL deciles then sorts partly by fold, not by conviction:
"decile 10" is whichever years scored high in absolute terms, and "decile 1" is
whichever years scored low. The decile table stops being a ranking check and
becomes a comparison between years - and the per-year blind win rate in V73
ranges from .2969 to .5061, so between-year differences are large enough to
swamp the within-year signal and flip its sign. Simpson's paradox.

The fingerprint is the signal count per year. Under a FIXED threshold the model
fired 661 times in 2022, 487 in 2014, 347 in 2012 - and 1, 4, 6, 7 and 11 times
in 2026, 2024, 2020, 2025 and 2021. Three years carry 59% of all signals. No
market story explains a 661-to-1 range; a drifting score against a fixed cut
does.

WHAT THIS FILE CHECKS
---------------------
  1. WITHIN-FOLD ranking. Build the decile table inside each test year and
     average, plus the distribution of per-year Spearmans. If the within-year
     relationship is positive while the pooled one is negative, the inversion
     is the artifact and the ranking is sound.
  2. RANK-NORMALISED score. Convert each fold's scores to within-fold
     percentiles, then re-cut. This is also the fix, not only the diagnostic: a
     relative rule ("top X% of scores in this window") is what the model should
     deploy if its scale drifts.
  3. SIGNALS PER YEAR under the absolute cut versus the relative cut.
  4. SIGN TEST with a minimum signal count per year. V73's 12/16 counted years
     holding 1, 4, 6, 7 and 11 trades as full draws, four of which it scored as
     wins. A year with one trade is a coin flip, not a year.
  5. YEAR-DROP robustness. Remove the biggest contributing years one at a time
     and see whether the tail precision survives.
"""

import os
import sys
import argparse
from math import comb

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v70 as M70

SEED = 74
OUT_DIR = "thesis_tables_v74"
MIN_SIGNALS = (0, 15, 30, 50, 100)
N_BINS = 10


def _decile_table(p, y, r, n_bins=N_BINS):
    q = pd.qcut(pd.Series(p), n_bins, labels=False, duplicates="drop")
    f = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float),
                      "r": np.asarray(r, float), "b": q})
    g = f.groupby("b").agg(n=("y", "size"), score=("p", "mean"),
                           win=("y", "mean"), R=("r", "mean"))
    return g.reset_index(drop=True)


def _mono(v):
    d = np.diff(np.asarray(v, float))
    return int((d > 0).sum()), int(len(d))


def _spearman(a, b):
    s = pd.Series(np.asarray(a, float)).corr(
        pd.Series(np.asarray(b, float)), method="spearman")
    return float(s) if np.isfinite(s) else np.nan


# =============================================================================
# 1 + 2  pooled vs within-fold vs rank-normalised
# =============================================================================
def scale_diagnostic(te, n_bins=N_BINS, min_rows=200):
    """Pooled deciles, deciles within each test year, and rank-normalised."""
    pooled = _decile_table(te["p"], te["label"], te["r_multiple"], n_bins)

    per_year, rows = [], []
    for y, g in te.groupby("year"):
        if len(g) < min_rows:
            continue
        t = _decile_table(g["p"], g["label"], g["r_multiple"], n_bins)
        if len(t) < 3:
            continue
        per_year.append(t)
        rows.append({"Year": int(y), "n": len(g),
                     "score min": float(g["p"].min()),
                     "score max": float(g["p"].max()),
                     "score mean": float(g["p"].mean()),
                     "blind win": float(g["label"].mean()),
                     "Spearman win": _spearman(t["score"], t["win"]),
                     "Spearman R": _spearman(t["score"], t["R"])})
    drift = pd.DataFrame(rows)

    within = None
    if per_year:
        stacked = pd.concat(per_year).groupby(level=0).mean(numeric_only=True)
        within = stacked.reset_index(drop=True)

    # rank-normalise inside each fold, then re-cut globally
    t2 = te.copy()
    t2["p_rank"] = t2.groupby("year")["p"].rank(pct=True)
    ranked = _decile_table(t2["p_rank"], t2["label"], t2["r_multiple"], n_bins)
    return {"pooled": pooled, "within": within, "ranked": ranked,
            "drift": drift, "te_ranked": t2}


# =============================================================================
# 3  signals per year, absolute cut vs relative cut
# =============================================================================
def signals_per_year(te, quantile):
    abs_cut = float(np.quantile(te["p"], 1.0 - quantile))
    fired_abs = te["p"] >= abs_cut
    rel = te.groupby("year")["p"].rank(pct=True)
    fired_rel = rel >= (1.0 - quantile)
    rows = []
    for y, g in te.groupby("year"):
        m = te["year"] == y
        a, r = te[m & fired_abs], te[m & fired_rel]
        rows.append({"Year": int(y), "Candidates": len(g),
                     "Fired (absolute)": len(a),
                     "Win rate (absolute)": (float(a["label"].mean())
                                             if len(a) else np.nan),
                     "Fired (relative)": len(r),
                     "Win rate (relative)": (float(r["label"].mean())
                                             if len(r) else np.nan),
                     "Blind win rate": float(g["label"].mean())})
    df = pd.DataFrame(rows)
    a, r = df["Fired (absolute)"], df["Fired (relative)"]
    return df, {"abs_min": int(a.min()), "abs_max": int(a.max()),
                "abs_cv": float(a.std() / a.mean()) if a.mean() else np.nan,
                "rel_min": int(r.min()), "rel_max": int(r.max()),
                "rel_cv": float(r.std() / r.mean()) if r.mean() else np.nan,
                "abs_top3_share": float(a.nlargest(3).sum() / a.sum())
                if a.sum() else np.nan}


# =============================================================================
# 4  sign test with a floor on signals per year
# =============================================================================
def sign_test_by_floor(te, quantile, floors=MIN_SIGNALS, relative=False):
    if relative:
        fired = te.groupby("year")["p"].rank(pct=True) >= (1.0 - quantile)
    else:
        fired = te["p"] >= float(np.quantile(te["p"], 1.0 - quantile))
    per = []
    for y, g in te.groupby("year"):
        f = te[(te["year"] == y) & fired]
        if not len(f):
            continue
        per.append({"Year": int(y), "Signals": len(f),
                    "Fired win": float(f["label"].mean()),
                    "Blind win": float(g["label"].mean()),
                    "Fired R": float(f["r_multiple"].mean()),
                    "Blind R": float(g["r_multiple"].mean())})
    p = pd.DataFrame(per)
    if p.empty:
        return None, None
    rows = []
    for fl in floors:
        s = p[p["Signals"] >= fl]
        if len(s) < 3:
            continue
        n = len(s)
        k = int((s["Fired win"] > s["Blind win"]).sum())
        kr = int((s["Fired R"] > s["Blind R"]).sum())
        pv = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        pr = sum(comb(n, i) for i in range(kr, n + 1)) / 2 ** n
        rows.append({"Min signals/year": fl, "Years kept": n,
                     "Signals kept": int(s["Signals"].sum()),
                     "Beat on win rate": f"{k}/{n}", "p (win)": pv,
                     "Beat on R": f"{kr}/{n}", "p (R)": pr})
    return p, pd.DataFrame(rows)


# =============================================================================
# 5  drop the biggest contributing years
# =============================================================================
def year_drop(te, quantile, n_drop=4):
    fired = te[te["p"] >= float(np.quantile(te["p"], 1.0 - quantile))]
    order = fired["year"].value_counts()
    rows, dropped = [], []
    for i in range(n_drop + 1):
        keep = fired[~fired["year"].isin(dropped)]
        base = te[~te["year"].isin(dropped)]
        if not len(keep):
            break
        rows.append({"Dropped": ", ".join(str(d) for d in dropped) or "none",
                     "Signals left": len(keep),
                     "Precision": float(keep["label"].mean()),
                     "Blind": float(base["label"].mean()),
                     "Lift": float(keep["label"].mean()
                                   - base["label"].mean()),
                     "Expectancy R": float(keep["r_multiple"].mean()),
                     "Blind R": float(base["r_multiple"].mean())})
        if i < n_drop and i < len(order):
            dropped.append(int(order.index[i]))
    return pd.DataFrame(rows)


# =============================================================================
# RUNNER
# =============================================================================
def run_v74(model_file="entry_model_v70.joblib", price_cache=None,
            out_dir=OUT_DIR, quantile=None, seed=SEED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model = M70.load_model(model_file)
    pr = model["provenance"]
    price_cache = price_cache or pr["price_cache"]
    q = quantile if quantile is not None else model["quantile"]

    print("=" * 92)
    print("V74 - SCORE SCALE DIAGNOSTIC")
    print("=" * 92)
    print(f"  model {model_file} | cache {price_cache} | cut top {q:.0%}")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, pr["horizon"], E64.STEP,
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, pr["label_mode"])
    te = E64.walk_forward(d, model["features"], E64.MIN_TRAIN_YEARS, 1, seed,
                          pr["horizon"], verbose=False)
    print(f"  {len(te):,} out-of-sample trades, {te['year'].nunique()} years, "
          f"{te['ticker'].nunique()} names")

    # ---- 1 + 2 ------------------------------------------------------------
    sd = scale_diagnostic(te)
    print("\n" + "=" * 92)
    print("  1  RANKING: POOLED vs WITHIN-YEAR vs RANK-NORMALISED")
    print("=" * 92)
    for lab, t in (("pooled deciles (what V73 printed)", sd["pooled"]),
                   ("deciles built INSIDE each year, then averaged",
                    sd["within"]),
                   ("score rank-normalised within year, then re-cut",
                    sd["ranked"])):
        if t is None:
            continue
        mw, nw = _mono(t["win"])
        mr, nr = _mono(t["R"])
        print(f"\n  {lab}")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"    win rate rises {mw}/{nw} steps, Spearman "
              f"{_spearman(t['score'], t['win']):+.3f}, spread "
              f"{float(t['win'].iloc[-1] - t['win'].iloc[0]):+.4f}")
        print(f"    mean R   rises {mr}/{nr} steps, Spearman "
              f"{_spearman(t['score'], t['R']):+.3f}, spread "
              f"{float(t['R'].iloc[-1] - t['R'].iloc[0]):+.4f}")
    if sd["drift"] is not None and len(sd["drift"]):
        print("\n  per-year score range - if these do not overlap, pooled "
              "deciles sort by YEAR:")
        print(sd["drift"].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))
        dr = sd["drift"]
        print(f"\n    mean per-year Spearman on win rate: "
              f"{dr['Spearman win'].mean():+.3f} "
              f"(positive in {int((dr['Spearman win'] > 0).sum())}/{len(dr)} "
              f"years)")
        print(f"    score mean ranges {dr['score mean'].min():+.4f} .. "
              f"{dr['score mean'].max():+.4f} across years")
        sd["drift"].to_csv(f"{out_dir}/score_drift.csv", index=False)

    # ---- 3 ----------------------------------------------------------------
    spy, sstat = signals_per_year(te, q)
    print("\n" + "=" * 92)
    print("  2  SIGNALS PER YEAR: fixed threshold vs top-X%-within-year")
    print("=" * 92)
    print(spy.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  absolute cut: {sstat['abs_min']} to {sstat['abs_max']} signals "
          f"per year, CV {sstat['abs_cv']:.2f}, "
          f"top 3 years hold {sstat['abs_top3_share']:.0%} of all signals")
    print(f"  relative cut: {sstat['rel_min']} to {sstat['rel_max']} signals "
          f"per year, CV {sstat['rel_cv']:.2f}")
    print("  A wildly uneven absolute count is the fingerprint of a score whose"
          " scale moves.")
    spy.to_csv(f"{out_dir}/signals_per_year.csv", index=False)

    # ---- 4 ----------------------------------------------------------------
    for lab, rel in (("absolute cut", False), ("relative cut", True)):
        per, st = sign_test_by_floor(te, q, relative=rel)
        if st is None or st.empty:
            continue
        print("\n" + "=" * 92)
        print(f"  3  SIGN TEST vs MINIMUM SIGNALS PER YEAR  ({lab})")
        print("=" * 92)
        print(st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("  A year holding one trade is a coin flip, not a draw. If p "
              "climbs as the floor")
        print("  rises, the headline was carried by near-empty years.")
        st.to_csv(f"{out_dir}/sign_test_{'rel' if rel else 'abs'}.csv",
                  index=False)

    # ---- 5 ----------------------------------------------------------------
    yd = year_drop(te, q)
    print("\n" + "=" * 92)
    print("  4  YEAR-DROP ROBUSTNESS  (remove the biggest contributors)")
    print("=" * 92)
    print(yd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  If the lift collapses after one or two years come out, the result "
          "is those years,")
    print("  not the model - and it should be reported as a regime finding "
          "instead.")
    yd.to_csv(f"{out_dir}/year_drop.csv", index=False)

    print(f"\n  wrote CSVs to {out_dir}/")
    return {"scale": sd, "signals_per_year": spy, "signal_stat": sstat,
            "year_drop": yd}


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default="entry_model_v70.joblib")
    ap.add_argument("--price-cache", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_v74(c.model_file, c.price_cache, c.out_dir, c.quantile, c.seed)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v70.joblib"
    RUN_PRICE_CACHE = None      # None uses the cache the model was trained on
    RUN_OUT_DIR     = "thesis_tables_v74"
    RUN_QUANTILE    = None      # None uses the model's own deployed cut
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_v74(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
                out_dir=RUN_OUT_DIR, quantile=RUN_QUANTILE, seed=RUN_SEED)
