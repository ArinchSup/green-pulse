"""
V83 - TIES, NOT DRIFT: WHY V81 STILL FLOODS, AND THE FIX.

WHAT V82 SHOWED
---------------
V82 rescored each fold's trailing window with that fold's own model, which
removes fold-to-fold scale drift completely. V81 still fired 7,637 times in
2010 - 53% of that year's candidates - against 7,735 under the old cut. So the
drift diagnosis was wrong.

The arithmetic points one way. Over a year, the monthly cut is the 99th
percentile of a window that increasingly contains that year's own scores. More
than half of those rows can sit at or above it only if they are TIED AT THE CUT.
Under `score >= cut`, a tied block is admitted whole. Some fold models must be
close to flat: a few distinct values, a large share of candidates on the top
one.

WHY A FOLD WOULD GO FLAT (a hypothesis this file tests, not a finding)
---------------------------------------------------------------------
V81 forces every feature to be "low is good". Folds trained on years dominated
by 2008 - when oversold stocks kept falling - see data that CONTRADICTS that
direction. A monotone model that cannot fit the allowed direction has little
choice but to stay flat, so its predictions collapse onto a handful of values.
If that is right, the flat folds are the early ones, and the problem fades as
2008 becomes a smaller share of each fold's training data.

WHAT THIS FILE DOES
-------------------
  [3] DEGENERACY per year: distinct score values, the share sitting on the
      year's maximum, and the modal share - for V81's fold models, V76's, and
      the SERVED model scoring the whole history (which is what production
      faces). This confirms or kills the tie explanation directly.

  [4]-[5] every arm under BOTH rules:
        >=   the current rule; a tied block at the cut is admitted whole
        >    strict; a block tied AT the cut is never admitted. A model that
             cannot tell candidates apart does not fire on them - the right
             behaviour for a model meant to fire only when very confident.
      Both are reported for every arm, because strict also changes features
      with natural mass points (range60_position is exactly 0 for every stock
      at its 60-day low). The cost to those arms is shown, not hidden.

  WARM START FOR EVERY ARM. V82 computed non-model arms on the test rows only,
  so their trailing windows started cold in 2008, and composite v2 lost the
  whole of 2008 because NaN scores in a window turned its cut to NaN. Here every
  non-model arm is scored on the full dataset from 2005 with NaN-aware cuts,
  then read off on the test rows - the same warm start the model arms already
  had through their rescored trailing windows.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import restricted_model_v80 as V80
import measure_v82 as V82

SEED = 81
OUT_DIR = "thesis_tables_v83"
QUANTILE = V82.QUANTILE
MIN_ROWS = V82.MIN_ROWS
SPLIT_YEAR = V82.SPLIT_YEAR


# =============================================================================
# RULES
# =============================================================================
def rolling_cut(dates, scores, quantile=QUANTILE, min_rows=MIN_ROWS):
    """Per-row monthly cut from the trailing window. NaN-aware."""
    win = V82._win()
    di = pd.DatetimeIndex(dates)
    s = np.asarray(scores, float)
    o = np.argsort(di.values, kind="mergesort")
    ds, ss = di.values[o], s[o]
    codes, uniq = pd.factorize(pd.PeriodIndex(di, freq="M"))
    cut = np.full(len(s), np.nan)
    for k, m in enumerate(uniq):
        t = m.to_timestamp()
        lo = np.searchsorted(ds, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(ds, t.to_datetime64(), "left")
        w = ss[lo:hi]
        w = w[np.isfinite(w)]
        if w.size < min_rows:
            continue
        cut[codes == k] = float(np.quantile(w, 1.0 - quantile))
    return cut


def apply_rule(score, cut, strict):
    s = np.asarray(score, float)
    ok = np.isfinite(cut) & np.isfinite(s)
    return ok & ((s > cut) if strict else (s >= cut))


def fold_cut(te, refs, quantile=QUANTILE, min_rows=MIN_ROWS):
    """V82's fold-consistent cut, NaN-aware, returned per row."""
    win = V82._win()
    cut = np.full(len(te), np.nan)
    per = pd.PeriodIndex(pd.DatetimeIndex(te["date"]), freq="M")
    yrs = te["year"].to_numpy()
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
            w = w[np.isfinite(w)]
            if w.size < min_rows:
                continue
            cut[rows[pr == m]] = float(np.quantile(w, 1.0 - quantile))
    return cut


# =============================================================================
# DEGENERACY
# =============================================================================
def degeneracy(frame, score):
    """
    'Share >= own p99' is the number that decides it. For a continuous score
    it is ~1%. If a plateau sits on or just under the top, the 99th percentile
    lands ON the plateau and this share jumps - which is exactly what makes
    `>=` admit half a year. 'Share at max' alone misses a plateau one notch
    below a handful of higher values; the development data showed that case.
    """
    s = np.round(np.asarray(score, float), 9)
    rows = []
    for y in sorted(pd.unique(frame["year"])):
        v = s[(frame["year"] == y).to_numpy()]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        vals, cnt = np.unique(v, return_counts=True)
        p99 = float(np.quantile(v, 0.99))
        rows.append({"Year": int(y), "Rows": int(v.size),
                     "Distinct": int(vals.size),
                     "Share >= own p99": float((v >= p99).mean()),
                     "Share at max": float(cnt[-1] / v.size),
                     "Modal share": float(cnt.max() / v.size)})
    return pd.DataFrame(rows)


# =============================================================================
# RUNNER
# =============================================================================
def run_v83(model_in="entry_model_v81.joblib", price_cache=None,
            out_dir=OUT_DIR, n_draws=200, compare_deployed=True,
            seed=SEED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    m81 = M81.load_model(model_in)
    prov = m81["provenance"]
    price_cache = price_cache or prov["price_cache"]
    feats, dirs = m81["features"], m81["directions"]
    fd = list(zip(feats, dirs))

    print("=" * 100)
    print("V83 - TIES, NOT DRIFT")
    print("=" * 100)
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, prov["horizon"], prov["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year
    d["row_id"] = np.arange(len(d))

    print("\n  [1] full-universe breadth")
    b, _ = V82.full_breadth(price_cache, names)
    d["breadth_full"] = pd.DatetimeIndex(d["date"]).normalize().map(b) \
        .to_numpy(float)

    print("  [2] walk-forwards (V81, and V76 for comparison)")
    te, refs = V82.walk_forward_ref(d, feats, dirs, prov["min_train_years"],
                                    prov["n_seeds"], prov["seed"],
                                    prov["horizon"])
    te28, refs28 = None, None
    if compare_deployed:
        f28 = E64.resolve_features("no_confirm")
        te28, refs28 = V82.walk_forward_ref(d, f28, None,
                                            prov["min_train_years"],
                                            prov["n_seeds"], prov["seed"],
                                            prov["horizon"])

    # ---- [3] degeneracy ---------------------------------------------------
    print("\n" + "=" * 100)
    print("  [3] DEGENERACY - are some models nearly flat?")
    print("=" * 100)
    dg81 = degeneracy(te, te["p"])
    served = M81.predict(m81["boosters"], d[feats], m81["mode"])
    dgs = degeneracy(d, served)
    show = dg81.merge(dgs, on="Year", how="outer",
                      suffixes=(" (V81 fold)", " (served)"))
    if te28 is not None:
        dg28 = degeneracy(te28, te28["p"]).rename(columns={
            c: f"{c} (V76 fold)" for c in ("Rows", "Distinct",
                                           "Share >= own p99",
                                           "Share at max", "Modal share")})
        show = show.merge(dg28[["Year", "Distinct (V76 fold)",
                                "Share >= own p99 (V76 fold)"]],
                          on="Year", how="left")
    keep = ["Year", "Distinct (V81 fold)", "Share >= own p99 (V81 fold)",
            "Modal share (V81 fold)", "Distinct (served)",
            "Share >= own p99 (served)"] + \
        [c for c in show.columns if "V76" in c]
    print(show[keep].to_string(index=False,
                               float_format=lambda v: f"{v:.3f}"))
    show.to_csv(f"{out_dir}/degeneracy.csv", index=False)
    print("\n  'Share >= own p99' should be ~0.010 for a continuous score. "
          "Well above that means")
    print("  a plateau sits at the cut, and `>=` admits the whole plateau.")
    flat = dg81[dg81["Share >= own p99"] > 0.03]["Year"].tolist()
    print(f"  V81 fold years over 3%: {flat or 'none'}")
    sv = dgs[dgs["Share >= own p99"] > 0.03]["Year"].tolist()
    print(f"  SERVED-model years over 3%: {sv or 'none'}"
          f"  <- what production faces")

    # ---- arms under both rules -------------------------------------------
    rid = te["row_id"].to_numpy()
    arms = []          # (label, kind, frame, score, cut)
    arms.append(("V81", "v81", te, te["p"].to_numpy(float),
                 fold_cut(te, refs)))
    if te28 is not None:
        arms.append(("V76 (28 feats)", "v76", te28,
                     te28["p"].to_numpy(float), fold_cut(te28, refs28)))
    # non-model arms: score and cut on the FULL dataset (warm from 2005),
    # then read off on the test rows
    def full_arm(label, kind, score_d):
        cut_d = rolling_cut(d["date"], score_d)
        return (label, kind, te, np.asarray(score_d, float)[rid], cut_d[rid])
    arms.append(full_arm("composite v2", "comp2", V82.composite_v2(d, fd)))
    for f, sgn in fd:
        arms.append(full_arm(f"{'low' if sgn < 0 else 'high'} {f}", "single",
                             sgn * d[f].to_numpy(float)))
    for i in range(3):
        arms.append(full_arm(f"random #{i+1}", "floor",
                             np.random.default_rng(5000 + i).random(len(d))))

    # ---- [4] coverage ------------------------------------------------------
    print("\n" + "=" * 100)
    print("  [4] COVERAGE under each rule")
    print("=" * 100)
    fires, cov_rows, per_year = {}, [], {}
    for label, kind, fr, sc, cut in arms:
        for rule, strict in ((">=", False), (">", True)):
            f = apply_rule(sc, cut, strict)
            fires[(label, rule)] = (kind, fr, f)
            k, rate, st = V82.coverage(fr, f)
            per_year[f"{label} {rule}"] = k
            cov_rows.append({"Arm": label, "Rule": rule, "Total": st["total"],
                             "Min yr": st["min"], "Max yr": st["max"],
                             "CV": st["cv"], "Zero yrs": st["zero_years"],
                             "Max rate": st["max_rate"]})
    cov = pd.DataFrame(cov_rows)
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cov.to_csv(f"{out_dir}/coverage.csv", index=False)
    pyc = pd.DataFrame(per_year).fillna(0).astype(int)
    pyc.to_csv(f"{out_dir}/signals_per_year.csv")
    cols = [c for c in ("V81 >=", "V81 >", "composite v2 >",
                        "V76 (28 feats) >") if c in pyc.columns]
    print("\n" + pyc[cols].to_string())

    # ---- [5] matched control -----------------------------------------------
    years = sorted(te["year"].unique())
    late = [y for y in years if y >= SPLIT_YEAR]
    tabs, excess = {}, {}
    for scope, yrs in (("held-out 2017+", late), ("all years", years)):
        rows = []
        for (label, rule), (kind, fr, f) in fires.items():
            r, ex = V80.evaluate(fr, f, label, kind, yrs, n_draws, seed)
            if r is None:
                continue
            pc, kc, nc, worst, sd = M81.conditional_sign_test(ex)
            r.update({"Rule": rule, "Cond yrs": f"{kc}/{nc}",
                      "Cond sign p": pc, "Worst year pp": worst})
            rows.append(r)
            if scope == "all years" and rule == ">":
                excess[label] = V82.excess_regime(fr, f)
        tabs[scope] = pd.DataFrame(rows)
        tabs[scope].to_csv(f"{out_dir}/arms_{scope.split()[0]}.csv",
                           index=False)
    cols = ["Arm", "Rule", "Signals", "Prec lift pp", "90% CI lo",
            "90% CI hi", "ExpR lift", "Cond yrs", "Sign yrs", "Sign p",
            "Worst year pp"]
    for scope in ("held-out 2017+", "all years"):
        print("\n" + "=" * 100)
        print(f"  [5] {scope.upper()} - matched control, both rules")
        print("=" * 100)
        t = tabs[scope].sort_values(["Kind", "Arm", "Rule"])
        print(t[cols].to_string(index=False,
                                float_format=lambda v: f"{v:.3f}"))

    # ---- [6] regime under the strict rule ---------------------------------
    print("\n" + "=" * 100)
    print("  [6] REGIME AT ENTRY (full-universe breadth), STRICT rule, all "
          "years")
    print("=" * 100)
    A = tabs["all years"]
    sing = A[(A["Kind"] == "single") & (A["Rule"] == ">")]
    best = sing.loc[sing["Prec lift pp"].idxmax(), "Arm"] if len(sing) \
        else None
    reg_rows = []
    for label in ("V81", "composite v2", best, "V76 (28 feats)"):
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
    reg = pd.DataFrame(reg_rows)
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    reg.to_csv(f"{out_dir}/regime_strict.csv", index=False)

    # ---- [7] verdict -------------------------------------------------------
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    cv = cov.set_index(["Arm", "Rule"])
    for a in ("V81", "composite v2", best, "V76 (28 feats)"):
        if a is None:
            continue
        for rule in (">=", ">"):
            if (a, rule) not in cv.index:
                continue
            c = cv.loc[(a, rule)]
            line = (f"  {a:<22} {rule:<2}  coverage {int(c['Total']):>6,} "
                    f"(max yr {int(c['Max yr']):,}, CV {c['CV']:.2f})")
            for scope, T in (("held-out", tabs["held-out 2017+"]),
                             ("all", A)):
                r = T[(T["Arm"] == a) & (T["Rule"] == rule)]
                if len(r):
                    r = r.iloc[0]
                    line += (f" | {scope} {r['Prec lift pp']:+6.2f}pp "
                             f"ExpR {r['ExpR lift']:+.3f}")
            print(line)
    fl = A[(A["Kind"] == "floor") & (A["Rule"] == ">")]["Prec lift pp"]
    if len(fl):
        print(f"\n  floor (random, strict, all years): {fl.min():+.2f} .. "
              f"{fl.max():+.2f} pp")
    print(f"\n  wrote CSVs to {out_dir}/")
    return tabs, cov, reg, show


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN         = "entry_model_v81.joblib"
    RUN_PRICE_CACHE      = None
    RUN_OUT_DIR          = "thesis_tables_v83"
    RUN_N_DRAWS          = 200
    RUN_COMPARE_DEPLOYED = True
    RUN_SEED             = SEED
    # -------------------------------------------------------------------------

    run_v83(model_in=RUN_MODEL_IN, price_cache=RUN_PRICE_CACHE,
            out_dir=RUN_OUT_DIR, n_draws=RUN_N_DRAWS,
            compare_deployed=RUN_COMPARE_DEPLOYED, seed=RUN_SEED)
