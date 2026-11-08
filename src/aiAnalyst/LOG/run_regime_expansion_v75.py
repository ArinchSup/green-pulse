"""
V75 - DOES THE EDGE SURVIVE TWO MORE REGIMES?

THE QUESTION
------------
V74 localised the entire tail edge to four reversal episodes - 2012, 2014, 2015
and 2022. Remove them and precision falls below blind. Four observations cannot
support a claim, and more NAMES cannot help: 2,347 tickers inside 2022 is still
one 2022. The only scarce resource is independent market regimes.

The dataset starts 2005-01-13 but MIN_TRAIN_YEARS = 5 holds walk-forward testing
back to 2010, so 2008 and 2009 - the GFC crash and its recovery, the sharpest
reversal in the sample - have never been tested. Dropping the window to 3 years
adds them.

2008 is also the only genuinely OUT-OF-REGIME test available here. Its fold
trains on 2005 through roughly late 2007: a model that has never seen a crash,
asked to call entries during one. Nothing else in this dataset asks that.

A CONFOUND THAT TURNED OUT NOT TO EXIST
---------------------------------------
This file was built expecting one: shortening the window should retrain every
fold on less data, mixing "two new regimes" with "a weaker model everywhere".

It does not, and the reason matters. `E64.walk_forward` trains each fold on ALL
data before the fold minus the embargo:

    tr = d[pd.DatetimeIndex(d["date"]) < start - embargo]

so the training set is EXPANDING, not rolling. `min_train_years` only gates
which year testing may begin in; it never caps how much history a later fold
sees. The 2017 fold trains on the same rows whether the setting is 3 or 5.

So three arms still run, but the third is now an integrity check rather than a
control:

    A   min_train = 5, all its test years          the V74 baseline
    B   min_train = 3, all its test years          the new run, 2008 onward
    B'  arm B restricted to A's test years         must equal A exactly

B' SHOULD come back identical to A - same signals, same precision, same
Spearman. It did on the 62-name cache. If it ever does not, the fold boundaries
or the seed are not deterministic across settings and the comparison is invalid,
which is worth catching before reading anything else.

The useful consequence: A and B differ ONLY by the two extra test years, so the
experiment is clean with no adjustment needed.

Then the question that matters: do 2008 and 2009 land as hits or misses, and
does the year-drop still collapse once they are in?

    two more hits  -> six episodes, and the reversal claim has support
    two more misses -> the four episodes were luck, and it is settled either way

No model file is needed or written. The purged walk-forward IS the measurement;
a serving fit would only be scored on data it had seen.
"""

import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import evaluate_sniper_metrics_v73 as V73
import evaluate_score_scale_v74 as V74

SEED = 75
OUT_DIR = "thesis_tables_v75"


def _arm(te, label, quantile, seed, n_draws, out_dir, verbose=True):
    """Every V73/V74 check, on one arm's out-of-sample record."""
    print("\n" + "=" * 92)
    print(f"  ARM {label}")
    print("=" * 92)
    yrs = sorted(te["year"].unique())
    print(f"  {len(te):,} out-of-sample trades, test years "
          f"{yrs[0]}-{yrs[-1]} ({len(yrs)} years), "
          f"{te['ticker'].nunique()} names")

    out = {"label": label, "n_oos": len(te), "years": yrs}

    sc = V73.scorecard(te, quantile, seed, n_draws)
    if sc:
        board, fired, ctl = sc
        print(f"\n  scorecard vs random entry in the SAME NAME and MONTH "
              f"({len(fired):,} signals, {len(ctl['draws'])} draws)")
        print(board.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        board.to_csv(f"{out_dir}/scorecard_{label}.csv", index=False)
        out["board"] = board
        out["precision"] = float(fired["label"].mean())
        out["expectancy"] = float(fired["r_multiple"].mean())
        out["n_fired"] = int(len(fired))
    out["blind"] = float(te["label"].mean())
    out["blind_R"] = float(te["r_multiple"].mean())

    for lab, rel in (("absolute", False), ("relative", True)):
        per, st = V74.sign_test_by_floor(te, quantile, relative=rel)
        if st is None or st.empty:
            continue
        print(f"\n  sign test, {lab} cut")
        print(st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        st.to_csv(f"{out_dir}/sign_{lab}_{label}.csv", index=False)
        out[f"sign_{lab}"] = st
        out[f"per_year_{lab}"] = per

    spy, sstat = V74.signals_per_year(te, quantile)
    print(f"\n  signals per year: absolute {sstat['abs_min']}-"
          f"{sstat['abs_max']} (CV {sstat['abs_cv']:.2f}), "
          f"relative {sstat['rel_min']}-{sstat['rel_max']} "
          f"(CV {sstat['rel_cv']:.2f})")
    spy.to_csv(f"{out_dir}/signals_per_year_{label}.csv", index=False)
    out["signals_per_year"] = spy

    yd = V74.year_drop(te, quantile)
    print(f"\n  year-drop robustness")
    print(yd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    yd.to_csv(f"{out_dir}/year_drop_{label}.csv", index=False)
    out["year_drop"] = yd
    out["lift_full"] = float(yd["Lift"].iloc[0])
    out["lift_dropped"] = float(yd["Lift"].iloc[-1])

    sd = V74.scale_diagnostic(te)
    if sd["within"] is not None:
        w = sd["within"]
        out["within_spearman_win"] = V74._spearman(w["score"], w["win"])
        out["within_spearman_R"] = V74._spearman(w["score"], w["R"])
        print(f"\n  within-year ranking: Spearman "
              f"{out['within_spearman_win']:+.3f} on win rate, "
              f"{out['within_spearman_R']:+.3f} on R")
    return out


def run_v75(price_cache="price_cache_v43", horizon="MID", step=None,
            feature_set="no_confirm", label_mode="barrier", quantile=0.01,
            window_a=5, window_b=3, n_draws=200, seed=SEED, out_dir=OUT_DIR,
            tickers=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    step = E64.STEP if step is None else step

    print("=" * 92)
    print("V75 - REGIME EXPANSION: does the edge survive 2008-2009?")
    print("=" * 92)
    print(f"  cache {price_cache} | horizon {horizon} | label {label_mode} | "
          f"features {feature_set} | cut top {quantile:.0%}")
    print(f"  arm A min_train={window_a}   arm B min_train={window_b}")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    if tickers:
        names = [t for t in tickers if t in names]
    print(f"\n  building dataset once, reused by both arms ({len(names)} "
          f"tickers)")
    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    if d.empty:
        raise RuntimeError("no candidate entries built")
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    feats = E64.resolve_features(feature_set)
    print(f"  {len(d):,} candidate entries, {d['year'].min()}-"
          f"{d['year'].max()}, {len(feats)} features")

    arms = {}
    for lab, win in (("A", window_a), ("B", window_b)):
        print(f"\n  walk-forward, min_train_years={win} ...")
        te = E64.walk_forward(d, feats, win, 5, seed, horizon, verbose=False)
        arms[lab] = _arm(te, lab, quantile, seed, n_draws, out_dir, verbose)
        arms[lab]["te"] = te

    # B' - arm B cut back to arm A's test years. The training window is
    # expanding, so this MUST reproduce arm A exactly; it is a determinism
    # check on the fold boundaries, not a control.
    a_years = set(arms["A"]["years"])
    te_bp = arms["B"]["te"][arms["B"]["te"]["year"].isin(a_years)]
    if len(te_bp):
        arms["B'"] = _arm(te_bp, "B'", quantile, seed, n_draws, out_dir,
                          verbose)
        arms["B'"]["te"] = te_bp

    # ---- the new years, on their own ---------------------------------------
    new_years = sorted(set(arms["B"]["years"]) - a_years)
    if new_years:
        print("\n" + "=" * 92)
        print(f"  THE NEW YEARS: {', '.join(str(y) for y in new_years)}")
        print("=" * 92)
        per = arms["B"].get("per_year_relative")
        if per is not None:
            rows = per[per["Year"].isin(new_years)]
            if len(rows):
                print(rows.to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))
                hits = int((rows["Fired win"] > rows["Blind win"]).sum())
                print(f"\n  beat blind on win rate in {hits}/{len(rows)} of "
                      f"the new years")
                print(f"  2008 is the only out-of-regime test here: its fold "
                      f"never saw a crash.")
        pa = arms["B"].get("per_year_absolute")
        if pa is not None:
            thin = pa[pa["Year"].isin(new_years)]
            if len(thin):
                print(f"\n  under the ABSOLUTE cut these years fired: "
                      + ", ".join(f"{int(r.Year)}: {int(r.Signals)}"
                                  for r in thin.itertuples()))

    # ---- side by side ------------------------------------------------------
    rows = []
    for lab in ("A", "B'", "B"):
        a = arms.get(lab)
        if not a:
            continue
        sr = a.get("sign_relative")
        p30 = np.nan
        if sr is not None and len(sr):
            m = sr[sr["Min signals/year"] == 30]
            if len(m):
                p30 = float(m["p (win)"].iloc[0])
        rows.append({
            "Arm": lab,
            "Test years": f"{a['years'][0]}-{a['years'][-1]}",
            "n years": len(a["years"]),
            "Signals": a.get("n_fired", np.nan),
            "Precision": a.get("precision", np.nan),
            "Blind": a["blind"],
            "Lift": (a.get("precision", np.nan) - a["blind"]),
            "Expectancy R": a.get("expectancy", np.nan),
            "Blind R": a["blind_R"],
            "Sign p (rel, floor 30)": p30,
            "Lift after 4 drops": a.get("lift_dropped", np.nan),
            "Within-yr Spearman": a.get("within_spearman_win", np.nan)})
    comp = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("  COMPARISON")
    print("=" * 92)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    comp.to_csv(f"{out_dir}/comparison.csv", index=False)

    a, bp = arms.get("A"), arms.get("B'")
    if a and bp:
        same = (a.get("n_fired") == bp.get("n_fired")
                and np.isclose(a.get("precision", np.nan),
                               bp.get("precision", np.nan), equal_nan=True))
        print(f"\n  INTEGRITY CHECK  B' reproduces A: {same}")
        if not same:
            print("  B' MUST equal A - the training window is expanding, so "
                  "restricting arm B to")
            print("  A's years should give identical folds. It did not, so the "
                  "fold boundaries or the")
            print("  seed are not deterministic across settings. Nothing below "
                  "is a valid test until")
            print("  that is fixed.")

    print("\n  HOW TO READ THIS, IN ORDER:")
    print("  1. The integrity check above. If B' does not equal A, stop.")
    print("  2. The new-years table. Hits or misses, and how many signals the")
    print("     absolute cut even produced there.")
    print("  3. 'Lift after 4 drops' on arm B. If it is still negative, the")
    print("     extra regimes did not help and the result is settled.")
    print(f"\n  wrote CSVs to {out_dir}/")
    return arms, comp


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE = "price_cache_v43"   # the 337-name cache
    RUN_HORIZON     = "MID"               # matches the V70 model
    RUN_FEATURE_SET = "no_confirm"        # matches the V70 model
    RUN_LABEL_MODE  = "barrier"           # matches the V70 model
    RUN_QUANTILE    = 0.01                # the sniper cut
    RUN_WINDOW_A    = 5                   # baseline training window
    RUN_WINDOW_B    = 3                   # shortened window, adds 2008-2009
    RUN_N_DRAWS     = 200                 # control draws; p floor is 1/(N+1)
    RUN_OUT_DIR     = "thesis_tables_v75"
    RUN_TICKERS     = None                # None uses every ticker in the cache
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    run_v75(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
            feature_set=RUN_FEATURE_SET, label_mode=RUN_LABEL_MODE,
            quantile=RUN_QUANTILE, window_a=RUN_WINDOW_A,
            window_b=RUN_WINDOW_B, n_draws=RUN_N_DRAWS,
            out_dir=RUN_OUT_DIR, tickers=RUN_TICKERS, seed=RUN_SEED)
