"""
V77 - IS THE SPACING RULE COSTING YOU THE EFFECT?

THE PROBLEM
-----------
V76's matched-control lift came in at +3.87pp precision / +0.111R. The five
earlier runs averaged +8.21pp / +0.227R. The effect appears to have halved.

But V76 changed TWO things at once, and I should have separated them before
reporting a number:

    absolute threshold  ->  rolling causal threshold
    unspaced            ->  spacing rule (max 3 signals per week)

Every earlier figure came from an UNSPACED absolute cut. V76 is rolling AND
spaced. So "the causal rule halved the effect" is not established - the spacing
cap is at least as likely a cause, and on a 337-name universe it is the suspect.

WHY THE UNIVERSE SIZE MATTERS HERE
----------------------------------
With STEP = 5, each ticker produces a candidate every 5 trading days, so a
337-name universe produces roughly 337 candidates a week. One percent of that
is ~3.4 per week, right at the 3-per-week cap - and signals are LUMPY, so the
weeks that do fire want far more than 3 and get truncated hard. Observed: 2,837
unspaced signals become 620 spaced, a 78% cut.

On a 62-name development cache the same rule produces ~0.6 firing candidates a
week, so the cap almost never binds: 325 unspaced becomes 312 spaced. NOTE: that
cache was later found not to behave like a real market (V80's sanity check), so
its numbers illustrate the density argument only - they are not evidence. V67
kept the rule after judging filters at the 10% cut on a different metric; that
verdict does not transfer to the 1% rolling rule at 337 names.

Discarding 78% of signals at random would leave the lift alone and only widen
the interval. Halving the lift requires the DISCARDED signals to have been
better than the kept ones - that is, "top 3 by score within the week" selects
slightly perversely, which is consistent with the weak within-year rank
correlation (+0.12 to +0.33) already measured. Plausible, and testable.

WHAT THIS RUNS
--------------
Four rules on one dataset build and one walk-forward, each scored against the
same name-and-month matched control:

    absolute 1%, unspaced      what every earlier figure used
    rolling  1%, unspaced      the causal threshold, spacing off
    rolling  1%, spaced 3/wk   what V76 shipped
    rolling  1%, spaced 10/wk  a looser cap, in case some spacing helps

(The 62-name development cache's numbers once quoted here are withdrawn: its
prices do not behave like a real market. The 337-name run is the evidence.)
"""

import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76
import evaluate_sniper_metrics_v73 as V73

SEED = 77


def run_v77(price_cache="price_cache_v43", horizon="MID", step=None,
            feature_set="no_confirm", label_mode="barrier", quantile=0.01,
            window_days=252, min_train_years=3, n_seeds=5, n_draws=200,
            seed=SEED, out_dir="thesis_tables_v77", verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    step = E64.STEP if step is None else step

    print("=" * 96)
    print("V77 - ROLLING vs ABSOLUTE, SPACED vs UNSPACED")
    print("=" * 96)
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    feats = E64.resolve_features(feature_set)
    te = E64.walk_forward(d, feats, min_train_years, n_seeds, seed, horizon,
                          verbose=False).reset_index(drop=True)
    print(f"  {len(te):,} out-of-sample trades, {te['year'].nunique()} years, "
          f"{te['ticker'].nunique()} names")

    rules = [
        ("absolute 1%, unspaced",   dict(kind="abs")),
        ("rolling 1%, unspaced",    dict(kind="roll", spaced=False)),
        ("rolling 1%, spaced 3/wk", dict(kind="roll", spaced=True,
                                         max_per_week=3)),
        ("rolling 1%, spaced 10/wk", dict(kind="roll", spaced=True,
                                          max_per_week=10)),
    ]
    rows = []
    for label, cfg in rules:
        if cfg["kind"] == "abs":
            cut = float(np.quantile(te["p"], 1.0 - quantile))
            fire = te["p"].to_numpy(float) >= cut
        else:
            fire, _ = M76.fire_rolling(
                te, quantile, window_days, spaced=cfg["spaced"],
                max_per_week=cfg.get("max_per_week", 3))
        f = te[fire]
        if not len(f):
            continue
        floor = float(te["p"].min()) - 1.0
        sc = V73.scorecard(te.assign(p=np.where(fire, te["p"], floor)),
                           len(f) / len(te), seed, n_draws)
        b = sc[0].set_index("Metric") if sc else None

        def ctl(metric, col="Control median"):
            return float(b.loc[metric, col]) if b is not None else np.nan

        def beat(metric):
            return b.loc[metric, "Better than"] if b is not None else ""

        st = M76.sign_test_rolling(te, fire)
        p15 = next((r for r in (st or {}).get("by_floor", [])
                    if r["floor"] == 15), None)
        yrs = f.groupby("year").size()
        rows.append({
            "Rule": label, "Signals": len(f),
            "Years": int(f["year"].nunique()),
            "Signal CV": float(yrs.std() / yrs.mean()) if yrs.mean() else np.nan,
            "Precision": float(f["label"].mean()),
            "Control": ctl("precision"),
            "Prec lift pp": (float(f["label"].mean()) - ctl("precision")) * 100,
            "Beat (prec)": beat("precision"),
            "ExpR": float(f["r_multiple"].mean()),
            "Control R": ctl("expectancy_R"),
            "ExpR lift": float(f["r_multiple"].mean()) - ctl("expectancy_R"),
            "Beat (R)": beat("expectancy_R"),
            "Sign p (floor 15)": p15["p_win"] if p15 else np.nan,
            "Years (floor 15)": p15["years"] if p15 else np.nan})

    out = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("  RESULTS  (matched control: same name, same month, same count)")
    print("=" * 96)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv(f"{out_dir}/rule_comparison.csv", index=False)

    if len(out) >= 3:
        best = out.loc[out["Prec lift pp"].idxmax()]
        print(f"\n  LARGEST LIFT: {best['Rule']}  "
              f"(+{best['Prec lift pp']:.2f}pp, {best['Signals']} signals)")
        sp = out[out["Rule"].str.contains("spaced 3")]
        un = out[out["Rule"].str.contains("rolling 1%, unspaced")]
        if len(sp) and len(un):
            dp = float(un["Prec lift pp"].iloc[0] - sp["Prec lift pp"].iloc[0])
            ds = int(un["Signals"].iloc[0] - sp["Signals"].iloc[0])
            print(f"  SPACING COSTS: {dp:+.2f}pp of precision lift and "
                  f"{ds} signals ({ds/max(int(un['Signals'].iloc[0]),1):.0%} "
                  f"of them).")
            print(f"  If that is a large positive number, set APPLY_SPACED = "
                  f"False in V76 and refit.")
    print(f"\n  wrote {out_dir}/rule_comparison.csv")
    return out


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE = "price_cache_v43"
    RUN_HORIZON     = "MID"
    RUN_FEATURE_SET = "no_confirm"
    RUN_LABEL_MODE  = "barrier"
    RUN_QUANTILE    = 0.01
    RUN_WINDOW_DAYS = 252
    RUN_MIN_TRAIN   = 3
    RUN_N_SEEDS     = 5
    RUN_N_DRAWS     = 200
    RUN_OUT_DIR     = "thesis_tables_v77"
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    run_v77(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
            feature_set=RUN_FEATURE_SET, label_mode=RUN_LABEL_MODE,
            quantile=RUN_QUANTILE, window_days=RUN_WINDOW_DAYS,
            min_train_years=RUN_MIN_TRAIN, n_seeds=RUN_N_SEEDS,
            n_draws=RUN_N_DRAWS, out_dir=RUN_OUT_DIR, seed=RUN_SEED)
