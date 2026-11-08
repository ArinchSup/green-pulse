"""
V78 - EVERY TABLE FOR THE RESULTS SECTION, IN ONE RUN.

Loads the deployed V76 model, rebuilds its out-of-sample record once, and emits
twelve tables as stdout, CSV and markdown. Nothing here is tuned or searched:
every table reports a quantity this project already established, and the one
new thing - feature importance - is measured two ways and both are labelled
with what they can and cannot support.

TABLES
  1   Dataset and configuration
  2   Signal rule comparison          absolute vs rolling vs spaced
  3   HEADLINE: matched control        the claim, all seven metrics
  4   Replication record              seven runs, with an in-band check
  5   Per-year detail
  6   Per-year sign test by floor
  7   Ranking quality                 pooled vs within-year (Simpson's paradox)
  8   Score drift by year             why the threshold must be relative
  9   Feature importance: gain        what the trees split on
  10  Feature importance: permutation what the deployed model actually leans on
  11  Regime boundary                 where it works and where it fails
  12  Methodological corrections       the seven wrong numbers and their fixes

ON TABLE 10, WHICH IS THE ONE WORTH READING
-------------------------------------------
Gain importance (Table 9) is what a thesis reader expects, but it is an
IN-SAMPLE quantity: it counts how much each feature improved training splits,
and a feature can dominate it while contributing nothing out of sample.

Table 10 shuffles one feature at a time and re-measures the thing the project
actually validated - the matched-control precision and expectancy lift. A
feature that matters will cost the lift when scrambled. This ties importance to
the claim rather than to a metric the model does not win on (its aggregate
Brier skill is +0.0003, so generic importance measures have almost no signal to
apportion).

CAVEAT, stated rather than buried: permutation runs against the SERVING
boosters, which were fitted on all data. So Table 10 answers "what does the
deployed model lean on", not "what generalises". A true out-of-sample
permutation would need a refit per feature per fold - 28 x 19 x 5 fits - which
is not worth the compute for a descriptive table.
"""

import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76
import evaluate_sniper_metrics_v73 as V73
import evaluate_score_scale_v74 as V74

SEED = 78
OUT_DIR = "thesis_results_v78"

# The accumulated record. Static on purpose: these are runs already made, and
# re-deriving them would cost seven walk-forwards. Table 4 checks the CURRENT
# run against this band rather than asking you to trust the band.
REPLICATION = pd.DataFrame([
    ("V73",       73, 1, "absolute", "2010-2026", 8.35, 0.2319, -8.59),
    ("V75 arm A", 75, 1, "absolute", "2010-2026", 8.54, 0.2252, -8.59),
    ("V75 arm B", 75, 1, "absolute", "2008-2026", 7.75, 0.2261, -9.23),
    ("V75 arm A", 75, 5, "absolute", "2010-2026", 8.28, 0.2181, -8.12),
    ("V75 arm B", 75, 5, "absolute", "2008-2026", 8.11, 0.2317, -9.16),
    ("V77",       77, 5, "rolling",  "2008-2026", 7.99, 0.2234, np.nan),
    ("V76 refit", 76, 5, "rolling",  "2008-2026", 7.77, 0.2166, -8.13),
], columns=["Run", "Seed", "Ensemble", "Rule", "Test years",
            "Precision lift (pp)", "Expectancy lift (R)",
            "Stop-out change (pp)"])

# The band above was recorded on the 337-name cache. A different universe has a
# different effect size, so Table 4's in-band check only applies there.
REPLICATION_N_TICKERS = 337

CORRECTIONS = pd.DataFrame([
    ("Compounded return as the metric",
     "model 856% vs random 1,381% -> 'loses to luck'",
     "Unpaired on time in market. A zero-skill signal lost 2.4x to an "
     "unmatched control invested in 59 months vs 24. Replaced with a control "
     "matched on name, month and position count."),
    ("'Beat the best of N random draws'",
     "a ~p<0.025 bar carried by one draw",
     "The max of 20 draws sits near the 97.5th percentile and moves with the "
     "seed. Replaced with the model's rank in the control distribution."),
    ("Sign test with no floor on signals per year",
     "12/16 years, p = 0.0384",
     "Counted years holding 1, 4, 6, 7 and 11 trades as full draws, four of "
     "them as wins. At a 30-signal floor: 8/10, p = 0.055."),
    ("Single-seed walk-forward measurement",
     "year-drop lift -1.29 / +1.81 / -0.87 pp",
     "Seed variance exceeded the effect on every unconditional quantity. The "
     "year-drop test straddles zero and supports no claim either way."),
    ("Pooled score deciles",
     "Spearman -0.745: 'the score is inverted'",
     "Simpson's paradox. Per-year score means span 0.079-0.353, so pooled "
     "deciles sort by year. Within-year the correlation is positive."),
    ("Within-year percentile rank as the 'relative' cut",
     "12/17 years, p = 0.0717, signal CV 0.12",
     "Ranking inside a calendar year needs December's scores to act in "
     "January. A feasibility bound, not a deployable rule. The causal "
     "trailing-window rule gives CV ~1.0."),
    ("Spacing rule (max 3 signals/week)",
     "+3.86pp lift on 621 signals",
     "Removes 87% of signals and halves the lift at 337-name density, so the "
     "discarded signals were better than the kept ones. V67 kept it after "
     "judging filters at the 10% cut on a different metric; the verdict did "
     "not transfer to the 1% rolling rule. Disabled."),
], columns=["What was wrong", "The number it produced", "Why, and the fix"])


def _emit(df, name, out_dir, float_fmt="{:.4f}"):
    df.to_csv(f"{out_dir}/{name}.csv", index=False)
    try:
        with open(f"{out_dir}/{name}.md", "w") as fh:
            fh.write(df.to_markdown(index=False))
    except Exception:
        pass
    print(df.to_string(index=False,
                       float_format=lambda v: float_fmt.format(v)))


def _head(n, title):
    print("\n" + "=" * 96)
    print(f"  TABLE {n}  {title}")
    print("=" * 96)


# =============================================================================
# 9 + 10  IMPORTANCE
# =============================================================================
def gain_importance(model):
    """Mean gain and split count across the serving ensemble. In-sample."""
    feats = model["features"]
    gains, weights = {f: [] for f in feats}, {f: [] for f in feats}
    for b in model["boosters"]:
        bst = b.get_booster()
        g = bst.get_score(importance_type="gain")
        w = bst.get_score(importance_type="weight")
        for f in feats:
            gains[f].append(float(g.get(f, 0.0)))
            weights[f].append(float(w.get(f, 0.0)))
    rows = [{"Feature": f,
             "Mean gain": float(np.mean(gains[f])),
             "Gain SD": float(np.std(gains[f], ddof=1)) if len(gains[f]) > 1
                        else 0.0,
             "Mean splits": float(np.mean(weights[f]))} for f in feats]
    df = pd.DataFrame(rows).sort_values("Mean gain", ascending=False)
    tot = df["Mean gain"].sum()
    df["Share of gain"] = df["Mean gain"] / tot if tot else np.nan
    return df.reset_index(drop=True)


def permutation_importance(d, features, horizon, quantile, window_days,
                           split_frac=0.6, n_seeds=3, seed=SEED, repeats=1,
                           verbose=True):
    """
    Cost to the VALIDATED claim when one feature is scrambled, OUT OF SAMPLE.

    For each feature: shuffle that column in the held-out part, re-predict,
    re-derive the rolling mask, and recompute the matched-control precision and
    expectancy lift. The drop is the importance.

    THIS FITS ITS OWN HOLD-OUT and does not touch the serving ensemble. The
    first version of this function re-predicted with the serving boosters,
    which were trained on all data - its baseline read +19.33pp against the
    walk-forward's +6.15pp, because it was measuring the in-sample fit. An
    in-sample importance table printed beside out-of-sample results is read as
    a result, so it is not worth the convenience.

    One ensemble is fitted on the earliest `split_frac` of dates, purged by the
    holding window, and everything is measured on the remainder. That is a
    single train/test split rather than the full walk-forward, so its baseline
    will not match Table 3 exactly - it has one model instead of nineteen and
    less training data. Read the RANKING and the relative shares, not the
    absolute pp.

    The control is recomputed for every shuffle rather than held fixed, because
    scrambling a feature changes WHICH candidates fire and the control must
    track the same name-months.
    """
    from xgboost import XGBClassifier

    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    di = pd.DatetimeIndex(d["date"])
    split = di.min() + (di.max() - di.min()) * split_frac
    tr = d[di < split - embargo]
    teh = d[di >= split].reset_index(drop=True)
    if verbose:
        print(f"  hold-out split at {split.date()} (embargo {embargo.days}d): "
              f"train {len(tr):,} rows, test {len(teh):,} rows")
    if len(tr) < 2000 or len(teh) < 2000:
        print("  hold-out too small for a permutation table - skipped")
        return None, None

    boosters = []
    for s in range(n_seeds):
        m = XGBClassifier(random_state=seed + s, **E64.XGB_PARAMS)
        m.fit(tr[features], tr["label"], verbose=False)
        boosters.append(m)

    rng = np.random.default_rng(seed)

    def lift(frame):
        p = np.mean([b.predict_proba(frame[features])[:, 1]
                     for b in boosters], axis=0)
        t = frame.assign(p=p).reset_index(drop=True)
        fire, _ = M76.fire_rolling(t, quantile, window_days, spaced=False)
        f = t[fire]
        if len(f) < 50:
            return np.nan, np.nan, int(len(f))
        floor = float(t["p"].min()) - 1.0
        sc = V73.scorecard(t.assign(p=np.where(fire, t["p"], floor)),
                           len(f) / len(t), seed, 40)
        if not sc:
            return np.nan, np.nan, int(len(f))
        b = sc[0].set_index("Metric")
        return (float(f["label"].mean()) - float(b.loc["precision",
                                                       "Control median"]),
                float(f["r_multiple"].mean()) - float(b.loc["expectancy_R",
                                                            "Control median"]),
                int(len(f)))

    te, feats = teh, features
    base_p, base_r, base_n = lift(te)
    if verbose:
        print(f"  hold-out baseline: {base_n} signals, precision lift "
              f"{base_p*100:+.2f}pp, expectancy lift {base_r:+.4f}R")

    rows = []
    for i, f in enumerate(feats, 1):
        dp, dr, ns = [], [], []
        for _ in range(repeats):
            sh = te.copy()
            sh[f] = rng.permutation(sh[f].to_numpy())
            lp, lr, n = lift(sh)
            dp.append(base_p - lp)
            dr.append(base_r - lr)
            ns.append(n)
        rows.append({"Feature": f,
                     "Precision lift lost (pp)": float(np.nanmean(dp)) * 100,
                     "Expectancy lift lost (R)": float(np.nanmean(dr)),
                     "Signals when scrambled": int(np.mean(ns))})
        if verbose and i % 5 == 0:
            print(f"    [{i}/{len(feats)}]")
    df = pd.DataFrame(rows).sort_values("Precision lift lost (pp)",
                                        ascending=False)
    if np.isfinite(base_p) and base_p != 0:
        df["Share of lift lost"] = df["Precision lift lost (pp)"] / (base_p
                                                                     * 100)
    # Is this table readable at all? If scrambling a single feature moves the
    # lift by more than the whole lift, the deltas are noise and the ranking
    # means nothing. An importance table that cannot rank must say so rather
    # than let its top row be read as a finding.
    dl = df["Precision lift lost (pp)"].to_numpy(float)
    base_pp = base_p * 100
    stat = {"base_precision_lift_pp": base_pp,
            "base_expectancy_lift_R": base_r, "base_signals": base_n,
            "repeats": repeats, "split": str(split.date()),
            "delta_sd": float(np.nanstd(dl, ddof=1)),
            "max_abs_delta": float(np.nanmax(np.abs(dl))),
            "n_negative": int((dl < 0).sum()), "n_features": int(len(dl)),
            "readable": bool(np.nanmax(np.abs(dl)) <= abs(base_pp)
                             and (dl < 0).sum() <= 0.25 * len(dl))}
    return df.reset_index(drop=True), stat


# =============================================================================
# RUNNER
# =============================================================================
def run_v78(model_file="entry_model_v76.joblib", price_cache=None,
            out_dir=OUT_DIR, n_draws=200, seed=SEED, permutation=True,
            perm_repeats=1, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model = M76.load_model(model_file)
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    q, win = r["quantile"], r["window_days"]

    print("=" * 96)
    print("V78 - RESULTS SECTION TABLES")
    print("=" * 96)
    print(f"  model {model_file} | cache {price_cache}")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, p["horizon"], p["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year
    te = E64.walk_forward(d, model["features"], p["min_train_years"],
                          p["n_seeds"], p["seed"], p["horizon"],
                          verbose=False).reset_index(drop=True)
    fire, cuts = M76.fire_rolling(te, q, win, spaced=r["spaced"],
                                  max_per_week=r["max_per_week"])
    fired = te[fire]

    # ---- 1 ----------------------------------------------------------------
    _head(1, "Dataset and configuration")
    cfg = pd.DataFrame([
        ("Universe", f"{p['n_tickers']} US equities"),
        ("Candidate entries", f"{p['n_rows']:,}"),
        ("Sample period", f"{p['from']} to {p['trained_through']}"),
        ("Out-of-sample trades", f"{len(te):,}"),
        ("Test years", f"{te['year'].min()}-{te['year'].max()} "
                       f"({te['year'].nunique()} years)"),
        ("Candidate spacing", f"every {p['step']} trading days per name"),
        ("Horizon", f"{p['horizon']} ({E64.HORIZON_CONFIGS[p['horizon']]['eval_days']} days)"),
        ("Label", f"triple barrier, {p['label_mode']}, ties to the stop"),
        ("Features", f"{p['feature_set']} ({len(model['features'])} columns)"),
        ("Validation", f"purged walk-forward, embargo = holding window, "
                       f"min {p['min_train_years']}y train"),
        ("Ensemble", f"{p['n_seeds']} seeds, averaged"),
        ("Signal rule", f"top {q:.1%} of the trailing {win}d universe "
                        f"distribution, refit {r['refit_freq']}"),
        ("Per-week cap", "none" if not r["spaced"] else str(r["max_per_week"])),
        ("Signals fired", f"{len(fired):,} ({len(fired)/len(te):.2%} of "
                          f"candidates), {fired['year'].nunique()} years"),
    ], columns=["Item", "Value"])
    _emit(cfg, "t01_configuration", out_dir)

    # ---- 2 ----------------------------------------------------------------
    _head(2, "Signal rule comparison (matched control throughout)")
    rows = []
    for label, kind, sp, mpw in (("absolute, unspaced", "abs", False, 0),
                                 ("rolling, unspaced", "roll", False, 0),
                                 ("rolling, spaced 3/wk", "roll", True, 3),
                                 ("rolling, spaced 10/wk", "roll", True, 10)):
        if kind == "abs":
            m = te["p"].to_numpy(float) >= float(np.quantile(te["p"], 1 - q))
        else:
            m, _ = M76.fire_rolling(te, q, win, spaced=sp, max_per_week=mpw)
        f = te[m]
        if not len(f):
            continue
        floor = float(te["p"].min()) - 1.0
        sc = V73.scorecard(te.assign(p=np.where(m, te["p"], floor)),
                           len(f) / len(te), seed, n_draws)
        b = sc[0].set_index("Metric") if sc else None
        cm = (lambda k: float(b.loc[k, "Control median"])
              if b is not None else np.nan)
        yrs = f.groupby("year").size()
        rows.append({"Rule": label, "Signals": len(f),
                     "Years firing": int(f["year"].nunique()),
                     "Signal CV": float(yrs.std() / yrs.mean()),
                     "Precision": float(f["label"].mean()),
                     "Precision lift (pp)": (float(f["label"].mean())
                                             - cm("precision")) * 100,
                     "Expectancy (R)": float(f["r_multiple"].mean()),
                     "Expectancy lift (R)": (float(f["r_multiple"].mean())
                                             - cm("expectancy_R"))})
    t2 = pd.DataFrame(rows)
    _emit(t2, "t02_rule_comparison", out_dir)
    print("\n  The deployed rule is 'rolling, unspaced'. The absolute rule's "
          "larger raw lift is")
    print("  concentrated in fewer years; the spaced rules discard signals "
          "that were better than")
    print("  the ones they keep.")

    # ---- 3 ----------------------------------------------------------------
    _head(3, "HEADLINE - model vs random entry in the same name and month")
    sc = V73.scorecard(te.assign(p=np.where(fire, te["p"],
                                            float(te["p"].min()) - 1.0)),
                       len(fired) / len(te), seed, n_draws)
    if sc:
        _emit(sc[0], "t03_matched_control", out_dir)
        print(f"\n  {len(fired):,} signals, {n_draws} control draws. The "
              f"control holds name, calendar")
        print(f"  month and position count fixed, so only the entry DAY "
              f"differs. p is the one-sided")
        print(f"  empirical rank and cannot fall below {1/(n_draws+1):.4f} "
              f"with {n_draws} draws.")

    # ---- 4 ----------------------------------------------------------------
    _head(4, "Replication record across runs")
    rep = REPLICATION.copy()
    _emit(rep, "t04_replication", out_dir)
    for c in ("Precision lift (pp)", "Expectancy lift (R)"):
        v = rep[c].dropna().to_numpy(float)
        print(f"  {c:<24} n={len(v)}  mean {v.mean():+.4f}  "
              f"SD {v.std(ddof=1):.4f}  range {v.min():+.4f} .. {v.max():+.4f}")
    if sc:
        b = sc[0].set_index("Metric")
        cur_p = (float(fired["label"].mean())
                 - float(b.loc["precision", "Control median"])) * 100
        lo, hi = (rep["Precision lift (pp)"].min(),
                  rep["Precision lift (pp)"].max())
        print(f"\n  IN-BAND CHECK  this run's precision lift is {cur_p:+.2f}pp "
              f"against the recorded")
        print(f"  band [{lo:+.2f}, {hi:+.2f}].")
        if p["n_tickers"] < REPLICATION_N_TICKERS * 0.9:
            print(f"  NOT COMPARABLE: the band was recorded on "
                  f"{REPLICATION_N_TICKERS} tickers and this run has "
                  f"{p['n_tickers']}.")
            print(f"  A different universe has a different effect size; the "
                  f"check is only meaningful")
            print(f"  on the cache the band came from.")
        else:
            print(f"  Verdict: "
                  f"{'INSIDE' if lo - 0.5 <= cur_p <= hi + 0.5 else 'OUTSIDE'}"
                  f" the band.")

    # ---- 5 + 6 ------------------------------------------------------------
    _head(5, "Per-year detail")
    st = M76.sign_test_rolling(te, fire)
    per = pd.DataFrame(st["per_year"])
    per["Win lift"] = per["Fired win"] - per["Blind win"]
    per["R lift"] = per["Fired R"] - per["Blind R"]
    _emit(per, "t05_per_year", out_dir)

    _head(6, "Per-year sign test by minimum signals")
    t6 = pd.DataFrame(st["by_floor"]).rename(columns={
        "floor": "Min signals/year", "years": "Years kept",
        "beat_win": "Beat on win rate", "p_win": "p (win)",
        "beat_R": "Beat on R", "p_R": "p (R)"})
    _emit(t6, "t06_sign_test", out_dir)
    print("\n  This test needs no effective-sample estimate. It is also "
          "SEED-FRAGILE at this")
    print("  sample size: the same rule on the same data gave 12/17 "
          "(p=0.072) under one seed")
    print("  and 11/17 (p=0.166) under another. Quote the range, not the "
          "better end.")

    # ---- 7 ----------------------------------------------------------------
    _head(7, "Ranking quality: pooled vs within-year (Simpson's paradox)")
    sd = V74.scale_diagnostic(te)
    rows = []
    for lab, t in (("pooled", sd["pooled"]), ("within-year", sd["within"]),
                   ("rank-normalised", sd["ranked"])):
        if t is None:
            continue
        rows.append({"View": lab,
                     "Spearman (win)": V74._spearman(t["score"], t["win"]),
                     "Spearman (R)": V74._spearman(t["score"], t["R"]),
                     "Decile 1 win": float(t["win"].iloc[0]),
                     "Decile 10 win": float(t["win"].iloc[-1]),
                     "Spread (win)": float(t["win"].iloc[-1]
                                           - t["win"].iloc[0])})
    _emit(pd.DataFrame(rows), "t07_ranking_quality", out_dir)
    if sd["pooled"] is not None:
        _emit(sd["pooled"], "t07b_pooled_deciles", out_dir)

    # ---- 8 ----------------------------------------------------------------
    _head(8, "Score drift by year, and why the threshold must be relative")
    spy, sstat = V74.signals_per_year(te, q)
    _emit(spy, "t08_signals_per_year", out_dir)
    print(f"\n  absolute cut: {sstat['abs_min']}-{sstat['abs_max']} signals "
          f"per year (CV {sstat['abs_cv']:.2f}), top 3 years hold "
          f"{sstat['abs_top3_share']:.0%}")
    if sd["drift"] is not None and len(sd["drift"]):
        _emit(sd["drift"], "t08b_score_drift", out_dir)

    # ---- 9 ----------------------------------------------------------------
    _head(9, "Feature importance: gain (in-sample, serving ensemble)")
    gi = gain_importance(model)
    _emit(gi, "t09_importance_gain", out_dir)
    print("\n  Gain counts how much each feature improved TRAINING splits. It "
          "is in-sample and a")
    print("  feature can dominate it while contributing nothing out of "
          "sample. Table 10 is the")
    print("  one tied to the validated claim.")

    # ---- 10 ---------------------------------------------------------------
    if permutation:
        _head(10, "Feature importance: cost to the matched-control lift "
                  "(out-of-sample)")
        pi, pstat = permutation_importance(d, model["features"], p["horizon"],
                                           q, win, n_seeds=p["n_seeds"],
                                           seed=seed, repeats=perm_repeats,
                                           verbose=verbose)
        if pi is not None:
            _emit(pi, "t10_importance_permutation", out_dir)
            print(f"\n  Single train/test split at {pstat['split']}, purged by "
                  f"the holding window:")
            print(f"  {pstat['base_signals']} signals, precision lift "
                  f"{pstat['base_precision_lift_pp']:+.2f}pp, expectancy lift "
                  f"{pstat['base_expectancy_lift_R']:+.4f}R.")
            print(f"  Each row scrambles ONE feature ({pstat['repeats']} "
                  f"repeat(s)) in the held-out part and")
            print(f"  reports how much of that lift is lost. It fits its own "
                  f"model, so this is")
            print(f"  out-of-sample - but it is ONE model on less data, not "
                  f"the nineteen-fold")
            print(f"  walk-forward, so the baseline will not match Table 3.")
            print(f"\n  IS THIS TABLE READABLE?")
            print(f"    baseline lift            {pstat['base_precision_lift_pp']:+.2f} pp")
            print(f"    SD of the deltas          {pstat['delta_sd']:.2f} pp")
            print(f"    largest single delta      {pstat['max_abs_delta']:.2f} pp")
            print(f"    features where scrambling HELPED  "
                  f"{pstat['n_negative']}/{pstat['n_features']}")
            if pstat["readable"]:
                print(f"    READABLE: deltas are small against the lift and "
                      f"few features help when")
                print(f"    scrambled, so the ranking carries information.")
            else:
                # Name the criterion that failed. An earlier version asserted
                # both, and printed "the delta exceeds the whole lift" on a run
                # where it did not - a verdict that states untrue reasons is
                # worse than no verdict.
                big = pstat["max_abs_delta"] > abs(pstat[
                    "base_precision_lift_pp"])
                many = pstat["n_negative"] > 0.25 * pstat["n_features"]
                print(f"    PARTIALLY READABLE AT BEST. Failing criteria:")
                if big:
                    print(f"      - the largest single delta "
                          f"({pstat['max_abs_delta']:.2f}pp) exceeds the whole "
                          f"lift ({abs(pstat['base_precision_lift_pp']):.2f}pp)"
                          f" - the deltas are noise end to end.")
                if many:
                    print(f"      - {pstat['n_negative']}/"
                          f"{pstat['n_features']} features IMPROVE the lift "
                          f"when destroyed, so the")
                    print(f"        lower part of the ranking carries no "
                          f"information.")
                sd = pstat["delta_sd"]
                clear = pi[pi["Precision lift lost (pp)"] > 2 * sd]
                if len(clear) and not big:
                    print(f"\n    Features clearing 2 SD of the noise "
                          f"({2*sd:.2f}pp) - the only ones reportable:")
                    for row in clear.itertuples():
                        print(f"      {row.Feature:<20} "
                              f"{getattr(row, '_2'):.2f}pp  "
                              f"({getattr(row, '_2')/sd:.1f} SD, "
                              f"{getattr(row, '_2')/abs(pstat['base_precision_lift_pp']):.0%}"
                              f" of the lift)")
                    print(f"    Report THOSE and say importance could not be "
                          f"established for the rest.")
                else:
                    print(f"\n    Report Table 9 as the in-sample split "
                          f"statistic it is, and say that")
                    print(f"    out-of-sample importance could not be "
                          f"established at this sample size.")

    # ---- 11 ---------------------------------------------------------------
    _head(11, "Regime boundary: where it works and where it fails")
    best = per.nlargest(4, "Win lift")[["Year", "Signals", "Fired win",
                                        "Blind win", "Win lift", "R lift"]]
    worst = per.nsmallest(4, "Win lift")[["Year", "Signals", "Fired win",
                                          "Blind win", "Win lift", "R lift"]]
    t11 = pd.concat([best.assign(Group="best 4"),
                     worst.assign(Group="worst 4")])
    _emit(t11, "t11_regime", out_dir)
    print("\n  Failures are scattered across easy and hard years, so no rule "
          "predicts when the")
    print("  model works. 2008 is the only out-of-regime fold - its model "
          "never saw a crash.")

    # ---- 12 ---------------------------------------------------------------
    _head(12, "Methodological corrections")
    _emit(CORRECTIONS, "t12_corrections", out_dir)

    print(f"\n  wrote CSV and markdown for every table to {out_dir}/")
    return {"config": cfg, "rules": t2, "per_year": per, "sign": t6,
            "gain": gi, "scorecard": sc[0] if sc else None}


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE   = "entry_model_v76.joblib"
    RUN_PRICE_CACHE  = None     # None uses the cache the model was trained on
    RUN_OUT_DIR      = "thesis_results_v78"
    RUN_N_DRAWS      = 200      # control draws; p floor is 1/(N+1)
    RUN_PERMUTATION  = True     # Table 10. The slow one - 28 features x refit-free
                                # re-predict. Set False for a quick pass.
    RUN_PERM_REPEATS = 3        # averages the permutation noise down; 1 is too few
    RUN_SEED         = SEED
    # -------------------------------------------------------------------------

    run_v78(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
            out_dir=RUN_OUT_DIR, n_draws=RUN_N_DRAWS,
            permutation=RUN_PERMUTATION, perm_repeats=RUN_PERM_REPEATS,
            seed=RUN_SEED)
