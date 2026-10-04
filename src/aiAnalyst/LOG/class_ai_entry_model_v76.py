"""
V76 - DEPLOYABLE ENTRY MODEL: the rule the results actually support.

WHY V76 EXISTS
--------------
V70 is measured honestly but it SHIPS the wrong rule. Three defects, each
established by this project's own output, each fixed here. Nothing is tuned:
no feature search, no hyperparameter sweep, no threshold chosen by looking at a
test score. Searching for more edge on this dataset after establishing its
measurement ceiling is how a false positive gets manufactured, and every fix
below is forced by a result rather than chosen for one.

1. THE THRESHOLD WAS ABSOLUTE AND THE SCORE SCALE DRIFTS.
   V70 fires above a fixed score (0.6279) taken from training scores. With
   PREDICT_MODE = "expectancy" each fold regresses the R multiple under its own
   volatility regime, so per-year score means span 0.079 to 0.353. Against a
   fixed cut that produced 876 signals in 2008 and 1 in 2009 - CV 1.55 across
   years, 0 signals in 2026. Of those 876 in 2008, roughly 734 would not have
   fired under a relative rule, and 2008's fired win rate was 0.23.

   V76 fires on a ROLLING PERCENTILE: top `quantile` of the scores the universe
   produced over the trailing `WINDOW_DAYS`, recomputed every month. Strictly
   causal - the threshold for a date uses only dates before it. It fires in all
   19 test years where the absolute cut missed one, and its per-year sign test
   is four times better (p = 0.072 against 0.274) with 17 of 19 years clearing a
   15-signal floor against the absolute rule's 11.

   Note the realized firing rate is ~1.6%, not the nominal 1%: the threshold
   comes from the TRAILING window, and the score level drifts upward over parts
   of the sample, so current scores clear a stale cut more often than 1% of the
   time. That is a property of being causal, not a defect. If a sparser signal
   is wanted, lower `QUANTILE` to meet a product requirement - never to improve
   a measured lift.

   The threshold is recomputed MONTHLY, not daily, for two reasons: it is what a
   desk actually does, and a daily recompute over 4,300 dates buys nothing a
   monthly one does not.

2. THE SCORES WERE PRESENTED AS CONFIDENCE AND THEY ARE NOT PROBABILITIES.
   They are predicted R multiples; they run negative. V76 fits an ISOTONIC map
   from score to P(target before stop) and stores it, so a user-facing number
   means what it says. Isotonic is monotone, so it cannot change the ranking or
   which candidates fire - it only relabels the axis.

   CAVEAT, stated rather than hidden: the map is fitted on the purged
   walk-forward record, whose scores come from per-fold models. The SERVING
   booster is fitted on all data and its score distribution is not identical, so
   the mapping is approximate. The rolling rule is rank-based and therefore
   largely insulated from that; the printed probability is not. Treat it as
   calibrated to within a few points, not exactly.

3. describe() REPORTED THE WRONG TEST.
   V70 printed "THE LOWER BOUND DOES NOT CLEAR THE BLIND RATE ... not a buy
   signal", judging the model against buying any candidate anywhere. That
   comparison is unpaired, inherits all between-year and between-name variance,
   and moved +10.6 / +13.1 / +11.9 pp across three seeds on identical data. The
   claim that replicated is the PAIRED one: against random entry in the same
   name and the same month, +8.21 pp precision (SD 0.30) and +0.227 R
   (SD 0.006) over five runs. V76 reports that, and states the conditional.

WHAT V76 DOES NOT CLAIM
-----------------------
  - It does not say which name to buy. The validated effect is conditional on a
    name and a month the user has already chosen.
  - No sign test over years reaches p < 0.05; the best is p = 0.072.
  - In 2008, the one out-of-regime fold, it matched the blind win rate exactly
    (0.2324 vs 0.2328) while cutting expectancy loss roughly in half. It reduces
    stop-outs; it does not manufacture winners when nothing wins.
"""

import os
import sys
import hashlib

import numpy as np
import pandas as pd
import joblib

import class_ai_entry_v64 as E64

try:
    import evaluate_sniper_metrics_v73 as V73
except Exception:                                    # optional at import time
    V73 = None

PRICE_CACHE = "price_cache_v43"
HORIZON = E64.HORIZON
STEP = E64.STEP
MODEL_FILE = "entry_model_v76.joblib"

FEATURE_SET = "no_confirm"
LABEL_MODE = "barrier"
QUANTILE = 0.01              # top X% of the ROLLING reference distribution
WINDOW_DAYS = 252            # trailing calendar window for that distribution
MIN_WINDOW_ROWS = 2000       # below this the window is too thin to take a tail
REFIT_FREQ = "M"             # how often the threshold is recomputed
N_SEEDS = 5                  # V75 showed single-seed measurement is unreliable
SEED = 76

MAX_PER_WEEK = 3
APPLY_SPACED = False
# V77 measured the spacing rule on the full 337-name cache, against the same
# name-and-month matched control, and it is harmful at this signal density:
#
#   rolling 1%, unspaced       4,643 signals   +7.99 pp   +0.223 R   sign p .072
#   rolling 1%, spaced 3/wk      621 signals   +3.86 pp   +0.104 R   sign p .304
#   rolling 1%, spaced 10/wk   1,112 signals   +4.14 pp   +0.118 R   sign p .166
#
# It removes 87% of signals and halves the lift, so the signals it discards are
# BETTER than the ones it keeps - "top 3 by score this week" selects perversely,
# which follows from the weak within-year rank correlation (+0.12 to +0.33)
# already measured. V67 kept this rule after judging filters at the 10% cut,
# on a different metric (clustered lower bound against matched-size random
# filters); that verdict did not transfer to the 1% rolling rule, where 1% of
# ~337 weekly candidates is ~3.4 per week and the cap truncates every firing
# week. (An earlier version of this comment said V67 ran on a 62-name cache
# where the cap never bound. That figure came from a development cache whose
# prices do not behave like a real market - see V80's sanity check - and was
# not V67's run at all.)
#
# Spacing was solving a PORTFOLIO problem - too many positions at once - by
# throwing away information. The risk budget is where concentration belongs,
# not the signal rule.


def feature_signature(features, horizon, label_mode, rule):
    """Hash of everything that must match between training and serving."""
    blob = "|".join([",".join(features), horizon, label_mode, rule])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# =============================================================================
# THE ROLLING THRESHOLD
# =============================================================================
def rolling_threshold(dates, scores, quantile=QUANTILE,
                      window_days=WINDOW_DAYS, refit_freq=REFIT_FREQ,
                      min_rows=MIN_WINDOW_ROWS):
    """
    One threshold per refit period, from the trailing window only.

    For refit date t the threshold is the (1 - quantile) quantile of every score
    the universe produced in [t - window_days, t). The interval is closed on the
    left and OPEN ON THE RIGHT: date t itself is excluded, so no score can
    influence the threshold it is later tested against.

    Returns a Series indexed by period start, and a per-row threshold array.
    Rows before enough history exists get NaN and never fire - which is correct
    and is why the first months of any deployment are silent rather than wrong.
    """
    di = pd.DatetimeIndex(dates)
    s = np.asarray(scores, float)
    order = np.argsort(di.values, kind="mergesort")
    d_sorted, s_sorted = di[order], s[order]

    periods = pd.PeriodIndex(di, freq=refit_freq)
    starts = sorted(set(periods))
    win = pd.Timedelta(days=int(window_days * 365.25 / 252))

    cuts = {}
    for p in starts:
        t = p.to_timestamp()
        lo = np.searchsorted(d_sorted.values, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(d_sorted.values, t.to_datetime64(), "left")
        w = s_sorted[lo:hi]
        cuts[p] = (float(np.quantile(w, 1.0 - quantile))
                   if w.size >= min_rows else np.nan)
    cut_s = pd.Series(cuts).sort_index()
    return cut_s, periods.map(cuts).to_numpy(float)


def fire_rolling(te, quantile=QUANTILE, window_days=WINDOW_DAYS,
                 refit_freq=REFIT_FREQ, min_rows=MIN_WINDOW_ROWS,
                 spaced=APPLY_SPACED, max_per_week=MAX_PER_WEEK):
    """Boolean mask: score is in the rolling tail, after the spacing rule."""
    cuts, per_row = rolling_threshold(te["date"], te["p"], quantile,
                                      window_days, refit_freq, min_rows)
    fire = np.isfinite(per_row) & (te["p"].to_numpy(float) >= per_row)
    if not spaced:
        return fire, cuts
    t = te.loc[fire, ["date", "p"]].copy()
    if t.empty:
        return fire, cuts
    t["_wk"] = pd.PeriodIndex(pd.DatetimeIndex(t["date"]), freq="W")
    keep = (t.sort_values("p", ascending=False)
             .groupby("_wk").head(max_per_week).index)
    out = np.zeros(len(te), bool)
    out[te.index.get_indexer(keep)] = True
    return out, cuts


def sign_test_rolling(te, fire, floors=(0, 15, 30, 50)):
    """
    Per-year sign test on the CAUSAL rule.

    This matters more than it looks. V74 and V75 reported a "relative cut"
    computed as `te.groupby("year")["p"].rank(pct=True)` - a percentile rank
    WITHIN THE CALENDAR YEAR. In January that rank cannot be known: it depends
    on scores from the following December. So every "relative cut" sign test in
    V74/V75 (12/17, p = 0.0717 and the rest) is a FEASIBILITY UPPER BOUND, not
    a deployable result, and its tidy signal counts (CV 0.12) are partly an
    artifact of hindsight.

    The rolling rule here is causal, so this is the honest version of the same
    test and it is the one to quote. Expect it to be a little worse.
    """
    from math import comb
    te = te.reset_index(drop=True)
    f = te[np.asarray(fire, bool)]
    rows = []
    for y, g in te.groupby("year"):
        fy = f[f["year"] == y]
        if not len(fy):
            continue
        rows.append({"Year": int(y), "Signals": int(len(fy)),
                     "Fired win": float(fy["label"].mean()),
                     "Blind win": float(g["label"].mean()),
                     "Fired R": float(fy["r_multiple"].mean()),
                     "Blind R": float(g["r_multiple"].mean())})
    if not rows:
        return None
    per = pd.DataFrame(rows)
    out = []
    for fl in floors:
        s = per[per["Signals"] >= fl]
        if len(s) < 3:
            continue
        n = len(s)
        k = int((s["Fired win"] > s["Blind win"]).sum())
        kr = int((s["Fired R"] > s["Blind R"]).sum())
        out.append({"floor": fl, "years": n,
                    "beat_win": k,
                    "p_win": sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n,
                    "beat_R": kr,
                    "p_R": sum(comb(n, i) for i in range(kr, n + 1)) / 2 ** n})
    return {"per_year": per.to_dict("records"), "by_floor": out}


# =============================================================================
# CALIBRATION
# =============================================================================
def fit_calibration(te):
    """
    Isotonic map from score to P(target before stop).

    Monotone by construction, so it cannot reorder candidates or change which
    ones fire. It exists only so a number shown to a user means what it says.
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(te["p"].to_numpy(float), te["label"].to_numpy(float))
    p_hat = iso.predict(te["p"].to_numpy(float))
    y = te["label"].to_numpy(float)
    return iso, {"brier": float(np.mean((p_hat - y) ** 2)),
                 "brier_base": float(np.mean((y.mean() - y) ** 2)),
                 "n": int(len(te))}


# =============================================================================
# TRAIN
# =============================================================================
def train_entry_model(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                      feature_set=FEATURE_SET, label_mode=LABEL_MODE,
                      quantile=QUANTILE, window_days=WINDOW_DAYS,
                      n_seeds=N_SEEDS, seed=SEED, min_train_years=None,
                      n_draws=200, tickers=None, measure=True, verbose=True):
    from xgboost import XGBClassifier

    min_train_years = (E64.MIN_TRAIN_YEARS if min_train_years is None
                       else min_train_years)
    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    if verbose:
        print(f"  building dataset: {len(names)} tickers, horizon {horizon}")
    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    if d.empty:
        raise RuntimeError("no candidate entries built")
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    feats = E64.resolve_features(feature_set)

    measured, calib, calib_stat = None, None, None
    if measure:
        if verbose:
            print(f"\n  MEASUREMENT: purged walk-forward, {n_seeds}-seed "
                  f"ensemble (scored, never served)")
        te = E64.walk_forward(d, feats, min_train_years, n_seeds, seed,
                              horizon, verbose=verbose).reset_index(drop=True)
        fire, cuts = fire_rolling(te, quantile, window_days)
        fired = te[fire]
        measured = {
            "n_oos": int(len(te)), "n_fired": int(len(fired)),
            "years_fired": int(fired["year"].nunique()),
            "years_total": int(te["year"].nunique()),
            "precision": float(fired["label"].mean()),
            "expectancy_R": float(fired["r_multiple"].mean()),
            "loss_share": float((fired["outcome"] == "loss").mean()),
            "blind_precision": float(te["label"].mean()),
            "blind_expectancy_R": float(te["r_multiple"].mean()),
            "signals_per_year": fired.groupby("year").size().to_dict(),
        }
        a = fired.groupby("year").size()
        measured["signal_cv"] = float(a.std() / a.mean()) if a.mean() else None

        # the paired test - the claim that replicated.
        # V73.scorecard derives its own mask from a quantile, so the rolling
        # mask is handed to it by flooring every non-fired score to below the
        # minimum. The (1 - k/n) quantile then lands on the k-th largest value
        # and `>=` selects exactly the rolling signals. A -inf floor does NOT
        # work: np.quantile interpolates and returns nan, which silently gives
        # an empty mask and no scorecard at all.
        if V73 is not None and len(fired):
            floor = float(te["p"].min()) - 1.0
            proxy = te.assign(p=np.where(fire, te["p"], floor))
            sc = V73.scorecard(proxy, len(fired) / len(te), seed, n_draws)
            if sc:
                board, f2, _ = sc
                if len(f2) != len(fired):
                    raise RuntimeError(
                        f"matched-control proxy selected {len(f2)} rows but "
                        f"the rolling rule fires {len(fired)} - the quantile "
                        f"proxy is not reproducing the mask")
                measured["matched_control"] = board.to_dict("records")

        measured["sign_test"] = sign_test_rolling(te, fire)

        calib, calib_stat = fit_calibration(te)
        measured["calibration"] = calib_stat

    if verbose:
        print(f"\n  SERVING FIT: {n_seeds} boosters on all {len(d):,} rows")
    boosters = []
    for s in range(n_seeds):
        m = XGBClassifier(random_state=seed + s, **E64.XGB_PARAMS)
        m.fit(d[feats], d["label"], verbose=False)
        boosters.append(m)

    rule = (f"rolling_top_{quantile:.4f}_win{window_days}_{REFIT_FREQ}"
            f"_spaced{MAX_PER_WEEK if APPLY_SPACED else 0}")
    model = {
        "boosters": boosters,
        "features": feats,
        "feature_sig": feature_signature(feats, horizon, label_mode, rule),
        "rule": {"kind": "rolling_percentile", "quantile": quantile,
                 "window_days": window_days, "refit_freq": REFIT_FREQ,
                 "min_window_rows": MIN_WINDOW_ROWS,
                 "spaced": APPLY_SPACED, "max_per_week": MAX_PER_WEEK},
        "calibration": calib,
        "measured": measured,
        "provenance": {
            "n_tickers": len(names), "n_rows": int(len(d)),
            "from": str(pd.DatetimeIndex(d["date"]).min().date()),
            "trained_through": str(pd.DatetimeIndex(d["date"]).max().date()),
            "horizon": horizon, "label_mode": label_mode,
            "feature_set": feature_set, "step": step,
            "min_train_years": min_train_years,
            "n_seeds": n_seeds, "seed": seed, "price_cache": price_cache,
            "version": "v76",
        },
    }
    return model


def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)
    print(f"  saved {path}")
    return path


def load_model(path=MODEL_FILE):
    return joblib.load(path)


# =============================================================================
# DESCRIBE - reports the paired test, states the conditional
# =============================================================================
def describe(model):
    p, r, m = model["provenance"], model["rule"], model.get("measured")
    print("=" * 84)
    print("  ENTRY TIMING MODEL v76")
    print("=" * 84)
    print(f"  {p['n_tickers']} tickers, {p['n_rows']:,} candidate entries, "
          f"{p['from']} to {p['trained_through']}")
    print(f"  horizon {p['horizon']} | label {p['label_mode']} | "
          f"features {p['feature_set']} ({len(model['features'])} columns)")
    print(f"  FIRES on the top {r['quantile']:.1%} of scores from the trailing "
          f"{r['window_days']} days,")
    print(f"    recomputed {r['refit_freq']}. "
          + (f"At most {r['max_per_week']} signals per week."
             if r["spaced"] else
             "No per-week cap: V77 found spacing removes 87% of")
          )
    if not r["spaced"]:
        print(f"    signals and halves the lift at this density - "
              f"concentration belongs in the")
        print(f"    risk budget, not the signal rule.")
    print(f"    The threshold is RELATIVE, because the score scale drifts with "
          f"volatility regime")
    print(f"    and a fixed cut fires unevenly across years.")
    if not m:
        print("\n  NOT MEASURED.")
        return
    print(f"\n  MEASURED, purged walk-forward, {p['n_seeds']}-seed ensemble:")
    print(f"    fired {m['n_fired']:,} of {m['n_oos']:,} candidates in "
          f"{m['years_fired']}/{m['years_total']} years"
          + (f", signal CV {m['signal_cv']:.2f}" if m.get("signal_cv") else ""))
    print(f"    precision   {m['precision']:.1%}    "
          f"expectancy {m['expectancy_R']:+.3f}R    "
          f"stopped out {m['loss_share']:.1%}")

    mc = m.get("matched_control")
    if mc:
        print(f"\n  THE TEST THAT MATCHES THE CLAIM - random entry in the SAME "
              f"NAME and MONTH,")
        print(f"  same position count, only the entry day differs:")
        for row in mc:
            if row["Metric"] in ("precision", "expectancy_R", "loss_share"):
                print(f"    {row['Metric']:<13} model {row['Model']:>9}   "
                      f"control {row['Control median']:>9}   "
                      f"beat {row['Better than']}   p {row['p (one-sided)']:.3f}")
    st = m.get("sign_test")
    if st and st.get("by_floor"):
        print(f"\n  PER-YEAR SIGN TEST on the CAUSAL rolling rule "
              f"(no n_eff needed):")
        for r in st["by_floor"]:
            print(f"    floor {r['floor']:>3} signals/yr, {r['years']} years:  "
                  f"win {r['beat_win']}/{r['years']} p={r['p_win']:.4f}   "
                  f"R {r['beat_R']}/{r['years']} p={r['p_R']:.4f}")
        print(f"    V74/V75 quoted a 'relative cut' ranked WITHIN the calendar "
              f"year, which needs")
        print(f"    December's scores to act in January. Those p-values are a "
              f"feasibility bound.")
        print(f"    These are the causal ones and they are what belongs in the "
              f"thesis.")

    c = m.get("calibration")
    if c:
        skill = 1 - c["brier"] / c["brier_base"] if c["brier_base"] else np.nan
        print(f"\n  calibrated score -> probability (isotonic): Brier "
              f"{c['brier']:.4f} vs {c['brier_base']:.4f} base, "
              f"skill {skill:+.4f}")
        if np.isfinite(skill) and skill < 0.01:
            print(f"    THIS IS ZERO. The probability carries no information "
                  f"beyond the base rate -")
            print(f"    on the 337-name cache every scored candidate came back "
                  f"within rounding of")
            print(f"    0.4154. DO NOT show it as a per-candidate confidence: a "
                  f"number that is the")
            print(f"    base rate wearing a per-trade label reads as "
                  f"informative when it is not.")
            print(f"    Show the binary signal plus the measured precision of "
                  f"fired signals against")
            print(f"    the matched control - that is the honest number and it "
                  f"is in the block above.")
            print(f"    (The edge lives in a 1% tail; aggregate calibration "
                  f"cannot see a tail.)")

    print(f"\n  WHAT THIS MODEL CLAIMS, AND WHAT IT DOES NOT")
    print(f"    It claims that GIVEN a name and a month the user has already")
    print(f"    chosen, the days it fires on are better entries than other days")
    print(f"    in that same name and month.")
    print(f"    REPLICATED over SEVEN runs - three seeds, two ensemble sizes,")
    print(f"    two test windows (2010-2026, 2008-2026) and BOTH threshold")
    print(f"    rules (absolute and rolling):")
    print(f"      precision  +8.11 pp  (SD 0.30, range 7.75 - 8.54)")
    print(f"      expectancy +0.225 R  (SD 0.006, range 0.217 - 0.232)")
    print(f"      stop-outs  -8.64 pp")
    print(f"    All 200/200 matched control draws beaten on every run.")
    print(f"\n    It does NOT tell you which name to buy - the effect is")
    print(f"    conditional on a name and month already chosen.")
    print(f"    The PER-YEAR sign test does NOT establish year-over-year")
    print(f"    consistency and is seed-fragile at this sample size: the same")
    print(f"    rule on the same data gave 12/17 years (p=0.072) under one seed")
    print(f"    and 11/17 (p=0.166) under another. One year flipping moves p by")
    print(f"    more than twofold, so quote the range, never the better end.")
    print(f"\n    MECHANISM: out-of-sample permutation importance puts 57% of")
    print(f"    the lift on mkt_ret_20 - the 20-day MARKET return - the only")
    print(f"    feature clearing 2 SD of the permutation noise. The model has")
    print(f"    largely learned to buy market PULLBACKS. The matched control")
    print(f"    does not absorb this because it fixes the month while the")
    print(f"    20-day market return moves a lot inside a month.")
    print(f"\n    THE LIMIT THAT FOLLOWS: in 2008, the only fold whose model")
    print(f"    never saw a crash, it fired 92 signals at an 8.7% win rate")
    print(f"    against 23.3% blind, for -0.558R against the blind -0.242R.")
    print(f"    It did NOT reduce the loss there - it roughly DOUBLED it. A")
    print(f"    pullback-buying rule has no defence in a regime where")
    print(f"    pullbacks keep falling.")


# =============================================================================
# SERVE
# =============================================================================
def score(model, tickers, price_cache=None, as_of=None,
          reference_tickers=None, verbose=True):
    """
    Score candidates and apply the rolling rule.

    The rolling threshold needs a REFERENCE DISTRIBUTION: the scores the
    universe produced over the trailing window. There is no way to decide
    whether a score is in the top 1% without it, so this refuses rather than
    quietly falling back to a fixed cut - which is the defect V76 exists to fix.
    """
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    feats = model["features"]
    sig = feature_signature(feats, p["horizon"], p["label_mode"],
                            f"rolling_top_{r['quantile']:.4f}_"
                            f"win{r['window_days']}_{r['refit_freq']}_"
                            f"spaced{r['max_per_week'] if r['spaced'] else 0}")
    if sig != model["feature_sig"]:
        raise RuntimeError("feature/rule signature mismatch - this model was "
                           "built under a different configuration")

    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    ref = [t for t in (reference_tickers or available) if t in available]
    if len(ref) < 20:
        raise RuntimeError(
            f"the rolling threshold needs a reference universe and only "
            f"{len(ref)} tickers are available. Pass reference_tickers, or "
            f"point price_cache at the full cache. This model has no fixed "
            f"score cut to fall back on, by design.")

    if verbose:
        print(f"  scoring {len(ref)} reference tickers to build the trailing "
              f"distribution")
    d = E64.build_dataset(ref, price_cache, p["horizon"], p["step"],
                          verbose=False)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    if as_of:
        d = d[pd.DatetimeIndex(d["date"]) <= pd.Timestamp(as_of)]
    d = d.sort_values("date").reset_index(drop=True)
    d["p"] = np.mean([b.predict_proba(d[feats])[:, 1]
                      for b in model["boosters"]], axis=0)

    fire, cuts = fire_rolling(d, r["quantile"], r["window_days"],
                              r["refit_freq"], r["min_window_rows"],
                              r["spaced"], r["max_per_week"])
    d["fires"] = fire
    if model.get("calibration") is not None:
        d["prob_win"] = model["calibration"].predict(d["p"].to_numpy(float))

    want = set(tickers)
    out = d[d["ticker"].isin(want)].copy()
    keep = ["ticker", "date", "p", "fires"] + \
           (["prob_win"] if "prob_win" in out.columns else [])
    latest = (out.sort_values("date").groupby("ticker").tail(1)[keep]
                 .reset_index(drop=True))
    return {"latest": latest, "all": out[keep], "thresholds": cuts}


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE = "price_cache_v43"
    RUN_HORIZON     = "MID"
    RUN_FEATURE_SET = "no_confirm"
    RUN_LABEL_MODE  = "barrier"
    RUN_QUANTILE    = 0.01        # top X% of the ROLLING reference window
    RUN_WINDOW_DAYS = 252         # trailing window for that reference
    RUN_N_SEEDS     = 5
    RUN_MIN_TRAIN   = 3           # 3 adds 2008-2009 as test years
    RUN_N_DRAWS     = 200
    RUN_MODEL_FILE  = "entry_model_v76.joblib"
    RUN_MEASURE     = True
    RUN_SCORE_THESE = ["AMD", "NVDA", "AAPL"]   # [] to skip the serve demo
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    model = train_entry_model(
        price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
        feature_set=RUN_FEATURE_SET, label_mode=RUN_LABEL_MODE,
        quantile=RUN_QUANTILE, window_days=RUN_WINDOW_DAYS,
        n_seeds=RUN_N_SEEDS, min_train_years=RUN_MIN_TRAIN,
        n_draws=RUN_N_DRAWS, measure=RUN_MEASURE, seed=RUN_SEED)
    save_model(model, RUN_MODEL_FILE)
    print()
    describe(model)

    if RUN_SCORE_THESE:
        print("\n" + "=" * 84)
        print(f"  SERVE DEMO: {', '.join(RUN_SCORE_THESE)}")
        print("=" * 84)
        res = score(model, RUN_SCORE_THESE, RUN_PRICE_CACHE)
        print(res["latest"].to_string(index=False))
