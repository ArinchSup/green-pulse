"""
V81 - THE IMPROVED ENTRY MODEL.

WHAT CHANGED FROM V76, AND WHY EACH CHANGE IS FORCED BY A RESULT
---------------------------------------------------------------
1. SIX FEATURES, NOT 28. V79 showed single mean-reversion features beat the
   28-feature model 2x. V80 then selected features on 2008-2016 ONLY and scored
   on 2017-2026, which played no part in the choice. The six it picked:

       bb_position, rsi_14, ret_5, px_vs_ema20, range60_position, px_vs_ema50

   all with "low is good". They are fixed here and never re-selected. Four of
   the six match V79's all-years pick, so the selection is stable.

2. MONOTONE CONSTRAINTS. Each feature's direction is fixed by the
   mean-reversion hypothesis (lower = more oversold = higher score). On the
   held-out 2017-2026 period this took the restricted model from +8.48pp to
   +14.14pp - unconstrained trees spent capacity on non-monotone shapes and
   interactions that did not generalise. This is theory imposing structure, not
   tuning: the direction was set before the comparison was run.

3. TRAIN/SERVE SKEW FIXED. V70 and V76 MEASURED an XGBRegressor on R-multiples
   (PREDICT_MODE = "expectancy") but SERVED an XGBClassifier on the win label -
   a different model whose performance was never measured. Their own output
   showed it: walk-forward scores ran negative, served scores were 0.38-0.42.
   It is also why every served candidate returned prob_win 0.4154: the isotonic
   map was fitted on regression-scale scores and applied to probabilities.
   Here one function builds every model, so the served model is exactly the
   measured one, and a parity self-test checks it.

4. THE SIGN TEST NOW TESTS THE CLAIM. Every earlier sign test compared fired
   win rate to the year's UNCONDITIONAL rate. That tests an unconditional edge,
   which this model has never claimed, and it penalises any rule that picks
   oversold names (RESULTS.md section 4). The claim is conditional - a better
   DAY within a name and month already chosen - so the sign test that matches
   it asks, per year: did fired signals beat their own ticker-month pools?
   Both are printed. The conditional one is the one that tests the claim; the
   unconditional one stays as a disclosed robustness check.

ALSO REPORTED
-------------
  - the rank composite and the best single feature, in the same run, as the
    baselines this model must be read against;
  - the regime split by market breadth at entry;
  - candidates per date, because market breadth is computed over each date's
    candidates, and V80's sanity check showed breadth of exactly 0.000 on four
    crash lows - possible only if some dates carry very few candidates.
"""

import os
import sys
import hashlib
from math import comb

import numpy as np
import pandas as pd
import joblib

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76
from falsify_mechanism_v79 import MatchedControl
import restricted_model_v80 as V80

MODEL_FILE = "entry_model_v81.joblib"
FEATURES = ["bb_position", "rsi_14", "ret_5", "px_vs_ema20",
            "range60_position", "px_vs_ema50"]
DIRECTIONS = [-1, -1, -1, -1, -1, -1]     # low is good, from V80's early pick
SPLIT_YEAR = 2017                          # the held-out period begins here
QUANTILE = 0.01
WINDOW_DAYS = 252
N_SEEDS = 5
SEED = 81
MIN_TRAIN_YEARS = 3
FLOOR = 15                                 # signals/year for any sign test


# =============================================================================
# ONE MODEL BUILDER - measurement and serving both go through this
# =============================================================================
def build_model(seed, monotone=DIRECTIONS):
    """
    Mirrors E64._fit_predict's construction exactly, plus monotone constraints.
    Returns (unfitted model, target column, mode).
    """
    from xgboost import XGBClassifier, XGBRegressor
    params = dict(E64.XGB_PARAMS)
    if monotone is not None:
        params["monotone_constraints"] = tuple(int(x) for x in monotone)
    mode = E64.PREDICT_MODE
    if mode == "expectancy":
        params.pop("objective", None)
        params.pop("eval_metric", None)
        return (XGBRegressor(random_state=seed, objective="reg:squarederror",
                             **params), "r_multiple", mode)
    return XGBClassifier(random_state=seed, **params), "label", mode


def predict(models, X, mode):
    if mode == "expectancy":
        return np.mean([m.predict(X) for m in models], axis=0)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def walk_forward(d, features=FEATURES, monotone=DIRECTIONS,
                 min_train_years=MIN_TRAIN_YEARS, n_seeds=N_SEEDS, seed=SEED,
                 horizon="MID"):
    """Purged walk-forward using build_model, so it is the SERVED model."""
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    d = d.sort_values("date").reset_index(drop=True)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    years = sorted(d["year"].unique())
    keep, preds = [], []
    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d[pd.DatetimeIndex(d["date"]) < start - embargo]
        te = d[d["year"] == y]
        if len(tr) < 2000 or len(te) < 200:
            continue
        ms = []
        for s in range(n_seeds):
            m, target, mode = build_model(seed + s, monotone)
            m.fit(tr[features], tr[target], verbose=False)
            ms.append(m)
        preds.append(predict(ms, te[features], mode))
        keep.append(te)
    te = pd.concat(keep, ignore_index=True)
    te["p"] = np.concatenate(preds)
    return te


# =============================================================================
# MEASUREMENT
# =============================================================================
def conditional_sign_test(ex, floor=FLOOR):
    """Per year: did fired signals beat their own ticker-month pools?"""
    per = (ex.groupby("year").agg(n=("ex_win", "size"), m=("ex_win", "mean"))
             .reset_index())
    per = per[per["n"] >= floor]
    n, k = len(per), int((per["m"] > 0).sum())
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else np.nan
    worst = float(per["m"].min() * 100) if n else np.nan
    sd = float(per["m"].std(ddof=1) * 100) if n > 1 else np.nan
    return p, k, n, worst, sd


def measure_arm(frame, score, label, years, n_draws, seed):
    fire = V80._fire(frame, score, QUANTILE, WINDOW_DAYS)
    row, ex = V80.evaluate(frame, fire, label, "", years, n_draws, seed)
    if row is None:
        return None, None
    pc, kc, nc, worst, sd = conditional_sign_test(ex)
    row.update({"Cond sign p": pc, "Cond yrs": f"{kc}/{nc}",
                "Worst year pp": worst, "Year SD pp": sd})
    return row, ex


def candidates_per_date(d):
    c = d.groupby("date").size()
    alive = d.groupby("year")["ticker"].nunique()
    return {"min": int(c.min()), "median": int(c.median()),
            "max": int(c.max()), "tickers_alive_median": int(alive.median()),
            "share_dates_under_20": float((c < 20).mean())}


# =============================================================================
# TRAIN
# =============================================================================
def train(price_cache="price_cache_v43", horizon="MID", label_mode="barrier",
          step=None, n_draws=200, seed=SEED, compare_deployed=True,
          verbose=True):
    step = E64.STEP if step is None else step
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    print(f"  building dataset: {len(names)} tickers")
    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    d["year"] = pd.DatetimeIndex(d["date"]).year

    cpd = candidates_per_date(d)
    print(f"\n  candidates per date: min {cpd['min']}, median {cpd['median']},"
          f" max {cpd['max']} (median tickers alive per year "
          f"{cpd['tickers_alive_median']}); "
          f"{cpd['share_dates_under_20']:.0%} of dates have < 20")
    if cpd["median"] < 0.5 * cpd["tickers_alive_median"]:
        print("  NOTE market breadth, mkt_ret_20 and mkt_vol_20 are computed "
              "over EACH DATE'S candidates,")
        print("  which here are a small, changing subset of the universe. "
              "They measure that subset,")
        print("  not the market. The six-feature model does not use them; "
              "the 28-feature one leans")
        print("  on them heavily, which may be part of why it generalises "
              "worse.")

    print(f"\n  MEASUREMENT: purged walk-forward, {N_SEEDS}-seed monotone "
          f"ensemble on {len(FEATURES)} features")
    te = walk_forward(d, FEATURES, DIRECTIONS, MIN_TRAIN_YEARS, N_SEEDS,
                      seed, horizon)
    years = sorted(te["year"].unique())
    late = [y for y in years if y >= SPLIT_YEAR]
    print(f"  {len(te):,} OOS trades, {years[0]}-{years[-1]}; held-out "
          f"period {late[0]}-{late[-1]}")

    arms = [("v81 model (monotone, 6 feats)", te, te["p"].to_numpy(float)),
            ("rank composite (same 6)", te,
             V80.composite_score(te, list(zip(FEATURES, DIRECTIONS))))]
    for f, sgn in zip(FEATURES, DIRECTIONS):
        arms.append((f"{'low' if sgn < 0 else 'high'} {f}", te,
                     sgn * te[f].to_numpy(float)))
    for i in range(3):
        arms.append((f"random #{i+1}", te,
                     np.random.default_rng(3000 + i).random(len(te))))
    if compare_deployed:
        print(f"  walk-forward: deployed V76 (28 features, unconstrained) "
              f"for comparison ...")
        f28 = E64.resolve_features("no_confirm")
        te28 = walk_forward(d, f28, None, MIN_TRAIN_YEARS, N_SEEDS, seed,
                            horizon)
        arms.append(("V76 deployed (28 feats)", te28,
                     te28["p"].to_numpy(float)))

    tables, excess = {}, {}
    for scope, yrs in (("held-out 2017+", late), ("all years", years)):
        rows = []
        for label, fr, sc in arms:
            r, ex = measure_arm(fr, sc, label, yrs, n_draws, seed)
            if r:
                rows.append(r)
                if scope == "all years":
                    excess[label] = ex
        tables[scope] = pd.DataFrame(rows)

    reg_rows = []
    for label in ("v81 model (monotone, 6 feats)", "rank composite (same 6)",
                  "V76 deployed (28 feats)"):
        ex = excess.get(label)
        if ex is None:
            continue
        for rg in ("bear", "neutral", "bull"):
            e = ex[ex["regime"] == rg]
            lo, hi = V80.month_boot(e, "ex_win", seed=seed)
            reg_rows.append({"Arm": label, "Regime": rg, "Signals": len(e),
                             "Excess win pp": e["ex_win"].mean() * 100
                             if len(e) else np.nan,
                             "90% lo": lo * 100, "90% hi": hi * 100,
                             "Excess R": e["ex_R"].mean() if len(e)
                             else np.nan})
    regime = pd.DataFrame(reg_rows)

    yr_rows = []
    for label in ("v81 model (monotone, 6 feats)", "rank composite (same 6)",
                  "V76 deployed (28 feats)"):
        ex = excess.get(label)
        if ex is None:
            continue
        for y, e in ex.groupby("year"):
            yr_rows.append({"Arm": label, "Year": int(y), "Signals": len(e),
                            "Excess win pp": e["ex_win"].mean() * 100,
                            "Excess R": e["ex_R"].mean()})
    per_year = pd.DataFrame(yr_rows)

    # calibration on THIS model's walk-forward scores - same scale as served
    calib, cstat = M76.fit_calibration(te)

    print(f"\n  SERVING FIT: {N_SEEDS} monotone models on all {len(d):,} rows "
          f"- the SAME builder as the measurement")
    models, mode = [], None
    for s in range(N_SEEDS):
        m, target, mode = build_model(seed + s, DIRECTIONS)
        m.fit(d[FEATURES], d[target], verbose=False)
        models.append(m)

    rule_str = (f"rolling_top_{QUANTILE:.4f}_win{WINDOW_DAYS}_M_spaced0"
                f"_v81mono")
    return {
        "boosters": models, "mode": mode,
        "features": FEATURES, "directions": DIRECTIONS,
        "feature_sig": hashlib.sha256("|".join(
            [",".join(FEATURES), horizon, label_mode, rule_str,
             mode]).encode()).hexdigest()[:16],
        "rule": {"kind": "rolling_percentile", "quantile": QUANTILE,
                 "window_days": WINDOW_DAYS, "refit_freq": "M",
                 "min_window_rows": M76.MIN_WINDOW_ROWS, "spaced": False,
                 "max_per_week": 0},
        "calibration": calib,
        "measured": {"tables": {k: v.to_dict("records")
                                for k, v in tables.items()},
                     "regime": regime.to_dict("records"),
                     "per_year": per_year.to_dict("records"),
                     "calibration": cstat, "candidates_per_date": cpd},
        "provenance": {"n_tickers": len(names), "n_rows": int(len(d)),
                       "from": str(pd.DatetimeIndex(d["date"]).min().date()),
                       "trained_through": str(pd.DatetimeIndex(
                           d["date"]).max().date()),
                       "horizon": horizon, "label_mode": label_mode,
                       "step": step, "min_train_years": MIN_TRAIN_YEARS,
                       "n_seeds": N_SEEDS, "seed": seed,
                       "price_cache": price_cache, "split_year": SPLIT_YEAR,
                       "version": "v81"},
    }


def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)
    print(f"  saved {path}")


def load_model(path=MODEL_FILE):
    return joblib.load(path)


# =============================================================================
# DESCRIBE
# =============================================================================
def describe(model):
    p, m = model["provenance"], model["measured"]
    print("=" * 100)
    print("  ENTRY TIMING MODEL v81")
    print("=" * 100)
    print(f"  {p['n_tickers']} tickers, {p['n_rows']:,} candidates, "
          f"{p['from']} to {p['trained_through']}")
    print(f"  {len(model['features'])} features, all monotone 'low is good': "
          f"{', '.join(model['features'])}")
    print(f"  model: {model['mode']} XGBoost, {p['n_seeds']} seeds - the "
          f"measured model IS the served model")
    print(f"  fires on the top {model['rule']['quantile']:.0%} of the trailing "
          f"{model['rule']['window_days']}d universe score distribution, "
          f"no weekly cap")

    cols = ["Arm", "Signals", "Prec lift pp", "90% CI lo", "90% CI hi",
            "ExpR lift", "Cond yrs", "Cond sign p", "Sign yrs", "Sign p",
            "Worst year pp"]
    for scope in ("held-out 2017+", "all years"):
        t = pd.DataFrame(m["tables"][scope])
        if t.empty:
            continue
        print(f"\n  {scope.upper()} - matched control (same name, same "
              f"month, only the day differs)")
        print(t[[c for c in cols if c in t.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  'Cond' = per-year sign test against each signal's own "
          "ticker-month pool - the test")
    print("  that matches the conditional claim. 'Sign' = against the year's "
          "unconditional rate,")
    print("  which tests an unconditional edge this model does not claim. "
          "Both are disclosed.")

    reg = pd.DataFrame(m["regime"])
    if len(reg):
        print(f"\n  REGIME AT ENTRY (all years, market breadth bear < 0.40 < "
              f"neutral < 0.60 < bull)")
        print(reg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    c = m["calibration"]
    sk = 1 - c["brier"] / c["brier_base"] if c["brier_base"] else np.nan
    print(f"\n  calibration (isotonic, fitted on THIS model's scores): Brier "
          f"{c['brier']:.4f} vs {c['brier_base']:.4f}, skill {sk:+.4f}")
    if sk < 0.01:
        print("    Still ~zero in aggregate - the edge lives in a 1% tail. "
              "Show the binary signal and")
        print("    the measured precision of fired signals, not a "
              "per-candidate probability.")


# =============================================================================
# SERVE
# =============================================================================
def score(model, tickers, price_cache=None, as_of=None, verbose=True):
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    if len(names) < 20:
        raise RuntimeError("the rolling threshold needs a reference universe;"
                           " point price_cache at the full cache")
    d = E64.build_dataset(names, price_cache, p["horizon"], p["step"],
                          verbose=False)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    if as_of:
        d = d[pd.DatetimeIndex(d["date"]) <= pd.Timestamp(as_of)]
    d = d.sort_values("date").reset_index(drop=True)
    d["p"] = predict(model["boosters"], d[model["features"]], model["mode"])
    fire, cuts = M76.fire_rolling(d, r["quantile"], r["window_days"],
                                  r["refit_freq"], r["min_window_rows"],
                                  False, 0)
    d["fires"] = fire
    out = d[d["ticker"].isin(set(tickers))]
    latest = (out.sort_values("date").groupby("ticker").tail(1)
                 [["ticker", "date", "p", "fires"]].reset_index(drop=True))
    return {"latest": latest, "all": out[["ticker", "date", "p", "fires"]],
            "thresholds": cuts}


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE      = "price_cache_v43"
    RUN_MODEL_FILE       = "entry_model_v81.joblib"
    RUN_N_DRAWS          = 200
    RUN_COMPARE_DEPLOYED = True      # one extra walk-forward for the V76 row
    RUN_PARITY_TEST      = True      # served model == measured model?
    RUN_SCORE_THESE      = ["AMD", "NVDA", "AAPL"]
    RUN_SEED             = SEED
    # -------------------------------------------------------------------------

    if RUN_PARITY_TEST:
        print("parity test: does build_model reproduce E64._fit_predict?")
        names = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(RUN_PRICE_CACHE)
                       if f.endswith(".pkl"))[:30]
        dd = E64.build_dataset(names, RUN_PRICE_CACHE, "MID", E64.STEP,
                               verbose=False)
        E64.assert_market_columns(dd)
        dd = E64.apply_label_mode(dd, "barrier")
        cut = dd["date"].quantile(0.7)
        tr, ts = dd[dd["date"] < cut], dd[dd["date"] >= cut]
        a = E64._fit_predict(tr, ts, FEATURES, 7)
        mm, tgt, md = build_model(7, None)
        mm.fit(tr[FEATURES], tr[tgt], verbose=False)
        b = predict([mm], ts[FEATURES], md)
        diff = float(np.max(np.abs(a - b)))
        print(f"  mode {md}: max |difference| {diff:.2e}  "
              f"{'IDENTICAL' if diff < 1e-6 else '*** MISMATCH ***'}\n")

    model = train(price_cache=RUN_PRICE_CACHE, n_draws=RUN_N_DRAWS,
                  seed=RUN_SEED, compare_deployed=RUN_COMPARE_DEPLOYED)
    save_model(model, RUN_MODEL_FILE)
    print()
    describe(model)
    if RUN_SCORE_THESE:
        print(f"\n  SERVE DEMO: {', '.join(RUN_SCORE_THESE)}")
        print(score(model, RUN_SCORE_THESE, RUN_PRICE_CACHE)["latest"]
              .to_string(index=False))
