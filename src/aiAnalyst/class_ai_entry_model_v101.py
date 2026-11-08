"""
V101 - THE FINAL SHORT-HORIZON ENTRY MODEL (a 10-trading-day trade).

WHAT IT IS
----------
V88's recipe - a classifier on the same ten "low is good" features, shallow
trees, 5 seeds - trained on V100's short trade:
    target = 0.5 x ATR% x sqrt(10), kept within 3.27%-16.33%
    stop   = 0.6 x target (reward:risk 1.67, break-even 37.5%)
    exit   at the close after 10 trading days if neither is hit
Trained on the 337 core stocks only, every year - the population every fold
model in V100 was trained on. Training on all 2,360 stocks would serve a model
nobody has measured.

WHAT IT CLAIMS (V100, pre-registered)
-------------------------------------
Given a stock the user already chose, the days it fires on are better entries
than other days in that stock and month: +11.7 pp target hit rate on 1,292
unseen liquid stocks, 2008-2026, 90% [+8.5, +14.9]. It does NOT beat the
simple rule chosen in advance (low RSI-14 alone): -0.3 pp [-3.6, +2.8] - the
two pick different days of about equal quality. It does not say which stock to
buy. Most trades end within about 4 days; 10 days is the longest.

CHECKS BEFORE IT IS SAVED (as V88)
----------------------------------
  1. MEASURED-MODEL PARITY  one V100 fold is refit with this file's builder; it
                            must reproduce V100's cached fold predictions.
  2. SERVING PARITY         the recommender's feature builder must give the same
                            features on every training row.
  3. CALIBRATION            may the probability be shown to the user? Among
                            fired signals, the fold models' mean probability
                            must sit within 3 pp of the real hit rate on the
                            core AND on the unseen liquid stocks.

The model file carries its own trade geometry, so the recommender
(entry_recommender_v101.py) prices stops and targets for the 10-day trade
without changing trade_config.py or anything V88 uses.

It reads V100's caches, so run V100 first; this takes about 10 minutes.
"""

import copy
import hashlib
import os
import textwrap
import time
import warnings

import numpy as np
import pandas as pd
import joblib

import trade_config as TC
import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import class_ai_entry_model_v88 as M88
import short_entry_model_v100 as V100

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_FILE = os.path.join("class_model", "entry_model_v101.joblib")
HORIZON = "SHORT"                  # the name the recommender serves it under
CFG = V86.COMBO_CFG                # V88's recipe
FEATURES, DIRECTIONS = V85.features_of(CFG)
SEED = V85.SEED                    # V100's fold models used the same seeds
N_SEEDS = 5
MODE = "probability"
CALIB_TOL = M88.CALIB_TOL          # 3 pp
GEOMETRY = V100.geometry(V100.HOLD)
V100.register(GEOMETRY)            # adds a key to trade_config; changes nothing


# =============================================================================
# THE MODEL - V88's builder, so the fold models in V100 are reproduced exactly
# =============================================================================
def predict(models, X):
    return M88.predict(models, X)


def fit_ensemble(frame, n_seeds=N_SEEDS, seed=SEED):
    return M88.fit_ensemble(frame, n_seeds, seed)


def levels(entry, atr, model=None):
    """Stop and target for the SHORT trade, with the geometry stored in the
    model file (trade_config's own levels are used for MID)."""
    g = (model or {}).get("geometry", GEOMETRY)
    V100.register(g)
    with V100.short_limits(g):
        return TC.compute_levels(entry, atr, g["key"])


def serving_frame(names, cache, horizon=None, step=E64.STEP, as_of=None,
                  chunk=200):
    """V88's serving features (every 5th bar on the training grid plus the
    latest bar). `horizon` is accepted for the recommender's interface; the
    short geometry is used either way."""
    with V100.short_limits(GEOMETRY):
        return M88.serving_frame(names, cache, GEOMETRY["key"], step, as_of,
                                 chunk)


# =============================================================================
# WHAT V100 MEASURED - stored in the file so the recommender can quote it
# =============================================================================
def measured_tables(v100_dir):
    m = {}
    for k, f in (("v100_primary", "primary.csv"), ("v100_trades", "trades.csv"),
                 ("v100_regimes", "regimes.csv"), ("v100_years", "years.csv"),
                 ("v100_opponent", "opponent_choice.csv")):
        p = os.path.join(v100_dir, f)
        m[k] = pd.read_csv(p).to_dict("records") if os.path.exists(p) else []
    for k, f in (("v100_verdict", "verdict.txt"),
                 ("v100_preregistration", "preregistration.txt")):
        p = os.path.join(v100_dir, f)
        m[k] = open(p).read() if os.path.exists(p) else ""
    return m


def claim_line(model):
    """One sentence on what an ENTER signal has meant, from V100's record."""
    me = model.get("measured", {})
    pr = pd.DataFrame(me.get("v100_primary", []))
    lift = "Lift over same stock-month, pp [90%]"
    if not len(pr) or lift not in pr or "Arm" not in pr:
        return ("ENTER days are top-1% entry days within the stock and month; "
                "the signal does not say whether to own the stock.")
    arm = pr["Arm"].astype(str)
    m = pr[arm == "model"]
    r = pr[(arm != "model") & ~arm.str.startswith("random")]
    out = (f"On unseen liquid stocks (2008-2026), ENTER days beat other days "
           f"in the same stock and month by {m[lift].iloc[0]} pp target hit "
           f"rate." if len(m) else "")
    if len(r):
        v = me.get("v100_verdict", "")
        how = ("did better" if "WORSE than the rule" in v else
               "did not do as well" if "BEATS the rule" in v else "did as well")
        out += (f" A simple '{r['Arm'].iloc[0]}' rule {how} "
                f"({r[lift].iloc[0]} pp).")
    return out + " It does not say whether to own the stock."


# =============================================================================
# TRAIN AND PACKAGE
# =============================================================================
def train(model_in81="entry_model_v81.joblib", core_cache=None,
          big_cache="price_cache_v68", hold=V100.HOLD, n_seeds=N_SEEDS,
          chunk=250, parity_year=2020, run_calibration=True,
          v100_dir=V100.OUT_DIR, verbose=True):
    t0 = time.time()
    W = 100
    g = V100.geometry(hold)
    pre = os.path.join(v100_dir, "preregistration.txt")
    if os.path.exists(pre) and f"after {hold} trading days" not in open(
            pre).read():
        raise SystemExit(f"V100 was pre-registered for a different hold than "
                         f"{hold} days - set RUN_HOLD to the hold V100 tested")
    core_cache = core_cache or M81.load_model(model_in81)["provenance"][
        "price_cache"]
    print("=" * W)
    print(f"V101 - THE FINAL SHORT-HORIZON ENTRY MODEL ({g['hold']}-day trade)")
    print("=" * W)

    verdict = ""
    vp = os.path.join(v100_dir, "verdict.txt")
    if os.path.exists(vp):
        verdict = open(vp).read()
    if "TIMING" not in verdict or "-> CONFIRMED" not in verdict:
        print("  *** V100's timing test is not confirmed (or V100 has not been "
              "run) - V101 trains the model anyway, but the claim does not "
              "stand")

    print("  [1] DATA - V100's short-trade data and fold models (cached)")
    core, fresh, wf, core_names, _ = V100.prepare(core_cache, big_cache, g,
                                                  n_seeds, chunk, verbose)
    print(f"      {len(core):,} core candidate days, "
          f"{core['ticker'].nunique()} stocks, {core['date'].min().date()} "
          f"to {core['date'].max().date()}")

    parity = {}
    print(f"\n  [2] MEASURED-MODEL PARITY - refit the {parity_year} fold with "
          f"this file's builder")
    diff, n = M88.measured_parity(core, wf["combo"]["core"], parity_year,
                                  n_seeds, g["key"])
    parity["measured"] = {"year": parity_year, "rows": n, "max_abs_diff": diff}
    ok = np.isfinite(diff) and diff < 1e-6
    print(f"      {n:,} rows, max |difference| vs V100's fold predictions "
          f"{diff:.2e}  {'IDENTICAL' if ok else '*** MISMATCH ***'}")

    print(f"\n  [3] SERVING FIT - {n_seeds} seeds on all {len(core):,} core "
          f"rows")
    models = fit_ensemble(core, n_seeds)

    print("\n  [4] SERVING PARITY - the recommender's feature builder vs the "
          "training rows")
    F = serving_frame(core_names, core_cache, step=V100.STEP)
    worst, n = M88.serving_parity(core, F)
    parity["serving"] = {"shared_rows": n, "train_rows": int(len(core)),
                         "max_abs_diff": worst}
    print(f"      {n:,} of {len(core):,} training rows rebuilt; max "
          f"|difference| {worst:.2e}  "
          f"{'IDENTICAL' if worst < 1e-12 and n == len(core) else '*** MISMATCH ***'}")
    lat = F[F["is_latest"]]
    print(f"      it also builds the latest bar for {len(lat):,} stocks "
          f"(latest {lat['date'].max().date()}); the last training row is "
          f"{core['date'].max().date()} (a label needs {g['hold']} more days)")

    calib_rows, calib_stats = [], []
    if run_calibration:
        print(f"\n  [5] CALIBRATION - out-of-sample fold probabilities, "
              f"{V100.YEARS[0]}-{V100.YEARS[-1]}")
        Uc, wc = V87.universe(core, np.ones(len(core), bool), wf, "core")
        Uf, wfr = V87.universe(fresh, fresh["liquid"].to_numpy(bool), wf,
                               "fresh")
        for lab, U, w in ((f"core {len(core_names)}", Uc, wc),
                          ("unseen, liquid", Uf, wfr)):
            rel, st = M88.calibration_check(U, *w["combo"], V100.YEARS, lab)
            calib_rows.append(rel)
            calib_stats.append(st)
        rel = pd.concat(calib_rows, ignore_index=True)
        print("      " + rel.to_string(index=False, float_format=lambda v:
                                       f"{v:.3f}").replace("\n", "\n      "))
        for st in calib_stats:
            print(f"      {st['universe']:<15} Brier skill {st['skill']:+.4f}; "
                  f"fired: mean p minus hit rate {st['fired_gap']:+.3f} on "
                  f"{st['fired_n']:,} signals")
    display_p = bool(calib_stats) and all(
        np.isfinite(s["fired_gap"]) and abs(s["fired_gap"]) <= CALIB_TOL
        for s in calib_stats)

    rule_str = (f"rolling_top_{M88.QUANTILE:.4f}_win{M88.WINDOW_DAYS}_M_"
                f"strict_v101")
    model = {
        "boosters": models, "mode": MODE, "features": FEATURES,
        "directions": DIRECTIONS, "config": copy.deepcopy(CFG),
        "geometry": dict(g),
        "feature_sig": hashlib.sha256("|".join(
            [",".join(FEATURES), g["key"], "barrier", rule_str, MODE])
            .encode()).hexdigest()[:16],
        "rule": {"kind": "rolling_percentile", "quantile": M88.QUANTILE,
                 "window_days": M88.WINDOW_DAYS, "refit_freq": "M",
                 "min_window_rows": M88.MIN_ROWS, "strict": True},
        "display_probability": display_p,
        "calibration": {"stats": calib_stats,
                        "table": (pd.concat(calib_rows).to_dict("records")
                                  if calib_rows else []),
                        "tolerance": CALIB_TOL},
        "parity": parity,
        "measured": measured_tables(v100_dir),
        "provenance": {"n_tickers": int(core["ticker"].nunique()),
                       "n_rows": int(len(core)),
                       "from": str(core["date"].min().date()),
                       "trained_through": str(core["date"].max().date()),
                       "horizon": HORIZON, "geometry_key": g["key"],
                       "hold_days": g["hold"], "label_mode": "barrier",
                       "step": V100.STEP, "n_seeds": n_seeds, "seed": SEED,
                       "price_cache": core_cache, "version": "v101",
                       "selected_in": "V88's recipe (sweep_options_v85.py, "
                                      "chosen on 60-day trades, 2017-2021); "
                                      "nothing re-tuned for 10 days",
                       "confirmed_in": "short_entry_model_v100.py "
                                       "(pre-registered, unseen stocks)"}}
    print(f"\n  trained in {(time.time() - t0) / 60:.1f} min")
    return model


def save_model(model, path=MODEL_FILE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"  saved {path}")


def load_model(path=MODEL_FILE):
    return joblib.load(path)


# =============================================================================
# DESCRIBE
# =============================================================================
def describe(model):
    p, g, me = model["provenance"], model["geometry"], model["measured"]
    W = 100
    print("\n" + "=" * W)
    print(f"  ENTRY TIMING MODEL v101 - SHORT horizon ({g['hold']}-day trade), "
          f"final")
    print("=" * W)
    print(f"  trained on {p['n_tickers']} stocks, {p['n_rows']:,} candidate "
          f"days, {p['from']} to {p['trained_through']}")
    print(f"  trade: target {g['k']} x ATR% x sqrt({g['hold']}) within "
          f"{g['floor']:.2%}-{g['ceil']:.2%}; stop {g['sl']} x target; exit "
          f"after {g['hold']} trading days")
    print(f"  XGBoost classifier (P[target before stop]), shallow, "
          f"{p['n_seeds']} seeds, every feature 'low is good':")
    print(f"    {', '.join(model['features'])}")
    print("  fires when the score is STRICTLY above the 99th percentile of the "
          "universe's trailing 252 days")
    print(f"  recipe: {p['selected_in']}")
    print(f"  confirmed in {p['confirmed_in']}")
    pr = pd.DataFrame(me.get("v100_primary", []))
    if len(pr):
        print("\n  V100 - UNSEEN LIQUID STOCKS, 2008-2026 (pre-registered)")
        print("  " + pr.fillna("").to_string(index=False)
              .replace("\n", "\n  "))
    if me.get("v100_verdict"):
        print("\n" + me["v100_verdict"].rstrip())
    par = model.get("parity", {})
    if par:
        m, s = par.get("measured", {}), par.get("serving", {})
        print(f"\n  PARITY: fold refit max |diff| {m.get('max_abs_diff', np.nan):.1e}"
              f"; serving features max |diff| "
              f"{s.get('max_abs_diff', np.nan):.1e} on "
              f"{s.get('shared_rows', 0):,} of {s.get('train_rows', 0):,} rows")
    print("\n  PROBABILITY DISPLAY: "
          + ("allowed - among fired signals the fold models' mean probability "
             "was within 3 pp of the hit rate on both sets of stocks"
             if model.get("display_probability") else
             "NOT allowed - show the binary signal and the measured hit rate"))
    print("\n  WHAT IT CLAIMS: " + textwrap.fill(
        claim_line(model) + " Most trades end within a few days.", width=84,
        subsequent_indent="  "))


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_CORE_CACHE  = None              # None = the cache V81/V88 were trained on
    RUN_BIG_CACHE   = "price_cache_v68" # V100's unseen stocks (for the checks)
    RUN_HOLD        = 10                # must equal V100's
    RUN_N_SEEDS     = 5                 # must equal V100's (cache key)
    RUN_CHUNK       = 250               # must equal V100's (cache key)
    RUN_PARITY_YEAR = 2020              # which fold to refit for the parity test
    RUN_CALIBRATION = True
    RUN_V100_DIR    = "thesis_tables_v100"
    RUN_MODEL_OUT   = MODEL_FILE    # class_model/entry_model_v101.joblib
    # -------------------------------------------------------------------------

    model = train(core_cache=RUN_CORE_CACHE, big_cache=RUN_BIG_CACHE,
                  hold=RUN_HOLD, n_seeds=RUN_N_SEEDS, chunk=RUN_CHUNK,
                  parity_year=RUN_PARITY_YEAR, run_calibration=RUN_CALIBRATION,
                  v100_dir=RUN_V100_DIR)
    save_model(model, RUN_MODEL_OUT)
    describe(model)
