"""
V88 - THE FINAL ENTRY MODEL.

WHAT CHANGED SINCE V84, AND WHY
-------------------------------
  MODEL    V85's chosen combination replaces V81: a CLASSIFIER (predicts
           win/loss), the six oversold features plus four short-horizon ones
           (rsi_2, ret_1, ret_3, dist_low10), all monotone "low is good",
           shallow trees (depth 3, 500 rounds), 5 seeds. V85 chose it on
           2017-2021 only. V86 then tested it, pre-registered, on 2,024
           stocks that played no part in building anything: +2.73 pp win
           rate over `low bb_position`, 90% [+0.67, +4.89], two-shot
           (Bonferroni) lower bound +0.34. V87 confirmed the lead survives
           resampling that respects overlapping trades (lower bound +0.31 /
           +0.29 with 3- / 6-month blocks). V81 alone did not clear zero.
  TRAINED  on the 337 core names only - the population every fold model in
           V86/V87 was trained on. Training on all 2,360 names would serve a
           model nobody has measured.
  SERVING  V84's score() built its rows with the TRAINING dataset builder,
           which only keeps days that already have 60 bars of future (it
           needs them for the label). So the "latest" signal it reported was
           at least 61 trading days old, and the current month's cut was set
           from a window missing its last three months. score() now builds
           features without labels: every 5th bar on the training grid PLUS
           the latest bar, the same eligibility filter, the same feature code.
           A parity test checks it gives bit-identical features on every row
           the training set also has.

WHAT IT CLAIMS
--------------
Given a stock and a month the user has already chosen, the days it fires on
are better entries than other days in that stock and month - about +10 pp win
rate on stocks it has never seen (+18 pp on the curated 337, which overstate
it: they are today's large companies, and dips in eventual winners recover).
It beats the simple rule chosen in advance (`low bb_position`); it is level
with the best single rule in hindsight (`low range60_position`). It does not
say which stock to buy.

PROBABILITIES
-------------
The classifier outputs a probability of hitting the target first. Whether
that number may be shown is decided by the calibration check below, on the
fold models' out-of-sample predictions: among fired signals, the mean
predicted probability must sit within 3 pp of the actual hit rate on BOTH the
core held-out years and the fresh names. Otherwise only the binary signal and
the measured hit rate are shown.
"""

import os
import sys
import copy
import time
import hashlib
import warnings

import numpy as np
import pandas as pd
import joblib

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import measure_v83 as V83
import sweep_options_v85 as V85
import out_of_universe_v86 as V86

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="divide by zero")

MODEL_FILE = os.path.join("class_model", "entry_model_v88.joblib")
CFG = V86.COMBO_CFG
FEATURES, DIRECTIONS = V85.features_of(CFG)
SEED = V85.SEED
N_SEEDS = 5
QUANTILE = V83.QUANTILE
WINDOW_DAYS = V82.WINDOW_DAYS
MIN_ROWS = V83.MIN_ROWS
PLATEAU_WARN = 0.03
CALIB_TOL = 0.03
HELD_YEARS = V86.HELD_YEARS
OPPONENT = V86.OPPONENT
COMBO = V86.COMBO_NAME
MODE = "probability"


# =============================================================================
# THE MODEL - the exact constructor the measurement used
# =============================================================================
def build_model(seed):
    return V85.build(CFG, seed)


def predict(models, X):
    return np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in models],
                   axis=0)


def fit_ensemble(frame, n_seeds=N_SEEDS, seed=SEED):
    models = []
    for s in range(n_seeds):
        m = build_model(seed + s)
        m.fit(frame[FEATURES], frame["label"], verbose=False)
        models.append(m)
    return models


# =============================================================================
# SERVING FEATURES - no label needed, so the latest bar can be scored
# =============================================================================
def serving_frame(names, cache, horizon="MID", step=E64.STEP, as_of=None,
                  chunk=200):
    """
    Every 5th bar on the training grid (bar 260, 265, ...) plus the latest
    bar, for every name, with the training set's eligibility filter and
    feature code. Rows on the grid are bit-identical to training rows.
    """
    parts = []
    cut = pd.Timestamp(as_of) if as_of else None
    for i in range(0, len(names), chunk):
        ch, rows = names[i:i + chunk], []
        for t in ch:
            df = E64.P2.load_prices(t, cache)
            if df is None:
                continue
            if cut is not None:
                df = df[df.index <= cut]
            if len(df) <= E64.MIN_HISTORY:
                continue
            pan = E64.feature_panel(df)
            pos = np.unique(np.r_[np.arange(E64.MIN_HISTORY, len(df), step),
                                  len(df) - 1])
            sub = pan.iloc[pos]
            ok = np.isfinite(sub[E64.FEATURES + E64.PATTERN_FEATURES]
                             .to_numpy(float)).all(axis=1)
            entry = df["Close"].to_numpy(float)[pos]
            atr = sub["_atr_abs"].to_numpy(float)
            ok &= np.isfinite(atr) & (atr > 0) & (entry > 0)
            ok &= np.array([bool(k) and bool(E64.compute_levels(
                e, float(a), horizon).get("valid"))
                for e, a, k in zip(entry, atr, ok)], bool)
            if not ok.any():
                continue
            f = pd.DataFrame({"ticker": t, "date": df.index[pos][ok],
                              "is_latest": (pos == len(df) - 1)[ok],
                              "entry": entry[ok]})
            for c in V85.BASE_FEATS + ["log_dollar_vol"]:
                f[c] = sub[c].to_numpy(float)[ok]
            rows.append(f)
        if not rows:
            continue
        F = pd.concat(rows, ignore_index=True)
        F["date_n"] = pd.DatetimeIndex(F["date"]).normalize()
        F = F.merge(V86._short(cache, ch), on=["ticker", "date_n"],
                    how="left")
        parts.append(F)
    if not parts:
        return pd.DataFrame()
    F = pd.concat(parts, ignore_index=True)
    F["date"] = pd.DatetimeIndex(F["date"])
    return F.sort_values(["date", "ticker"], kind="mergesort") \
        .reset_index(drop=True)


# =============================================================================
# CHECKS
# =============================================================================
def measured_parity(core, wf_combo_core, year, n_seeds, horizon):
    """Refit one fold with THIS file's builder; it must reproduce V86's cached
    fold predictions for the combo exactly."""
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    start = pd.Timestamp(f"{year}-01-01")
    dc = pd.DatetimeIndex(core["date"])
    tr = core[dc < start - embargo]
    te = core[core["year"] == year]
    p = predict(fit_ensemble(tr, n_seeds), te)
    ref = wf_combo_core[0].set_index("row_id")["p"]
    q = ref.reindex(te["row_id"].to_numpy()).to_numpy(float)
    ok = np.isfinite(q)
    return float(np.max(np.abs(p[ok] - q[ok]))) if ok.any() else np.nan, \
        int(ok.sum())


def serving_parity(core, F):
    m = core[["ticker", "date"] + FEATURES].merge(
        F[["ticker", "date"] + FEATURES], on=["ticker", "date"],
        suffixes=("_t", "_s"))
    worst = 0.0
    for c in FEATURES:
        a, b = m[c + "_t"].to_numpy(float), m[c + "_s"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any():
            return np.inf, len(m)
        if both.any():
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
    return worst, len(m)


def calibration_check(U, te, refs, years, label):
    """Out-of-sample fold probabilities against outcomes."""
    fire = V85.model_fire(U, te, refs)
    rid = te["row_id"].to_numpy(int)
    p = te["p"].to_numpy(float)
    y = U["label"].to_numpy(float)[rid]
    yr = U["year"].to_numpy()[rid]
    k = np.isin(yr, years) & np.isfinite(p)
    p, y, f = p[k], y[k], fire[rid][k]
    base = y.mean()
    brier = np.mean((p - y) ** 2)
    brier0 = np.mean((base - y) ** 2)
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    rel = [{"Universe": label, "Bucket": f"quintile {int(b) + 1}",
            "Rows": int((q == b).sum()), "Mean p": p[q == b].mean(),
            "Win rate": y[q == b].mean()} for b in np.unique(q)]
    rel.append({"Universe": label, "Bucket": "FIRED (top 1%)",
                "Rows": int(f.sum()),
                "Mean p": p[f].mean() if f.any() else np.nan,
                "Win rate": y[f].mean() if f.any() else np.nan})
    gap = (p[f].mean() - y[f].mean()) if f.any() else np.nan
    return pd.DataFrame(rel), {"universe": label, "brier": float(brier),
                               "brier_base": float(brier0),
                               "skill": float(1 - brier / brier0),
                               "fired_gap": float(gap),
                               "fired_n": int(f.sum())}


def _records(path):
    return pd.read_csv(path).to_dict("records") if os.path.exists(path) \
        else []


def measured_tables(v85_dir, v86_dir, v87_dir):
    m = {"v85_all_arms": _records(f"{v85_dir}/all_arms.csv"),
         "v85_seed_check": _records(f"{v85_dir}/seed_check.csv"),
         "v86_side_by_side": _records(f"{v86_dir}/side_by_side.csv"),
         "v86_arms": _records(f"{v86_dir}/arms_all_universes.csv"),
         "v86_regimes": _records(f"{v86_dir}/regimes.csv"),
         "v86_per_year": _records(f"{v86_dir}/per_year_primary.csv"),
         "v86_coverage": _records(f"{v86_dir}/coverage.csv"),
         "v87_blocks": _records(f"{v87_dir}/block_bootstrap.csv"),
         "v87_autocorrelation": _records(f"{v87_dir}/autocorrelation.csv")}
    for k, p in (("v86_verdict", f"{v86_dir}/verdict.txt"),
                 ("v87_verdict", f"{v87_dir}/verdict.txt"),
                 ("v86_preregistration", f"{v86_dir}/preregistration.txt"),
                 ("v87_preregistration", f"{v87_dir}/preregistration.txt")):
        m[k] = open(p).read() if os.path.exists(p) else ""
    return m


# =============================================================================
# TRAIN AND PACKAGE
# =============================================================================
def train(model_in="entry_model_v81.joblib", core_cache=None,
          big_cache="price_cache_v68", n_seeds=N_SEEDS, chunk=250,
          parity_year=2020, run_calibration=True, v85_dir="thesis_tables_v85",
          v86_dir="thesis_tables_v86", v87_dir="thesis_tables_v87",
          verbose=True):
    t0 = time.time()
    prov81 = M81.load_model(model_in)["provenance"]
    core_cache = core_cache or prov81["price_cache"]
    horizon, step = prov81["horizon"], prov81["step"]

    v87 = os.path.join(v87_dir, "verdict.txt")
    verdict = open(v87).read() if os.path.exists(v87) else ""
    if "CONFIRMED" not in verdict or "NOT CONFIRMED" in verdict:
        print("  *** V87 did not confirm the combo (or has not been run) - "
              "V88 packages it anyway, but the claim does not stand")

    # ---- data: V86's caches (core, and fresh + fold predictions if present)
    print("  [1] TRAINING DATA - the core names, exactly as V86 built them")
    wf = fresh = None
    try:
        import block_bootstrap_v87 as V87
        core, fresh, wf, n_core = V87.prepare(
            model_in, core_cache, big_cache, n_seeds, chunk, V86.LIQ_PCT,
            V86.DUP_CORR, verbose=verbose)
    except Exception as e:
        print(f"      V86 caches not usable ({e}); building the core only - "
              f"the measured-model parity and calibration checks are skipped")
        core, _ = V86.build_core(prov81, core_cache, verbose=verbose)
    print(f"      {len(core):,} candidate entries, "
          f"{core['ticker'].nunique()} names, {core['date'].min().date()} to "
          f"{core['date'].max().date()}")

    # ---- parity with the measured fold models -----------------------------
    parity = {}
    if wf is not None:
        print(f"\n  [2] MEASURED-MODEL PARITY - refit the {parity_year} fold "
              f"with this file's builder")
        diff, n = measured_parity(core, wf["combo"]["core"], parity_year,
                                  n_seeds, horizon)
        parity["measured"] = {"year": parity_year, "rows": n,
                              "max_abs_diff": diff}
        print(f"      {n:,} rows, max |difference| vs V86's fold "
              f"predictions {diff:.2e}  "
              f"{('IDENTICAL' if diff == 0 else 'IDENTICAL to float32 precision') if diff < 1e-6 else '*** MISMATCH ***'}")

    # ---- serving fit -------------------------------------------------------
    print(f"\n  [3] SERVING FIT - {n_seeds} seeds on all {len(core):,} core rows")
    models = fit_ensemble(core, n_seeds)

    # ---- serving features == training features ------------------------------
    print(f"\n  [4] SERVING-FEATURE PARITY - score()'s feature builder vs the "
          f"training rows")
    names = V86.cache_names(core_cache)
    F = serving_frame(names, core_cache, horizon, step)
    worst, n = serving_parity(core, F)
    parity["serving"] = {"shared_rows": n, "train_rows": int(len(core)),
                         "max_abs_diff": worst}
    print(f"      {n:,} of {len(core):,} training rows rebuilt by score(); "
          f"max |difference| {worst:.2e}  "
          f"{'IDENTICAL' if worst < 1e-12 and n == len(core) else '*** MISMATCH ***'}")
    lat = F[F["is_latest"]]
    print(f"      score() also builds the latest bar for "
          f"{len(lat):,} names (latest date {lat['date'].max().date()}); "
          f"the training builder's latest row is "
          f"{core['date'].max().date()}")

    # ---- calibration -----------------------------------------------------------
    calib_rows, calib_stats = [], []
    if run_calibration and wf is not None:
        print(f"\n  [5] CALIBRATION - out-of-sample fold probabilities, "
              f"{HELD_YEARS[0]}-{HELD_YEARS[-1]}")
        Uc, wc = V87.universe(core, np.ones(len(core), bool), wf, "core")
        Uf, wfr = V87.universe(fresh, fresh["liquid"].to_numpy(bool), wf,
                               "fresh")
        for lab, U, w in ((f"core {n_core}", Uc, wc),
                          ("fresh, liquid", Uf, wfr)):
            rel, st = calibration_check(U, *w["combo"], HELD_YEARS, lab)
            calib_rows.append(rel)
            calib_stats.append(st)
        rel = pd.concat(calib_rows, ignore_index=True)
        print("      " + rel.to_string(index=False, float_format=lambda v:
                                       f"{v:.3f}").replace("\n", "\n      "))
        for st in calib_stats:
            print(f"      {st['universe']:<14} Brier skill {st['skill']:+.4f}; "
                  f"fired: mean p minus hit rate {st['fired_gap']:+.3f} on "
                  f"{st['fired_n']:,} signals")
    display_p = bool(calib_stats) and all(
        np.isfinite(s["fired_gap"]) and abs(s["fired_gap"]) <= CALIB_TOL
        for s in calib_stats)

    rule_str = f"rolling_top_{QUANTILE:.4f}_win{WINDOW_DAYS}_M_strict_v88"
    model = {
        "boosters": models, "mode": MODE, "features": FEATURES,
        "directions": DIRECTIONS, "config": copy.deepcopy(CFG),
        "feature_sig": hashlib.sha256("|".join(
            [",".join(FEATURES), horizon, prov81["label_mode"], rule_str,
             MODE]).encode()).hexdigest()[:16],
        "rule": {"kind": "rolling_percentile", "quantile": QUANTILE,
                 "window_days": WINDOW_DAYS, "refit_freq": "M",
                 "min_window_rows": MIN_ROWS, "strict": True},
        "display_probability": display_p,
        "calibration": {"stats": calib_stats,
                        "table": (pd.concat(calib_rows).to_dict("records")
                                  if calib_rows else []),
                        "tolerance": CALIB_TOL},
        "parity": parity,
        "measured": measured_tables(v85_dir, v86_dir, v87_dir),
        "provenance": {"n_tickers": int(core["ticker"].nunique()),
                       "n_rows": int(len(core)),
                       "from": str(core["date"].min().date()),
                       "trained_through": str(core["date"].max().date()),
                       "horizon": horizon, "label_mode": prov81["label_mode"],
                       "step": step, "n_seeds": n_seeds, "seed": SEED,
                       "price_cache": core_cache, "version": "v88",
                       "selected_in": "sweep_options_v85.py (2017-2021)",
                       "confirmed_in": "out_of_universe_v86.py, "
                                       "block_bootstrap_v87.py"}}
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
    p, me = model["provenance"], model["measured"]
    W = 100
    print("=" * W)
    print("  ENTRY TIMING MODEL v88 - final")
    print("=" * W)
    print(f"  trained on {p['n_tickers']} names, {p['n_rows']:,} candidates, "
          f"{p['from']} to {p['trained_through']}")
    print(f"  XGBoost classifier (P[target before stop]), shallow, "
          f"{p['n_seeds']} seeds, every feature monotone 'low is good':")
    print(f"    {', '.join(model['features'])}")
    print(f"  fires when the score is STRICTLY above the 99th percentile of the "
          f"universe's trailing 252 days")
    print(f"  selected in {p['selected_in']}; confirmed in {p['confirmed_in']}")

    sb = pd.DataFrame(me.get("v86_side_by_side", []))
    if len(sb):
        print(f"\n  AGAINST {OPPONENT} (the rule fixed in advance), "
              f"{HELD_YEARS[0]}-{HELD_YEARS[-1]} - paired, pp of win rate")
        cols = [c for c in ["Universe", "Arm", "Signals", "Arm lift",
                            "bb lift", "vs bb", "lo", "hi", "lo (Bonf k=2)"]
                if c in sb.columns]
        print(sb[cols].to_string(index=False, float_format=lambda v:
                                 f"{v:+.2f}"))
    bl = pd.DataFrame(me.get("v87_blocks", []))
    if len(bl):
        b = bl[(bl["Universe"].str.startswith("fresh, liquid"))
               & (bl["Arm"] == COMBO)]
        if len(b):
            print(f"\n  ROBUST TO OVERLAPPING TRADES (V87, fresh liquid names): "
                  f"lower bounds by block length")
            print(b[["Years", "Block (months)", "Diff", "90% lo", "90% hi",
                     "Bonf lo"]].to_string(index=False,
                                           float_format=lambda v:
                                           f"{v:+.2f}"))
    for k in ("v87_verdict",):
        if me.get(k):
            print("\n" + me[k].rstrip())

    print(f"\n  PROBABILITY DISPLAY: "
          + ("allowed - among fired signals the fold models' mean probability "
             "was within 3 pp of the hit rate on both universes"
             if model.get("display_probability") else
             "NOT allowed - show the binary signal and the measured hit rate"))
    print("\n  WHAT IT CLAIMS: given a stock and month already chosen, the days "
          "it fires on are better")
    print("  entries than other days in that stock and month. It beats the rule "
          "chosen in advance")
    print("  on stocks it never saw; it is level with the best single rule in "
          "hindsight. It does not")
    print("  say which stock to buy. Expect about +10 pp win rate over a random "
          "day on an arbitrary")
    print("  stock - not the +18 pp measured on the curated 337.")


# =============================================================================
# SERVE
# =============================================================================
def score(model, tickers, price_cache=None, as_of=None):
    """
    Scores the LATEST bar of each requested ticker against the current
    month's cut, set from the whole cache's trailing 252 days. The cache is
    the reference universe: the 337 core names reproduce the measured core
    setting; the expanded cache reproduces V86's fresh-name setting.
    """
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    names = V86.cache_names(price_cache)
    if len(names) < 20:
        raise RuntimeError("the rolling cut needs a reference universe")
    F = serving_frame(names, price_cache, p["horizon"], p["step"], as_of)
    F["p"] = predict(model["boosters"], F)
    F["cut"] = V83.rolling_cut(F["date"], F["p"].to_numpy(float),
                               r["quantile"], r["min_window_rows"])
    F["fires"] = V83.apply_rule(F["p"].to_numpy(float),
                                F["cut"].to_numpy(float),
                                strict=r.get("strict", True))

    di = pd.DatetimeIndex(F["date"])
    last = pd.Period(di.max(), freq="M").to_timestamp()
    win = F.loc[(di >= last - V82._win()) & (di < last), "p"] \
        .to_numpy(float)
    cur = F.loc[di >= last, "cut"]
    c = float(cur.iloc[0]) if len(cur) and np.isfinite(cur.iloc[0]) \
        else np.nan
    share = float((win >= c).mean()) if win.size and np.isfinite(c) \
        else np.nan
    guard = {"share_at_or_above_cut": share,
             "plateau_warning": bool(np.isfinite(share)
                                     and share > PLATEAU_WARN)}

    out = F[F["ticker"].isin(set(tickers))]
    lastfire = (out[out["fires"]].groupby("ticker")["date"].max()
                .rename("last_fire"))
    latest = out[out["is_latest"]][["ticker", "date", "p", "cut", "fires"]] \
        .merge(lastfire, on="ticker", how="left").reset_index(drop=True)
    if not model.get("display_probability"):
        latest = latest.rename(columns={"p": "score"})
    return {"latest": latest, "guard": guard,
            "all": out[["ticker", "date", "is_latest", "p", "cut",
                        "fires"]]}


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN    = "entry_model_v81.joblib"  # only for provenance (cache,
                                                # horizon, step, label mode)
    RUN_CORE_CACHE  = None              # None = the cache V81 was trained on
    RUN_BIG_CACHE   = "price_cache_v68" # for the parity/calibration checks
    RUN_N_SEEDS     = 5
    RUN_CHUNK       = 250               # must equal V86's (cache key)
    RUN_PARITY_YEAR = 2020              # which fold to refit for the parity test
    RUN_CALIBRATION = True
    RUN_MODEL_OUT   = MODEL_FILE    # class_model/entry_model_v88.joblib
    RUN_SERVE_CACHE = None              # None = core; "price_cache_v68" = wide
    RUN_SCORE_THESE = ["AMD", "NVDA", "AAPL"]
    # -------------------------------------------------------------------------

    model = train(model_in=RUN_MODEL_IN, core_cache=RUN_CORE_CACHE,
                  big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS,
                  chunk=RUN_CHUNK, parity_year=RUN_PARITY_YEAR,
                  run_calibration=RUN_CALIBRATION)
    save_model(model, RUN_MODEL_OUT)
    print()
    describe(model)
    if RUN_SCORE_THESE:
        res = score(model, RUN_SCORE_THESE, RUN_SERVE_CACHE)
        g = res["guard"]
        print(f"\n  SERVE: {', '.join(RUN_SCORE_THESE)}   plateau guard: "
              f"{g['share_at_or_above_cut']:.3%} of the trailing window at or "
              f"above the cut"
              + ("  *** PLATEAU - retrain before trusting signals ***"
                 if g["plateau_warning"] else "  (healthy ~1%)"))
        print(res["latest"].to_string(index=False))
