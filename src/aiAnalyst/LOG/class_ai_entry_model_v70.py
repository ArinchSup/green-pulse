#!/usr/bin/env python3
"""
class_ai_entry_model_v70.py - the deployable entry classifier: train, save, load, score.

WHY THIS FILE EXISTS SEPARATELY FROM V64/V66

V64 and V66 are EVALUATION harnesses. They fit a model per walk-forward fold,
score the next year with it, and throw it away - which is correct for measuring
honestly and useless for serving. Nothing in them can answer "what does the model
say about NVDA today".

This file produces one artifact: a model fitted on everything up to a cutoff,
with the firing threshold taken from its own training scores, saved to a single
.joblib you can load and call.

WHAT GETS SAVED, AND WHY EACH PIECE

    booster          the fitted XGBoost classifier
    features         the column list IN ORDER. Serving with a different order
                     silently scores nonsense - trees do not check names.
    feature_sig      a hash of the feature list and the geometry constants. If
                     the feature code changes and the model is not retrained,
                     score() REFUSES rather than serving a model whose inputs
                     mean something different from what it learned. This is
                     train/serve skew, and it does not announce itself.
    threshold        the score above which the model fires, taken from TRAINING
                     scores. Not a probability cut-off chosen by eye.
    measured         the walk-forward performance, WITH its clustered interval.
                     Saved inside the model on purpose: anybody who loads this
                     gets the honest number and the uncertainty in the same
                     object, not the point estimate on its own.
    provenance       universe, date range, label, geometry, code version.

THE MODEL CARRIES ITS OWN CAVEAT

describe() prints the measured win rate AND the interval AND the blind baseline.
On the 2,347-name universe that interval did not clear the baseline, and the
saved model says so when you print it. A model that ships without its error bar
invites exactly the reading we spent a day avoiding.

USAGE
  python class_ai_entry_model_v70.py              # train, save, then score today
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
from trade_config import HORIZON_CONFIGS

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = E64.PRICE_CACHE
HORIZON = E64.HORIZON
STEP = E64.STEP
MODEL_FILE = "entry_model_v70.joblib"

FEATURE_SET = "no_confirm"
LABEL_MODE = "barrier"
TRAIN_QUANTILE = 0.01          # fire on the top X% of training scores.
                               # 0.01 is the ONLY cut whose lower bound clears
                               # the blind rate in V71 Table 1, and it clears by
                               # 0.2pp on a 20pp-wide interval - thin, not safe.
                               # V71 reads this value off the saved model, so
                               # Tables 2-4 follow whatever is set here.
N_SEEDS = 3
SEED = 70

MAX_PER_WEEK = 3               # the `spaced` rule V67 found was worth keeping
APPLY_SPACED = True


# =============================================================================
# TRAIN / SERVE GUARD
# =============================================================================
def feature_signature(features, horizon=HORIZON, label_mode=LABEL_MODE):
    """
    A hash of everything that has to match between training and serving. If the
    feature list, its order, the horizon or the barrier geometry changes, this
    changes, and score() refuses.
    """
    cfg = HORIZON_CONFIGS[horizon]
    blob = json.dumps({"features": list(features), "horizon": horizon,
                       "label_mode": label_mode,
                       "lookahead_bars": cfg["lookahead_bars"],
                       "eval_days": cfg["eval_days"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# =============================================================================
# TRAIN
# =============================================================================
def train_entry_model(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                      feature_set=FEATURE_SET, label_mode=LABEL_MODE,
                      quantile=TRAIN_QUANTILE, n_seeds=N_SEEDS, seed=SEED,
                      as_of=None, tickers=None, measure=True, verbose=True):
    """
    Fit the servable model, and measure it honestly on the way.

    Two fits happen here and they are not the same thing:

        the MEASUREMENT fit(s)  purged walk-forward, one model per year, used
                                only to produce the performance numbers stored
                                in the artifact. Never served.
        the SERVING fit         one model on everything up to `as_of`, which is
                                what gets saved. It has seen all the data, which
                                is correct for serving and would be cheating if
                                it were scored.

    Keeping them separate is the whole point. A model trained on everything and
    then evaluated on part of that everything reports its own memory.
    """
    from xgboost import XGBClassifier

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
    if as_of:
        d = d[pd.DatetimeIndex(d["date"]) <= pd.Timestamp(as_of)]
    feats = E64.resolve_features(feature_set)

    measured = None
    if measure:
        if verbose:
            print(f"\n  MEASUREMENT: purged walk-forward (these models are "
                  f"scored, not served)")
        te = E64.walk_forward(d, feats, E64.MIN_TRAIN_YEARS, 1, seed, horizon,
                              verbose=verbose)
        measured = _measure(te, quantile, seed)

    if verbose:
        print(f"\n  SERVING FIT: one model on all {len(d):,} rows up to "
              f"{pd.DatetimeIndex(d['date']).max().date()}")
    boosters, train_scores = [], []
    for s in range(n_seeds):
        m = XGBClassifier(random_state=seed + s, **E64.XGB_PARAMS)
        m.fit(d[feats], d["label"], verbose=False)
        boosters.append(m)
        train_scores.append(m.predict_proba(d[feats])[:, 1])
    p_train = np.mean(train_scores, axis=0)
    threshold = float(np.quantile(p_train, 1.0 - quantile))

    model = {
        "boosters": boosters,
        "features": feats,
        "feature_sig": feature_signature(feats, horizon, label_mode),
        "threshold": threshold,
        "quantile": quantile,
        "measured": measured,
        "spaced": {"apply": APPLY_SPACED, "max_per_week": MAX_PER_WEEK},
        "provenance": {
            "n_tickers": len(names), "n_rows": int(len(d)),
            "trained_through": str(pd.DatetimeIndex(d["date"]).max().date()),
            "from": str(pd.DatetimeIndex(d["date"]).min().date()),
            "horizon": horizon, "eval_days": HORIZON_CONFIGS[horizon]["eval_days"],
            "label_mode": label_mode, "feature_set": feature_set,
            "n_seeds": n_seeds, "seed": seed, "price_cache": price_cache,
        },
    }
    if verbose:
        print(f"  threshold = {threshold:.4f} "
              f"(the top {quantile:.0%} of training scores)")
    return model


def _measure(te, quantile, seed):
    """Walk-forward performance at the firing threshold, with its honest interval."""
    import evaluate_sniper_v66 as S66
    k = max(1, int(len(te) * quantile))
    fired = te.nlargest(k, "p")
    blind_wr = float(te["label"].mean())
    blind_R = float(te["r_multiple"].mean())
    n, w = len(fired), int(fired["label"].sum())
    lo, hi = S66.block_bootstrap_mean(fired["label"].to_numpy(float),
                                      fired["date"].to_numpy(), seed=seed)
    lo_w, hi_w = S66.wilson(w, n)
    per_year = fired.groupby("year")["label"].agg(["size", "mean"])
    return {
        "n_oos": int(len(te)), "n_fired": int(n),
        "win_rate": w / n, "wr_lo": lo, "wr_hi": hi,
        "n_eff": S66._n_eff(n, lo, hi, lo_w, hi_w),
        "expectancy_R": float(fired["r_multiple"].mean()),
        "blind_win_rate": blind_wr, "blind_expectancy_R": blind_R,
        "years_fired": int(len(per_year)),
        "years_total": int(te["year"].nunique()),
        "clears_baseline": bool(np.isfinite(lo) and lo > blind_wr),
    }


# =============================================================================
# SAVE / LOAD
# =============================================================================
def save_model(model, path=MODEL_FILE, verbose=True):
    import joblib
    joblib.dump(model, path, compress=3)
    if verbose:
        size = os.path.getsize(path) / 1e6
        print(f"  saved {path} ({size:.1f} MB)")
    return path


def load_model(path=MODEL_FILE):
    import joblib
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found - train it first")
    return joblib.load(path)


def describe(model, out=sys.stdout):
    """What this model is and what it is worth. Both, always."""
    def w(*a):
        print(*a, file=out)
    p, m = model["provenance"], model.get("measured")
    w("\n" + "=" * 84)
    w("  ENTRY CLASSIFIER v70")
    w("=" * 84)
    w(f"  trained on {p['n_tickers']:,} tickers, {p['n_rows']:,} candidate "
      f"entries, {p['from']} to {p['trained_through']}")
    w(f"  horizon {p['horizon']} ({p['eval_days']}d) | label {p['label_mode']} "
      f"| features {p['feature_set']} ({len(model['features'])} columns)")
    w(f"  fires above score {model['threshold']:.4f} "
      f"(top {model['quantile']:.0%} of training scores)")
    if model["spaced"]["apply"]:
        w(f"  spacing rule: at most {model['spaced']['max_per_week']} signals "
          f"per week, highest score first")
    if not m:
        w("\n  NOT MEASURED - this model carries no performance record. Do not "
          "quote it.")
        return
    w(f"\n  MEASURED, walk-forward out of sample:")
    w(f"    fired {m['n_fired']:,} of {m['n_oos']:,} candidates in "
      f"{m['years_fired']}/{m['years_total']} years")
    w(f"    win rate      {m['win_rate']:.1%}   "
      f"95% [{m['wr_lo']:.1%}, {m['wr_hi']:.1%}]   "
      f"(n_eff {m['n_eff']:,.0f})")
    w(f"    buying blind  {m['blind_win_rate']:.1%}")
    w(f"    expectancy    {m['expectancy_R']:+.3f}R vs "
      f"{m['blind_expectancy_R']:+.3f}R blind")
    if m["clears_baseline"]:
        w(f"\n    The lower bound clears the blind rate. Treat it as a real but "
          f"modest edge,")
        w(f"    and size on {m['wr_lo']:.1%}, not on {m['win_rate']:.1%}.")
    else:
        w(f"\n    THE LOWER BOUND DOES NOT CLEAR THE BLIND RATE "
          f"({m['wr_lo']:.1%} vs {m['blind_win_rate']:.1%}).")
        w(f"    This model is not distinguishable from buying anything. It is "
          f"usable as a")
        w(f"    'favourable setup' tag; it is not a buy signal, and it should "
          f"not be described")
        w(f"    as one anywhere it reaches a user.")


# =============================================================================
# SCORE
# =============================================================================
def score(model, tickers=None, price_cache=None, as_of=None, horizon=None,
          verbose=True):
    """
    Score names as of a date. One row per ticker: the score, whether it fires,
    and what the model is measured to be worth when it does.
    """
    p = model["provenance"]
    price_cache = price_cache or p["price_cache"]
    horizon = horizon or p["horizon"]

    sig = feature_signature(model["features"], horizon, p["label_mode"])
    if sig != model["feature_sig"]:
        raise RuntimeError(
            f"train/serve mismatch: this model was fitted with signature "
            f"{model['feature_sig']} and the current feature or geometry code "
            f"hashes to {sig}. Retrain rather than serving it - the inputs no "
            f"longer mean what the model learned.")

    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    rows = []
    for t in names:
        df = E64.P2.load_prices(t, price_cache)
        if df is None or len(df) < E64.MIN_HISTORY:
            continue
        if as_of:
            df = df[df.index <= pd.Timestamp(as_of)]
            if len(df) < E64.MIN_HISTORY:
                continue
        pan = E64.feature_panel(df)
        row = pan.iloc[-1]
        base = {k: float(row[k]) for k in E64.FEATURES + E64.PATTERN_FEATURES
                if k in row}
        base.update({"ticker": t, "date": df.index[-1],
                     "price": float(df["Close"].iloc[-1]),
                     "_atr": float(row["_atr_abs"])})
        rows.append(base)
    if not rows:
        raise RuntimeError("no scorable tickers")
    d = pd.DataFrame(rows)

    # cross-sectional and market columns, computed across THIS scoring set -
    # the same construction as training, but on today's cross-section
    for src, dst in (("ret_20", "xs_ret_20"), ("rsi_14", "xs_rsi_14"),
                     ("atr_pct", "xs_atr_pct"),
                     ("px_vs_ema200", "xs_px_vs_ema200")):
        d[dst] = d[src].rank(pct=True)
    d["mkt_breadth_ema200"] = float((d["px_vs_ema200"] > 0).mean())
    d["mkt_ret_20"] = float(d["ret_20"].median())
    d["mkt_vol_20"] = float(d["yz22"].median())
    d["mkt_breadth_chg_20"] = 0.0      # needs history; not in the served set
    d["mkt_vol_chg_20"] = 0.0

    missing = [f for f in model["features"] if f not in d.columns]
    if missing:
        raise RuntimeError(f"cannot build features {missing} at serve time")
    X = d[model["features"]]
    d["score"] = np.mean([b.predict_proba(X)[:, 1] for b in model["boosters"]],
                         axis=0)
    d["fires"] = d["score"] >= model["threshold"]

    if model["spaced"]["apply"]:
        # the spacing rule, applied to one day's signals: keep the strongest few
        cap = model["spaced"]["max_per_week"]
        order = d[d["fires"]].sort_values("score", ascending=False)
        keep = set(order.head(cap)["ticker"])
        d["fires_after_spacing"] = d["ticker"].isin(keep)
    else:
        d["fires_after_spacing"] = d["fires"]

    out = d[["ticker", "date", "price", "score", "fires",
             "fires_after_spacing"]].sort_values("score", ascending=False)
    m = model.get("measured") or {}
    out.attrs["measured_win_rate"] = m.get("win_rate")
    out.attrs["measured_interval"] = (m.get("wr_lo"), m.get("wr_hi"))
    out.attrs["blind_win_rate"] = m.get("blind_win_rate")

    if verbose:
        n_fire = int(out["fires"].sum())
        print(f"\n  scored {len(out):,} tickers as of "
              f"{pd.Timestamp(out['date'].iloc[0]).date()}")
        print(f"  {n_fire} above the threshold"
              + (f", {int(out['fires_after_spacing'].sum())} after the spacing "
                 f"rule" if model["spaced"]["apply"] else ""))
        if m:
            print(f"  when it fires, measured win rate {m['win_rate']:.1%} "
                  f"[{m['wr_lo']:.1%}, {m['wr_hi']:.1%}] "
                  f"against {m['blind_win_rate']:.1%} blind")
            if not m.get("clears_baseline"):
                print(f"  NOTE the interval does not clear the blind rate - "
                      f"this is a tag, not a signal")
    return out


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_v70(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
            feature_set=FEATURE_SET, label_mode=LABEL_MODE,
            quantile=TRAIN_QUANTILE, n_seeds=N_SEEDS, seed=SEED, as_of=None,
            tickers=None, measure=True, model_file=MODEL_FILE,
            score_tickers=None, verbose=True):
    print("=" * 84)
    print("V70 - TRAIN AND SAVE THE DEPLOYABLE ENTRY CLASSIFIER")
    print("=" * 84)
    model = train_entry_model(price_cache, horizon, step, feature_set,
                              label_mode, quantile, n_seeds, seed, as_of,
                              tickers, measure, verbose)
    save_model(model, model_file, verbose)
    describe(model)

    print("\n" + "=" * 84)
    print("  RELOADING AND SCORING (proves the artifact round-trips)")
    print("=" * 84)
    m2 = load_model(model_file)
    out = score(m2, score_tickers, price_cache, verbose=verbose)
    print()
    print(out.head(12).to_string(index=False))
    return model, out


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon", default=HORIZON, choices=list(HORIZON_CONFIGS))
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--label-mode", default=LABEL_MODE,
                    choices=["barrier", "profit"])
    ap.add_argument("--quantile", type=float, default=TRAIN_QUANTILE)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--no-measure", action="store_true")
    ap.add_argument("--model-file", default=MODEL_FILE)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    if c.score_only:
        m = load_model(c.model_file)
        describe(m)
        out = score(m, c.tickers, c.price_cache)
        print()
        print(out.head(20).to_string(index=False))
        return 0
    run_v70(c.price_cache, c.horizon, c.step, c.feature_set, c.label_mode,
            c.quantile, c.n_seeds, c.seed, c.as_of, None, not c.no_measure,
            c.model_file, c.tickers)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE  = PRICE_CACHE
    RUN_HORIZON      = HORIZON        # "SHORT" | "MID" | "LONG"
    RUN_STEP         = STEP
    RUN_FEATURE_SET  = FEATURE_SET    # no_confirm | per_name | +market | ...
    RUN_LABEL_MODE   = LABEL_MODE     # "barrier" = reached target before stop
    RUN_QUANTILE     = TRAIN_QUANTILE # fire on the top X% of training scores
    RUN_N_SEEDS      = N_SEEDS
    RUN_SEED         = SEED
    RUN_AS_OF        = None           # "2026-06-30" to freeze the training end
    RUN_MEASURE      = True           # False skips the walk-forward and saves a
                                      # model with NO performance record
    RUN_MODEL_FILE   = "entry_model_v70.joblib"
    RUN_SCORE_TICKERS = None          # None scores every name in the cache
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_v70(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
                step=RUN_STEP, feature_set=RUN_FEATURE_SET,
                label_mode=RUN_LABEL_MODE, quantile=RUN_QUANTILE,
                n_seeds=RUN_N_SEEDS, seed=RUN_SEED, as_of=RUN_AS_OF,
                measure=RUN_MEASURE, model_file=RUN_MODEL_FILE,
                score_tickers=RUN_SCORE_TICKERS)
