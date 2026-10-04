"""
V84 - THE FINAL ENTRY MODEL.

Nothing is retrained. This packages what V81-V83 established:

  SERVED MODEL   V81's boosters, unchanged: expectancy XGBoost, 5 seeds,
                 monotone "low is good" on six mean-reversion features chosen on
                 2008-2016 only. The served model is the measured model (V81's
                 parity test).
  RULE           fire when the score is STRICTLY above the 99th percentile of
                 the scores the same model gives the universe over the trailing
                 252 days, recomputed monthly. Strict, because V83 found early
                 fold models with a plateau at the cut - 53% of 2010's
                 candidates sat on one value - and `>=` admitted the whole
                 plateau. The served model has no plateau (share >= own p99 is
                 0.010 in every year), so strict costs it nothing; it is kept as
                 the rule so a future retrain cannot flood silently.
  MEASUREMENT    read from V83's strict-rule tables, which score each fold's
                 trailing window with that fold's own model - the way score()
                 runs in production.
  GUARD          score() reports the share of the trailing window at or above
                 the current cut. Around 1% is healthy; well above it means the
                 model has developed a plateau and its signals should not be
                 trusted until it is retrained.
"""

import os
import sys
import copy

import numpy as np
import pandas as pd
import joblib

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v83 as V83

MODEL_IN = "entry_model_v81.joblib"
MODEL_OUT = "entry_model_v84.joblib"
TABLES_DIR = "thesis_tables_v83"
PLATEAU_WARN = 0.03


# =============================================================================
# PACKAGE
# =============================================================================
def finalize(model_in=MODEL_IN, tables_dir=TABLES_DIR, model_out=MODEL_OUT):
    m = copy.copy(M81.load_model(model_in))
    m["rule"] = dict(m["rule"], strict=True)

    def rd(name):
        p = os.path.join(tables_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    held, allyrs = rd("arms_held-out.csv"), rd("arms_all.csv")
    if held is None or allyrs is None:
        raise RuntimeError(f"run measure_v83.py first - {tables_dir}/ "
                           f"has no arms tables")
    strict = lambda t: t[t["Rule"] == ">"].drop(columns="Rule") \
        if t is not None and "Rule" in t.columns else t
    cov = rd("coverage.csv")
    m["measured"] = {
        "held_out": strict(held).to_dict("records"),
        "all_years": strict(allyrs).to_dict("records"),
        "coverage": strict(cov).to_dict("records") if cov is not None else [],
        "regime": (rd("regime_strict.csv").to_dict("records")
                   if rd("regime_strict.csv") is not None else []),
        "degeneracy": (rd("degeneracy.csv").to_dict("records")
                       if rd("degeneracy.csv") is not None else []),
        "calibration": m.get("measured", {}).get("calibration"),
        "source": "measure_v83.py, strict rule, fold-consistent cut"}
    m["provenance"] = dict(m["provenance"], version="v84",
                           rule="strict >, rolling 1% of trailing 252d")
    joblib.dump(m, model_out)
    print(f"  saved {model_out} (served boosters identical to {model_in})")
    return m


def load_model(path=MODEL_OUT):
    return joblib.load(path)


# =============================================================================
# DESCRIBE
# =============================================================================
def describe(model):
    p, me = model["provenance"], model["measured"]
    print("=" * 96)
    print("  ENTRY TIMING MODEL v84 - final")
    print("=" * 96)
    print(f"  {p['n_tickers']} tickers, {p['from']} to {p['trained_through']}"
          f" | {model['mode']} XGBoost, {p['n_seeds']} seeds, monotone")
    print(f"  features (all 'low is good'): {', '.join(model['features'])}")
    print(f"  fires when the score is STRICTLY above the 99th percentile of "
          f"the trailing 252 days")

    cols = ["Arm", "Signals", "Prec lift pp", "90% CI lo", "90% CI hi",
            "ExpR lift", "Cond yrs", "Sign yrs", "Sign p", "Worst year pp"]
    want = ["V81", "composite v2", "low bb_position", "low rsi_14",
            "V76 (28 feats)", "random #1", "random #2", "random #3"]
    for title, key in (("HELD-OUT 2017-2026", "held_out"),
                       ("ALL YEARS 2008-2026", "all_years")):
        t = pd.DataFrame(me[key])
        if t.empty:
            continue
        t = t[t["Arm"].isin(want)].copy()
        t["Arm"] = pd.Categorical(t["Arm"], want, ordered=True)
        t = t.sort_values("Arm")
        print(f"\n  {title} - against random entry in the same name and "
              f"month")
        print(t[[c for c in cols if c in t.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

    reg = pd.DataFrame(me.get("regime", []))
    if len(reg):
        print("\n  BY MARKET REGIME at entry (full-universe breadth)")
        print(reg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  WHAT IT CLAIMS: given a name and a month the user has already "
          "chosen, the days")
    print("  it fires on are better entries than other days in that name and "
          "month. It does")
    print("  NOT say which stock to buy, and it is level with - not better "
          "than - the best")
    print("  single mean-reversion rules. Show the binary signal and the "
          "measured hit rate;")
    print("  its aggregate calibration is zero, so do not show a per-stock "
          "probability.")


# =============================================================================
# SERVE
# =============================================================================
def score(model, tickers, price_cache=None, as_of=None):
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    if len(names) < 20:
        raise RuntimeError("the rolling cut needs a reference universe")
    d = E64.build_dataset(names, price_cache, p["horizon"], p["step"],
                          verbose=False)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    if as_of:
        d = d[pd.DatetimeIndex(d["date"]) <= pd.Timestamp(as_of)]
    d = d.reset_index(drop=True)
    d["p"] = M81.predict(model["boosters"], d[model["features"]],
                         model["mode"])
    cut = V83.rolling_cut(d["date"], d["p"].to_numpy(float),
                          r["quantile"], r["min_window_rows"])
    d["cut"] = cut
    d["fires"] = V83.apply_rule(d["p"].to_numpy(float), cut,
                                strict=r.get("strict", True))

    # plateau guard on the most recent month
    di = pd.DatetimeIndex(d["date"])
    last = pd.Period(di.max(), freq="M").to_timestamp()
    win = d[(di >= last - V83.V82._win()) & (di < last)]["p"].to_numpy(float)
    cur = d.loc[di >= last, "cut"]
    c = float(cur.iloc[0]) if len(cur) and np.isfinite(cur.iloc[0]) \
        else np.nan
    share = float((win >= c).mean()) if win.size and np.isfinite(c) \
        else np.nan
    guard = {"share_at_or_above_cut": share,
             "plateau_warning": bool(np.isfinite(share)
                                     and share > PLATEAU_WARN)}

    out = d[d["ticker"].isin(set(tickers))]
    latest = (out.sort_values("date").groupby("ticker").tail(1)
                 [["ticker", "date", "p", "cut", "fires"]]
                 .reset_index(drop=True))
    return {"latest": latest, "guard": guard,
            "all": out[["ticker", "date", "p", "cut", "fires"]]}


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN    = "entry_model_v81.joblib"
    RUN_TABLES_DIR  = "thesis_tables_v83"     # written by measure_v83.py
    RUN_MODEL_OUT   = "entry_model_v84.joblib"
    RUN_PRICE_CACHE = None
    RUN_SCORE_THESE = ["AMD", "NVDA", "AAPL"]
    # -------------------------------------------------------------------------

    model = finalize(RUN_MODEL_IN, RUN_TABLES_DIR, RUN_MODEL_OUT)
    print()
    describe(model)
    if RUN_SCORE_THESE:
        res = score(model, RUN_SCORE_THESE, RUN_PRICE_CACHE)
        g = res["guard"]
        print(f"\n  SERVE: {', '.join(RUN_SCORE_THESE)}   plateau guard: "
              f"{g['share_at_or_above_cut']:.3%} of the trailing window at or "
              f"above the cut"
              + ("  *** PLATEAU - retrain before trusting signals ***"
                 if g["plateau_warning"] else "  (healthy ~1%)"))
        print(res["latest"].to_string(index=False))
