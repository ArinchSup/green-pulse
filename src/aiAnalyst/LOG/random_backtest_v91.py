"""
V91 - RANDOM-PICK BACKTEST OF THE ENTRY MODEL.

Draw N random (stock, date) picks. For each one ask the model "enter today?".
When it says ENTER, buy at that day's close with the model's stop and target,
then walk forward bar by bar:

    TARGET   the high reaches the target first        -> exit at the target
    STOP     the low reaches the stop first           -> exit at the stop
             (both on the same bar counts as the stop, on purpose)
    EXPIRED  neither within 60 trading days           -> exit at that close

and add up the profit after costs.

WHICH MODEL ANSWERS
-------------------
For past dates the answer comes from the WALK-FORWARD version of the final
model: each year is scored by a model trained only on earlier years, with the
same code as V88 (V88's parity test). The served V88 file was trained on all
these years, so asking it about them would grade it on data it has already
seen. Everything is read from V86's caches - nothing is retrained.

Random dates are drawn from the days the model scores: every 5th trading day
per stock, from 2008 (the first walk-forward year).

TWO COMPARISONS, SO THE NUMBERS MEAN SOMETHING
----------------------------------------------
  every pick       buy every random pick, ignoring the signal
  same-month day   for each ENTER, buy the same stock on a random OTHER day
                   of the same month - what the model is meant to beat

The model fires on about 1% of days, so N = 3,000 picks gives only about 30
trades; their average carries a wide interval. For a steadier picture raise
RUN_N (30,000 picks gives roughly 300 trades).
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
from trade_config import HORIZON_CONFIGS, compute_levels

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="divide by zero")


# =============================================================================
# TRADE SIMULATION - straight from the prices
# =============================================================================
class Prices:
    """Loads each stock once; keeps its arrays and ATR series."""

    def __init__(self, cache):
        self.cache, self._d = cache, {}

    def get(self, t):
        if t not in self._d:
            df = E64.P2.load_prices(t, self.cache)
            pan = E64.feature_panel(df)
            self._d[t] = {"idx": pd.DatetimeIndex(df.index),
                          "hi": df["High"].to_numpy(float),
                          "lo": df["Low"].to_numpy(float),
                          "cl": df["Close"].to_numpy(float),
                          "atr": pan["_atr_abs"].to_numpy(float)}
        return self._d[t]


def simulate(px, ticker, date, horizon="MID"):
    """One trade: buy at the close of `date`, stop and target from
    trade_config, walk forward up to the horizon's bar limit."""
    a = px.get(ticker)
    pos = a["idx"].get_indexer([pd.Timestamp(date)])[0]
    if pos < 0:
        hit = np.flatnonzero(a["idx"].normalize()
                             == pd.Timestamp(date).normalize())
        if not hit.size:
            return None
        pos = int(hit[0])
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    entry, atr = a["cl"][pos], a["atr"][pos]
    lv = compute_levels(entry, float(atr), horizon)   # numpy entry, as in training
    if not lv.get("valid"):
        return None
    res = E64.triple_barrier(a["hi"][pos + 1:], a["lo"][pos + 1:],
                             a["cl"][pos + 1:], entry, lv["stop"],
                             lv["target"], n_bars)
    if res is None:
        return None
    label, r, held, outcome = res
    if pos + held >= len(a["cl"]):
        return None
    exit_px = {"win": lv["target"], "loss": lv["stop"]}.get(
        outcome, a["cl"][pos + held])
    return {"entry": float(entry), "stop": lv["stop"],
            "target": lv["target"], "result": {"win": "TARGET",
                                               "loss": "STOP"}.get(
                outcome, "EXPIRED"),
            "exit_date": a["idx"][pos + held].date(), "exit": float(exit_px),
            "days_held": int(held), "return_pct": (exit_px / entry - 1) * 100,
            "r_multiple": float(r), "label": int(label)}


def summarize(name, t, position, cost_pct):
    if not len(t):
        return {"Trades": name, "Count": 0}
    ret = t["return_pct"].to_numpy(float) - cost_pct
    se = ret.std(ddof=1) / np.sqrt(len(ret)) if len(ret) > 1 else np.nan
    gains, losses = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return {"Trades": name, "Count": len(t),
            "Target %": (t["result"] == "TARGET").mean() * 100,
            "Stop %": (t["result"] == "STOP").mean() * 100,
            "Expired %": (t["result"] == "EXPIRED").mean() * 100,
            "Avg return %": ret.mean(),
            "90% range of avg": f"[{ret.mean() - 1.645 * se:+.2f}, "
                                f"{ret.mean() + 1.645 * se:+.2f}]"
            if np.isfinite(se) else "",
            "Median %": float(np.median(ret)),
            "Avg days held": t["days_held"].mean(),
            "Profit factor": gains / losses if losses > 0 else np.inf,
            "Net profit $": (ret / 100 * position).sum()}


# =============================================================================
# RUNNER
# =============================================================================
def run_v91(n=3000, seed=7, years=(2008, 2026), universe="core",
            position=10000, cost_pct=0.10, show_trades=40,
            save_csv="random_backtest_v91.csv",
            model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, verbose=False):
    t0 = time.time()
    W = 100
    print("=" * W)
    print(f"V91 - RANDOM-PICK BACKTEST: {n:,} random (stock, date) picks, "
          f"{universe} universe, {years[0]}-{years[1]}")
    print("=" * W)
    print("  reading V86's data and walk-forward models (cached) ...")
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    horizon = prov["horizon"]
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    if universe == "core":
        U, w = V87.universe(core, np.ones(len(core), bool), wf, "core")
        cache = core_cache
    elif universe == "fresh":
        U, w = V87.universe(fresh, fresh["liquid"].to_numpy(bool), wf,
                            "fresh")
        cache = big_cache
    else:
        raise ValueError('universe must be "core" or "fresh"')
    fires = V85.model_fire(U, *w["combo"])
    U = U.assign(signal=np.where(fires, "ENTER", "WAIT"))
    yr = U["year"].to_numpy()
    pool = U[(yr >= years[0]) & (yr <= years[1])]
    print(f"  {pool['ticker'].nunique()} stocks, {len(pool):,} scorable days "
          f"in {years[0]}-{years[1]}; the model says ENTER on "
          f"{(pool['signal'] == 'ENTER').mean():.2%} of them")

    # ---- random picks: a random stock, then a random day of that stock ----
    rng = np.random.default_rng(seed)
    by_t = {t: g.index.to_numpy() for t, g in pool.groupby("ticker")}
    names = sorted(by_t)
    n = min(n, len(pool))
    picks, seen = [], set()
    while len(picks) < n:
        t = names[rng.integers(len(names))]
        i = int(by_t[t][rng.integers(len(by_t[t]))])
        if i not in seen:
            seen.add(i)
            picks.append(i)
    P = pool.loc[picks, ["ticker", "date", "year", "signal", "label",
                         "r_multiple"]].rename(
        columns={"label": "label_ds", "r_multiple": "r_ds"})

    # ---- simulate every pick (the 'every pick' comparison needs them all) --
    px = Prices(cache)
    sims = [simulate(px, r.ticker, r.date, horizon)
            for r in P.itertuples(index=False)]
    ok = np.array([s is not None for s in sims])
    S = pd.DataFrame([s for s in sims if s is not None])
    P = pd.concat([P[ok].reset_index(drop=True), S], axis=1)
    same = float(np.mean((P["label_ds"].to_numpy() == P["label"].to_numpy())
                         & (np.abs(P["r_ds"].to_numpy()
                                   - P["r_multiple"].to_numpy()) < 1e-6)))
    print(f"  simulated {len(P):,} trades from the prices; outcome and R match "
          f"the model's training labels on {same:.1%} of them")

    # ---- same stock, same month, random other day, for each ENTER --------
    per = pd.PeriodIndex(pd.DatetimeIndex(pool["date"]), freq="M")
    key = pool["ticker"].astype(str).to_numpy() + "|" + \
        per.astype(str).to_numpy()
    grp = pd.Series(pool.index.to_numpy()).groupby(key).apply(list).to_dict()
    ent = P[P["signal"] == "ENTER"]
    ctrl = []
    for r in ent.itertuples(index=False):
        k = f"{r.ticker}|{pd.Period(pd.Timestamp(r.date), freq='M')}"
        others = [i for i in grp.get(k, [])
                  if pool.at[i, "date"] != r.date]
        if not others:
            continue
        j = others[rng.integers(len(others))]
        s = simulate(px, pool.at[j, "ticker"], pool.at[j, "date"], horizon)
        if s:
            ctrl.append(s)
    C = pd.DataFrame(ctrl)

    # ---- results --------------------------------------------------------------
    rows = [summarize("ENTER signals (traded)", ent, position, cost_pct),
            summarize("every pick, ignoring the signal", P, position,
                      cost_pct),
            summarize("same stock & month, random other day", C, position,
                      cost_pct)]
    T = pd.DataFrame(rows)
    print(f"\n  {len(P):,} picks -> {len(ent)} ENTER, "
          f"{len(P) - len(ent):,} WAIT   |   position ${position:,.0f} per "
          f"trade, cost {cost_pct:.2f}% round trip")
    print("\n" + "-" * W)
    print("  RESULTS (returns after costs)")
    print("-" * W)
    fm = {"Target %": "{:.1f}", "Stop %": "{:.1f}", "Expired %": "{:.1f}",
          "Avg return %": "{:+.2f}", "Median %": "{:+.2f}",
          "Avg days held": "{:.0f}", "Profit factor": "{:.2f}",
          "Net profit $": "{:+,.0f}"}
    D = T.copy()
    for c, f in fm.items():
        if c in D:
            D[c] = [f.format(v) if isinstance(v, (int, float, np.floating))
                    and np.isfinite(v) else "" for v in D[c]]
    print(D.to_string(index=False))
    if len(ent) and len(C):
        print(f"\n  ENTER vs a random other day in the same stock and month: "
              f"{T.loc[0, 'Avg return %'] - T.loc[2, 'Avg return %']:+.2f} pp "
              f"average return per trade, "
              f"{T.loc[0, 'Target %'] - T.loc[2, 'Target %']:+.1f} pp target "
              f"hit rate")
    if len(ent) < 100:
        print(f"\n  Only {len(ent)} trades: the average moves a lot from one "
              f"seed to the next. Raise RUN_N (about 1 trade per 100 picks) "
              f"for a steadier number.")

    if show_trades and len(ent):
        cols = ["ticker", "date", "entry", "stop", "target", "result",
                "exit_date", "exit", "days_held", "return_pct"]
        L = ent[cols].copy()
        L["date"] = pd.DatetimeIndex(L["date"]).date
        L["return_pct"] = L["return_pct"] - cost_pct
        L["profit $"] = L["return_pct"] / 100 * position
        L = L.sort_values("date")
        print("\n" + "-" * W)
        print(f"  ENTER TRADES ({min(show_trades, len(L))} of {len(L)}, by "
              f"date; return after costs)")
        print("-" * W)
        print(L.head(show_trades).to_string(
            index=False, float_format=lambda v: f"{v:,.2f}"))

    if save_csv:
        out = P.drop(columns=["label", "label_ds", "r_ds"],
                     errors="ignore").copy()
        out["date"] = pd.DatetimeIndex(out["date"]).date
        out["return_after_cost_pct"] = out["return_pct"] - cost_pct
        out["traded"] = out["signal"] == "ENTER"
        out.to_csv(save_csv, index=False)
        print(f"\n  saved every pick to {save_csv}")
    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return T, P, C


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_N           = 3000              # random (stock, date) picks
    RUN_SEED        = 7                 # change it to draw a different sample
    RUN_YEARS       = (2008, 2026)      # walk-forward years to draw dates from
    RUN_UNIVERSE    = "core"            # "core" = the 337; "fresh" = unseen
    RUN_POSITION    = 10000             # $ per trade
    RUN_COST_PCT    = 0.10              # round-trip cost, % of the position
    RUN_SHOW_TRADES = 40                # list this many ENTER trades
    RUN_SAVE_CSV    = "random_backtest_v91.csv"
    RUN_BIG_CACHE   = "price_cache_v68" # needed to read V86's caches
    RUN_CHUNK       = 250               # same as V86 (cache key)
    # -------------------------------------------------------------------------

    run_v91(n=RUN_N, seed=RUN_SEED, years=RUN_YEARS, universe=RUN_UNIVERSE,
            position=RUN_POSITION, cost_pct=RUN_COST_PCT,
            show_trades=RUN_SHOW_TRADES, save_csv=RUN_SAVE_CSV,
            big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK)
