"""
V92 - TEST THE SERVED V88 FILE ITSELF.

Every earlier result came from the walk-forward copies of the model, because
the served file (entry_model_v88.joblib) was trained on every year up to
2026-06-30 and cannot be graded on those years. There are two honest ways to
test the file itself, and this runs both:

  TEST 1 - LIVE PERIOD (out of time: the cleanest test)
      Every trading day AFTER the model's training ended, for every stock in
      the core 337 and the unseen stocks. Each day is scored by the V88 file
      exactly as entry_recommender_v90.py does it (same features, same cut
      from the same reference universe). Every ENTER becomes a trade:
          TARGET / STOP / EXPIRED (60 trading days), or OPEN if the data
          ends first - open trades are marked at the last close.
      Small today (about three months after training end), and it grows
      every time the price caches are refreshed and this is run again.
      Consecutive ENTER days in one stock are separate rows but one
      episode, so the stock count is shown beside the signal count.

  TEST 2 - UNSEEN STOCKS (out of universe, in time)
      The V88 file on the 2,024 stocks it never trained on, 2017-2026, set up
      exactly as V86 tested the walk-forward copies (liquid rows, each
      universe's own trailing cut). The file has never seen these stocks,
      but it has seen these YEARS through other stocks, so this is weaker
      than test 1. It answers: does the file behave like the model that was
      measured? (V86's walk-forward copies: +10.07 pp.)

Both tests compare ENTER trades with a random OTHER day in the same stock and
month - what the model is meant to beat - and with entering on any day.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v83 as V83
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import class_ai_entry_model_v88 as M88
import entry_recommender_v90 as R90
import random_backtest_v91 as V91
from trade_config import HORIZON_CONFIGS, compute_levels

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_PREFIX = "test_v88_v92"


# =============================================================================
# TRADES THAT MAY STILL BE OPEN
# =============================================================================
def simulate_open(px, ticker, date, horizon="MID"):
    """Like V91.simulate, but a trade whose window runs past the data is
    OPEN and marked at the last close instead of being dropped."""
    a = px.get(ticker)
    hit = np.flatnonzero(a["idx"].normalize()
                         == pd.Timestamp(date).normalize())
    if not hit.size:
        return None
    pos = int(hit[0])
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    entry, atr = a["cl"][pos], float(a["atr"][pos])
    lv = compute_levels(entry, atr, horizon)
    entry = float(entry)
    if not lv.get("valid"):
        return None
    avail = len(a["cl"]) - pos - 1
    if avail <= 0:
        return {"entry": entry, "stop": lv["stop"], "target": lv["target"],
                "result": "OPEN", "exit_date": a["idx"][pos].date(),
                "exit": entry, "days_held": 0, "return_pct": 0.0}
    res = E64.triple_barrier(a["hi"][pos + 1:], a["lo"][pos + 1:],
                             a["cl"][pos + 1:], entry, lv["stop"],
                             lv["target"], min(n_bars, avail))
    label, r, held, outcome = res
    if outcome == "win":
        result, exit_px = "TARGET", lv["target"]
    elif outcome == "loss":
        result, exit_px = "STOP", lv["stop"]
    else:
        result = "EXPIRED" if avail >= n_bars else "OPEN"
        exit_px = float(a["cl"][pos + held])
    return {"entry": entry, "stop": lv["stop"], "target": lv["target"],
            "result": result, "exit_date": a["idx"][pos + held].date(),
            "exit": float(exit_px), "days_held": int(held),
            "return_pct": (exit_px / entry - 1) * 100}


def summarize(name, t, cost_pct, position):
    if not len(t):
        return {"Group": name, "Trades": 0}
    ret = t["return_pct"].to_numpy(float) - cost_pct
    done = t["result"] != "OPEN"
    se = ret.std(ddof=1) / np.sqrt(len(ret)) if len(ret) > 1 else np.nan
    out = {"Group": name, "Trades": len(t),
           "Stocks": t["ticker"].nunique() if "ticker" in t else np.nan,
           "Open": int((~done).sum()),
           "Target %": (t["result"] == "TARGET").mean() * 100,
           "Stop %": (t["result"] == "STOP").mean() * 100,
           "Expired %": (t["result"] == "EXPIRED").mean() * 100,
           "Open %": (~done).mean() * 100,
           "Avg return %": ret.mean(),
           "90% range": f"[{ret.mean() - 1.645 * se:+.2f}, "
                        f"{ret.mean() + 1.645 * se:+.2f}]"
           if np.isfinite(se) else "",
           "Avg return % (closed only)": ret[done].mean() if done.any()
           else np.nan,
           "Net profit $": (ret / 100 * position).sum()}
    return out


def show(T, title, W=104):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    D = T.copy()
    fm = {"Target %": "{:.1f}", "Stop %": "{:.1f}", "Expired %": "{:.1f}",
          "Open %": "{:.1f}", "Avg return %": "{:+.2f}",
          "Avg return % (closed only)": "{:+.2f}",
          "Net profit $": "{:+,.0f}", "Lift pp": "{:+.2f}",
          "ExpR": "{:+.3f}", "vs rule pp": "{:+.2f}", "Fire rate": "{:.2%}"}
    for c, f in fm.items():
        if c in D:
            D[c] = [f.format(v) if isinstance(v, (int, float, np.floating))
                    and np.isfinite(v) else "" for v in D[c]]
    print(D.to_string(index=False))


def control_trades(px, rows_by_key, signals, rng, horizon, sim):
    """For each signal, a random OTHER eligible day of the same stock-month."""
    out = []
    for r in signals.itertuples(index=False):
        k = (r.ticker, pd.Period(pd.Timestamp(r.date), freq="M"))
        days = [d for d in rows_by_key.get(k, []) if d != pd.Timestamp(r.date)]
        if not days:
            continue
        d = days[rng.integers(len(days))]
        s = sim(px, r.ticker, d, horizon)
        if s:
            out.append({"ticker": r.ticker, "date": d, **s})
    return pd.DataFrame(out)


# =============================================================================
# TEST 1 - LIVE PERIOD
# =============================================================================
def recent_frame(names, cache, start, end, horizon, step=1, verbose=True):
    """V88's serving features for every bar in [start, end] - same eligibility
    and feature code as M88.serving_frame, restricted to the window."""
    parts, t0 = [], time.time()
    for i, t in enumerate(names, 1):
        df = E64.P2.load_prices(t, cache)
        if df is None or len(df) <= E64.MIN_HISTORY:
            continue
        idx = pd.DatetimeIndex(df.index).normalize()
        pos = np.flatnonzero((idx >= start) & (idx <= end))
        pos = pos[pos >= E64.MIN_HISTORY][::step]
        if not pos.size:
            continue
        pan = E64.feature_panel(df)
        sub = pan.iloc[pos]
        ok = np.isfinite(sub[E64.FEATURES + E64.PATTERN_FEATURES]
                         .to_numpy(float)).all(axis=1)
        entry = df["Close"].to_numpy(float)[pos]
        atr = sub["_atr_abs"].to_numpy(float)
        ok &= np.isfinite(atr) & (atr > 0) & (entry > 0)
        ok &= np.array([bool(k) and bool(compute_levels(
            e, float(a), horizon).get("valid"))
            for e, a, k in zip(entry, atr, ok)], bool)
        if not ok.any():
            continue
        f = pd.DataFrame({"ticker": t, "date": df.index[pos][ok],
                          "entry": entry[ok]})
        for c in V85.BASE_FEATS + ["log_dollar_vol"]:
            f[c] = sub[c].to_numpy(float)[ok]
        f["date_n"] = pd.DatetimeIndex(f["date"]).normalize()
        f = f.merge(V86._short(cache, [t]), on=["ticker", "date_n"],
                    how="left")
        parts.append(f)
        if verbose and i % 250 == 0:
            print(f"      [{i}/{len(names)}] {time.time() - t0:4.0f}s")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def live_test(rec, core, groups, start, end, cost_pct, position, seed,
              n_any=3000, verbose=True):
    horizon = rec.horizon
    rng = np.random.default_rng(seed)
    rows, trades = [], []
    for gname, (names, cache, liquid_only) in groups.items():
        if not names:
            continue
        last = "latest" if end.year >= 2100 else str(end.date())
        print(f"    {gname}: building features for {len(names)} stocks, "
              f"{start.date()} to {last} ...")
        F = recent_frame(names, cache, start, end, horizon, verbose=verbose)
        if not len(F):
            print(f"    {gname}: no data in the window")
            continue
        if liquid_only:
            fl = V86.liquidity_floor(core, F["date"])
            F = F[F["log_dollar_vol"].to_numpy(float) >= fl] \
                .reset_index(drop=True)
        F["score"] = M88.predict(rec.model["boosters"], F)
        F["cut"] = [rec.cut_for(d)["cut"] for d in F["date"]]
        F["enter"] = V83.apply_rule(F["score"].to_numpy(float),
                                    F["cut"].to_numpy(float),
                                    strict=rec.rule.get("strict", True))
        S = F[F["enter"]]
        px = V91.Prices(cache)
        sim = [simulate_open(px, r.ticker, r.date, horizon)
               for r in S.itertuples(index=False)]
        St = pd.DataFrame([{"ticker": r.ticker, "date": r.date, **s}
                           for r, s in zip(S.itertuples(index=False), sim)
                           if s])
        key = {}
        for t, d in zip(F["ticker"], pd.DatetimeIndex(F["date"])):
            key.setdefault((t, pd.Period(d, freq="M")), []).append(d)
        Ct = control_trades(px, key, S, rng, horizon, simulate_open)
        A = F.sample(min(n_any, len(F)), random_state=seed)
        At = pd.DataFrame([{"ticker": r.ticker, "date": r.date, **s}
                           for r in A.itertuples(index=False)
                           for s in [simulate_open(px, r.ticker, r.date,
                                                   horizon)] if s])
        rate = len(S) / len(F)
        for lab, T in ((f"{gname}: ENTER days", St),
                       (f"{gname}: same stock & month, other day", Ct),
                       (f"{gname}: any day (sample)", At)):
            row = summarize(lab, T, cost_pct, position)
            if lab.endswith("ENTER days"):
                row["Fire rate"] = rate
            rows.append(row)
        if len(St):
            St.insert(0, "group", gname)
            trades.append(St)
        print(f"    {gname}: {len(F):,} stock-days scored, {len(S)} ENTER "
              f"({rate:.2%}) in {S['ticker'].nunique()} stocks")
    return pd.DataFrame(rows), (pd.concat(trades, ignore_index=True)
                                if trades else pd.DataFrame())


# =============================================================================
# TEST 2 - UNSEEN STOCKS
# =============================================================================
def unseen_test(rec, fresh, wf, years, big_cache, cost_pct, position, seed,
                n_boot=4000, n_any=3000):
    horizon = rec.horizon
    U, w = V87.universe(fresh, fresh["liquid"].to_numpy(bool), wf, "fresh")
    p = np.asarray(M88.predict(rec.model["boosters"], U), float)
    cut = V83.rolling_cut(U["date"], p)
    f88 = V83.apply_rule(p, cut, strict=True)
    ffold = V85.model_fire(U, *w["combo"])
    s_bb = -U["bb_position"].to_numpy(float)
    frule = V83.apply_rule(s_bb, V83.rolling_cut(U["date"], s_bb),
                           strict=True)
    ev = V85.Evaluator(U, n_boot=n_boot)
    yrs = list(range(years[0], years[1] + 1))
    rows = []
    for lab, f in (("V88 file", f88), ("walk-forward copies (V86)", ffold),
                   ("rule: low bb_position", frule)):
        s = ev.summary(f, yrs)
        row = {"Model": lab, "Signals": s["signals"], "Lift pp": s["lift"],
               "90% CI": f"[{s['lo']:+.2f}, {s['hi']:+.2f}]",
               "ExpR": s["expR"], "Years beating own pool": s["cond"]}
        row["vs rule pp"], row["vs rule CI"] = np.nan, ""
        if lab != "rule: low bb_position":
            pr = ev.paired(f, frule, yrs)
            if pr:
                row["vs rule pp"] = pr["diff"]
                row["vs rule CI"] = f"[{pr['lo']:+.2f}, {pr['hi']:+.2f}]"
        rows.append(row)
    held = np.isin(U["year"].to_numpy(), yrs)
    both = (f88 & ffold & held).sum()
    agree = {"v88": int((f88 & held).sum()), "fold": int((ffold & held)
                                                         .sum()),
             "both": int(both)}

    # trades
    rng = np.random.default_rng(seed)
    px = V91.Prices(big_cache)
    S = U[f88 & held]
    St = pd.DataFrame([{"ticker": r.ticker, "date": r.date, **s}
                       for r in S.itertuples(index=False)
                       for s in [V91.simulate(px, r.ticker, r.date,
                                              horizon)] if s])
    key = {}
    for t, d in zip(U["ticker"], pd.DatetimeIndex(U["date"])):
        key.setdefault((t, pd.Period(d, freq="M")), []).append(d)
    Ct = control_trades(px, key, S, rng, horizon, V91.simulate)
    A = U[held].sample(min(n_any, int(held.sum())), random_state=seed)
    At = pd.DataFrame([{"ticker": r.ticker, "date": r.date, **s}
                       for r in A.itertuples(index=False)
                       for s in [V91.simulate(px, r.ticker, r.date,
                                              horizon)] if s])
    tr = pd.DataFrame([summarize(lab, T, cost_pct, position) for lab, T in
                       (("V88 ENTER trades", St),
                        ("same stock & month, other day", Ct),
                        ("any day (sample)", At))])
    tr = tr.drop(columns=["Open", "Open %", "Avg return % (closed only)"],
                 errors="ignore")
    return pd.DataFrame(rows), tr, agree, St


# =============================================================================
# RUNNER
# =============================================================================
def run_v92(tests=("live", "unseen"), model_file="entry_model_v88.joblib",
            live_start=None, live_end=None, live_groups=("core", "fresh"),
            unseen_years=(2017, 2026), position=10000, cost_pct=0.10,
            seed=11, model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", chunk=250, n_seeds=5,
            out_prefix=OUT_PREFIX, verbose=True):
    t0 = time.time()
    W = 104
    print("=" * W)
    print("V92 - TESTING THE SERVED V88 FILE")
    print("=" * W)
    rec = R90.EntryRecommender("MID", model_file=model_file,
                               universe_cache=core_cache, verbose=verbose)
    print(f"  {rec.model_file}: trained through {rec.trained_through.date()}"
          f", reference universe {rec.universe_cache}")
    print("  reading V86's data (cached) ...")
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=False)

    if "live" in tests:
        start = pd.Timestamp(live_start) if live_start else \
            rec.trained_through + pd.Timedelta(days=1)
        end = pd.Timestamp(live_end) if live_end else pd.Timestamp("2100-01-01")
        print(f"\n  TEST 1 - LIVE PERIOD: every day from {start.date()} "
              f"(after training ended)")
        core_names = V86.cache_names(core_cache)
        fresh_names = sorted(fresh["ticker"].unique())
        groups = {}
        if "core" in live_groups:
            groups[f"core {n_core}"] = (core_names, core_cache, False)
        if "fresh" in live_groups:
            groups["unseen, liquid"] = (fresh_names, big_cache, True)
        T1, trades1 = live_test(rec, core, groups, start, end, cost_pct,
                                position, seed, verbose=verbose)
        if len(T1):
            show(T1, f"TEST 1 - LIVE PERIOD from {start.date()}: the V88 file "
                     f"on days after its training ended (returns after "
                     f"{cost_pct:.2f}% cost)")
            print("  OPEN = still inside its 60-day window when the data ends, "
                  "marked at the last close. Rerun after refreshing prices "
                  "to extend the test.")
            if len(trades1):
                ends = pd.to_datetime(trades1["date"])
                print(f"  ENTER days run from {ends.min().date()} to "
                      f"{ends.max().date()}")
            T1.to_csv(f"{out_prefix}_live_summary.csv", index=False)
            if len(trades1):
                trades1.to_csv(f"{out_prefix}_live_trades.csv", index=False)

    if "unseen" in tests:
        print(f"\n  TEST 2 - UNSEEN STOCKS, {unseen_years[0]}-"
              f"{unseen_years[1]} (out of universe, in time) ...")
        L, T2, agree, trades2 = unseen_test(rec, fresh, wf, unseen_years,
                                            big_cache, cost_pct, position,
                                            seed)
        show(L, f"TEST 2a - LIFT over other days in the same stock and month, "
                f"unseen liquid stocks {unseen_years[0]}-{unseen_years[1]}")
        print(f"  ENTER days: V88 file {agree['v88']:,}, walk-forward copies "
              f"{agree['fold']:,}, both {agree['both']:,} - the file was "
              f"trained on more years, so it will not fire on exactly the "
              f"same days")
        show(T2, "TEST 2b - TRADES from the V88 file's ENTER days, unseen "
                 f"liquid stocks (returns after {cost_pct:.2f}% cost)")
        L.to_csv(f"{out_prefix}_unseen_lift.csv", index=False)
        T2.to_csv(f"{out_prefix}_unseen_trades_summary.csv", index=False)
        trades2.to_csv(f"{out_prefix}_unseen_trades.csv", index=False)

    print(f"\n  saved {out_prefix}_*.csv | {(time.time() - t0) / 60:.1f} min")


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TESTS        = ("live", "unseen")  # run one or both
    RUN_MODEL_FILE   = "entry_model_v88.joblib"
    RUN_LIVE_START   = None             # None = day after V88's training end
    RUN_LIVE_END     = None             # None = latest prices
    RUN_LIVE_GROUPS  = ("core", "fresh")  # stocks to scan in the live test
    RUN_UNSEEN_YEARS = (2017, 2026)
    RUN_POSITION     = 10000            # $ per trade
    RUN_COST_PCT     = 0.10             # round-trip cost, % of the position
    RUN_SEED         = 11
    RUN_BIG_CACHE    = "price_cache_v68"
    RUN_CHUNK        = 250              # same as V86 (cache key)
    # -------------------------------------------------------------------------

    run_v92(tests=RUN_TESTS, model_file=RUN_MODEL_FILE,
            live_start=RUN_LIVE_START, live_end=RUN_LIVE_END,
            live_groups=RUN_LIVE_GROUPS, unseen_years=RUN_UNSEEN_YEARS,
            position=RUN_POSITION, cost_pct=RUN_COST_PCT, seed=RUN_SEED,
            big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK)
