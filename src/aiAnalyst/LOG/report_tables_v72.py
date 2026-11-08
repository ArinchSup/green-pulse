"""
V72 - RESULTS TABLES, with an exposure-matched random control.

WHY THIS FILE EXISTS
--------------------
V71's Table 4 printed:

    Model signal                   856.43%   2,550 trades   0.57 R/trade
    Random signal (median of 20)   941.79%   2,550 trades
    Random signal (best draw)    1,381.29%   2,550 trades

and concluded "a backtest number that luck reaches is not evidence of a model".
The conclusion is sound in spirit but the comparison behind it is confounded,
and the confound is visible in the same table: the model carries 1,445 total R
across those 2,550 trades and the random arm carries about 640 (2,550 x the
0.25 R blind expectancy). The model has 2.26x the trade quality and still loses
on compounded return. Trade quality cannot produce that. Something else did.

That something else is TIME IN MARKET. V71's control assigned a uniform random
score to all 254,979 candidates and took the top 1%. A uniform draw spreads
evenly over every month in the sample, so the control is invested in nearly all
~198 months. The model's signals are clustered - it fired in 16 of 17 years and
concentrates inside those - so it is invested in far fewer. Returns compound per
ACTIVE month, so:

    control:  (1 + 0.05 * 0.25) ^ ~198  ~  11.7x
    model:    (1 + 0.05 * 0.57) ^ ~80   ~   9.6x

The arm with the worse trades wins because it compounds more often. That is a
statement about exposure, not about selection, and it is not what Table 4 claims
to measure.

WHAT CHANGED
------------
1. `matched_random_backtest` fires in exactly the months the model fired in,
   with exactly the number of signals the model had in each of those months,
   drawn at random from that month's candidates. Time in market is then
   identical across arms and the only remaining difference is WHICH trades were
   picked - which is the thing the control is supposed to isolate.
2. Every arm reports `Months in market` and `Total R`, so an exposure gap can
   never again hide inside a percentage.
3. V71's unmatched control is kept as a clearly labelled row. The gap between
   the two controls IS the exposure effect, measured.
4. `power_note` prints the effective sample size the measured lift would need
   before its lower bound could clear the blind rate. When a result fails, this
   separates "the effect is absent" from "the sample cannot resolve it" - which
   are different findings and point at different next steps.

Everything else - the dataset rebuild, Tables 1 and 2, buy & hold, monthly
compounding with a SHARED risk budget - is imported unchanged from V71.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

import report_tables_v71 as R71
import class_ai_entry_model_v70 as M70
import class_ai_entry_v64 as E64

from report_tables_v71 import (
    classification_table, confusion_table, buy_and_hold, _md,
    THRESHOLDS, RISK_TIERS, START_CAPITAL, OUT_DIR, MODEL_FILE, SEED,
)

N_DRAWS = 20


# =============================================================================
# COMPOUNDING - one implementation, used by every arm
# =============================================================================
def _compound(t, risk_frac, start=START_CAPITAL):
    """
    Compound an already-selected set of trades into an equity curve.

    Identical arithmetic to V71's `backtest`: group by the month of `date`, take
    the MEAN R multiple in that month (the risk budget is SHARED by the
    positions open in the period, not staked per trade), and apply it once.

    Extracted so the model arm and every control arm run through the same code.
    In V71 the model went through `backtest` and the controls went through
    `backtest` as well, but via a different selection path; pulling the
    compounding out makes it impossible for the arms to drift apart.
    """
    if t is None or len(t) == 0:
        return None
    t = t.sort_values("date").copy()
    total_R = float(t["r_multiple"].sum())
    t["_per"] = pd.PeriodIndex(pd.DatetimeIndex(t["date"]), freq="M")

    eq, peak, max_dd, curve = start, start, 0.0, []
    for _, g in t.groupby("_per"):
        eq += eq * risk_frac * float(g["r_multiple"].mean())
        if eq <= 0:
            eq = 0.0
            curve.append(eq)
            break
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        curve.append(eq)

    curve = np.asarray(curve, float)
    yrs = max((pd.DatetimeIndex(t["date"]).max()
               - pd.DatetimeIndex(t["date"]).min()).days / 365.25, 1e-6)
    rets = np.diff(np.r_[start, curve]) / np.r_[start, curve[:-1]]
    sharpe = (float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(12))
              if len(rets) > 2 and np.std(rets, ddof=1) > 0 else np.nan)
    return {"n_trades": int(len(t)), "months": int(len(curve)),
            "final": float(eq), "total_R": total_R,
            "R_per_trade": total_R / len(t),
            "return_pct": (eq / start - 1) * 100,
            "cagr_pct": ((eq / start) ** (1 / yrs) - 1) * 100 if eq > 0 else -100,
            "max_dd_pct": max_dd * 100, "sharpe": sharpe,
            "win_rate": float(t["label"].mean()), "years": yrs}


def _fire_mask(te, quantile):
    p = te["p"].to_numpy(float)
    return p >= float(np.quantile(p, 1.0 - quantile))


def backtest(te, risk_frac, start=START_CAPITAL, quantile=0.10):
    """The model arm. Same selection and arithmetic as V71, via `_compound`."""
    return _compound(te[_fire_mask(te, quantile)], risk_frac, start)


# =============================================================================
# CONTROLS
# =============================================================================
def matched_random_backtest(te, risk_frac, quantile, start=START_CAPITAL,
                            seed=SEED, n_draws=N_DRAWS):
    """
    Random control MATCHED on time in market.

    For every month the model fired in, this draws the same number of trades the
    model took that month, uniformly from that month's candidates. The control
    is therefore invested in exactly the same months, with exactly the same
    number of positions, and the only difference left is which trades.

    This is the control that answers "did the RANKING do anything". The
    unmatched control below answers a different and much weaker question.
    """
    te = te.copy()
    te["_per"] = pd.PeriodIndex(pd.DatetimeIndex(te["date"]), freq="M")
    fired = te[_fire_mask(te, quantile)]
    if fired.empty:
        return None
    want = fired.groupby("_per").size()

    by_month = {per: g for per, g in te.groupby("_per")}
    rng = np.random.default_rng(seed)

    outs = []
    for _ in range(n_draws):
        picks = []
        for per, k in want.items():
            pool = by_month.get(per)
            if pool is None or len(pool) == 0:
                continue
            k = int(min(k, len(pool)))
            idx = rng.choice(len(pool), size=k, replace=False)
            picks.append(pool.iloc[idx])
        if not picks:
            continue
        r = _compound(pd.concat(picks), risk_frac, start)
        if r:
            outs.append(r)
    if not outs:
        return None
    return _summarise_draws(outs)


def unmatched_random_backtest(te, risk_frac, quantile, start=START_CAPITAL,
                              seed=SEED, n_draws=N_DRAWS):
    """
    V71's control, kept so the exposure effect stays visible and quotable.

    A uniform score over ALL candidates, top `quantile` taken. Same trade COUNT
    as the model, but spread across nearly every month in the sample, so it
    compounds far more often than a clustered signal does.
    """
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n_draws):
        t = te.copy()
        t["p"] = rng.random(len(t))
        r = _compound(t[_fire_mask(t, quantile)], risk_frac, start)
        if r:
            outs.append(r)
    if not outs:
        return None
    return _summarise_draws(outs)


def name_matched_random_backtest(te, risk_frac, quantile, start=START_CAPITAL,
                                 seed=SEED, n_draws=N_DRAWS):
    """
    Random control matched on NAME as well as month - the entry-timing control.

    This is the one that maps onto what the model actually claims. A sniper does
    not claim you should be in the market; it claims that IF you are buying this
    name around this time, now is a better moment than another moment. So the
    control holds the name and the month fixed and moves only the day: for every
    signal on ticker T in month M, it draws a different candidate entry on T,
    also in M.

    Anything the two arms share - which names, which regimes, how much time in
    market, the whole sector and market-cap composition of the book - cancels.
    What is left is timing, which is the only thing the model is selling.

    Falls back to the same quarter, then the same year, for a ticker-month with
    no alternative candidate; counts how often it had to.
    """
    # reset the index so groupby positions and .iloc positions are the same
    te = te.reset_index(drop=True).copy()
    di = pd.DatetimeIndex(te["date"])
    te["_per"] = pd.PeriodIndex(di, freq="M")
    te["_qtr"] = pd.PeriodIndex(di, freq="Q")
    te["_yr"] = pd.PeriodIndex(di, freq="Y")
    fired = te[_fire_mask(te, quantile)]
    if fired.empty:
        return None

    pools = {k: dict(te.groupby(["ticker", k]).indices)
             for k in ("_per", "_qtr", "_yr")}
    rng = np.random.default_rng(seed)

    outs, widened = [], 0
    for _ in range(n_draws):
        picks, w = [], 0
        for tk, per, qtr, yr in zip(fired["ticker"], fired["_per"],
                                    fired["_qtr"], fired["_yr"]):
            pool = pools["_per"].get((tk, per))
            if pool is None or len(pool) < 2:
                pool = pools["_qtr"].get((tk, qtr))
                w += 1
                if pool is None or len(pool) < 2:
                    pool = pools["_yr"].get((tk, yr))
            if pool is None or len(pool) == 0:
                continue
            picks.append(int(rng.choice(pool)))
        if not picks:
            continue
        r = _compound(te.iloc[picks], risk_frac, start)
        if r:
            outs.append(r)
            widened = max(widened, w)
    if not outs:
        return None
    return _summarise_draws(outs, n_draws=len(outs), widened=widened)


def _summarise_draws(outs, **extra):
    """Pack a list of control draws into medians, spreads and the raw arrays."""
    ret = np.array([o["return_pct"] for o in outs], float)
    tr = np.array([o["total_R"] for o in outs], float)
    wr = np.array([o["win_rate"] for o in outs], float)
    med = int(np.argsort(ret)[len(ret) // 2])
    out = {"n_draws": len(outs), "median": outs[med],
           "best": outs[int(np.argmax(ret))],
           "returns": ret, "total_Rs": tr, "win_rates": wr,
           "median_return_pct": float(np.median(ret)),
           "best_return_pct": float(ret.max()),
           "worst_return_pct": float(ret.min()),
           "median_total_R": float(np.median(tr)),
           "median_win_rate": float(np.median(wr)),
           "mean_total_R": float(np.mean(tr)),
           "mean_months": float(np.mean([o["months"] for o in outs]))}
    out.update(extra)
    return out


def percentile_rank(value, draws):
    """
    Where the model sits INSIDE the control distribution.

    V71 asked whether the model beat the BEST of 20 random draws. The maximum of
    20 draws sits near the 97.5th percentile of the control distribution, so
    that question is a one-sided test at about p < 0.025 - stricter than the 5%
    bar the rest of this project uses - and it is carried by a single draw, so
    it moves when the seed moves. It is the wrong statistic twice over.

    The right one is the model's rank among the draws: 'beat 18 of 20' is an
    empirical p of about 0.10, quotable and stable. Report this, not the max.
    """
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(value) or draws.size == 0:
        return None
    beat = int((draws < value).sum())
    n = int(draws.size)
    return {"beat": beat, "n": n, "pct": 100.0 * beat / n,
            "p_one_sided": (n - beat + 1) / (n + 1)}


def concentration(te, quantile, ks=(0.01, 0.05, 0.10, 0.25)):
    """
    How much of the signal's total R comes from its few biggest winners.

    R per trade hides this, and it is the difference between a tradeable edge
    and a lottery ticket. If most of the R sits in a handful of trades, the
    average is real but unrepeatable: miss those few and the edge is gone. This
    is the one thing the per-trade table genuinely cannot show, and the reason
    some path-aware analysis stays in the thesis.

    A share is meaningless on its own: ANY positive-expectancy trade set is
    top-heavy, because the losers are capped at -1R and the winners are not. So
    the fired set is reported beside the same measure on all candidates. Only a
    GAP between the two columns says the signal is more lottery-like than the
    thing it selected from.
    """
    t = te[_fire_mask(te, quantile)]
    if t.empty:
        return None

    def shares(x):
        r = np.sort(np.asarray(x, float))[::-1]
        tot = float(r.sum())
        return r, tot, [float(r[:max(1, int(round(len(r) * k)))].sum()) / tot
                        if tot else np.nan for k in ks]

    _, tot, fired = shares(t["r_multiple"])
    _, _, blind = shares(te["r_multiple"])
    rows = [{"Top trades": f"top {k:.0%}",
             "n (fired)": max(1, int(round(len(t) * k))),
             "Share of fired R": f, "Share of blind R": b,
             "Gap": f - b}
            for k, f, b in zip(ks, fired, blind)]
    return pd.DataFrame(rows), tot


# =============================================================================
# HOW MUCH SAMPLE WOULD THIS NEED
# =============================================================================
def power_note(measured, base_rate):
    """
    A failing lower bound has two possible causes and they are not the same
    finding: the effect is absent, or the sample cannot resolve it. This
    inverts the normal-approximation interval to say how many EFFECTIVE bets
    the observed lift would need before its lower bound could clear the blind
    rate - and how many before it clears with a usable margin.

    n_eff is the clustered sample size the block bootstrap backs out, not the
    trade count. Trades on overlapping windows in the same names are one bet.
    """
    p = float(measured.get("win_rate", np.nan))
    n_eff = float(measured.get("n_eff", np.nan))
    base_rate = float(base_rate)
    if (not np.isfinite(p) or not np.isfinite(n_eff) or n_eff <= 0
            or not np.isfinite(base_rate)):
        return None
    lift = p - base_rate
    if lift <= 0:
        return {"lift": lift, "n_eff": n_eff, "need_clear": np.inf,
                "need_margin": np.inf}

    def need(target_bound):
        gap = p - target_bound
        if gap <= 0:
            return np.inf
        return float(np.ceil(p * (1 - p) * (1.96 / gap) ** 2))

    return {"lift": lift, "n_eff": n_eff,
            "need_clear": need(base_rate),
            "need_margin": need(base_rate + 0.03)}


# =============================================================================
# RUNNER
# =============================================================================
def run_tables_v72(model_file=MODEL_FILE, price_cache=None, out_dir=OUT_DIR,
                   thresholds=THRESHOLDS, start=START_CAPITAL,
                   risk_tiers=None, seed=SEED, n_draws=N_DRAWS, verbose=True):
    risk_tiers = risk_tiers or RISK_TIERS
    os.makedirs(out_dir, exist_ok=True)
    model = M70.load_model(model_file)
    p = model["provenance"]
    price_cache = price_cache or p["price_cache"]
    q = model["quantile"]

    print("=" * 96)
    print("V72 - RESULTS TABLES (exposure-matched controls)")
    print("=" * 96)
    M70.describe(model)

    meas = model.get("measured", {}) or {}
    pn = power_note(meas, float(meas.get("blind_win_rate", np.nan)))
    if pn and np.isfinite(pn["need_clear"]):
        print(f"\n  SAMPLE NEEDED FOR THIS LIFT")
        print(f"    observed lift over blind          {pn['lift']*100:+.1f} pp")
        print(f"    effective sample now (n_eff)      {pn['n_eff']:.0f}")
        print(f"    n_eff to clear the blind rate     {pn['need_clear']:.0f}"
              f"   ({pn['need_clear']/pn['n_eff']:.1f}x current)")
        print(f"    n_eff to clear it by 3pp          {pn['need_margin']:.0f}"
              f"   ({pn['need_margin']/pn['n_eff']:.1f}x current)")
        print(f"    A lift this size is not absent - it is unresolved. More "
              f"NAMES raise n_eff;")
        print(f"    more trades on the same names and dates do not.")

    print(f"\n  rebuilding the out-of-sample record (purged walk-forward)")
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, p["horizon"], E64.STEP,
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    te = E64.walk_forward(d, model["features"], E64.MIN_TRAIN_YEARS, 1, seed,
                          p["horizon"], verbose=False)
    print(f"  {len(te):,} out-of-sample trades, {te['year'].nunique()} years")

    # ---- Tables 1 and 2, unchanged from V71 --------------------------------
    t1, base = classification_table(te, thresholds, seed)
    print("\n" + "=" * 96)
    print("  TABLE 1  Classification performance by confidence threshold")
    print("=" * 96)
    print(f"  baseline (buy every candidate): {base:.1%} win rate")
    print(t1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t1.to_csv(f"{out_dir}/table1_classification.csv", index=False)

    t2 = confusion_table(te, q)
    print("\n" + "=" * 96)
    print(f"  TABLE 2  Confusion matrix at the deployed threshold (top {q:.0%})")
    print("=" * 96)
    print(t2.to_string(index=False))
    t2.to_csv(f"{out_dir}/table2_confusion.csv", index=False)

    # ---- Table 3 -----------------------------------------------------------
    bh = buy_and_hold(price_cache, te["date"])
    all_months = int(pd.PeriodIndex(pd.DatetimeIndex(te["date"]),
                                    freq="M").nunique())
    rows = []
    for tier, frac in risk_tiers.items():
        b = backtest(te, frac, start, q)
        if not b:
            continue
        rows.append({
            "Strategy": tier, "Portfolio risk / period": f"{frac:.0%}",
            "Trades": b["n_trades"], "Months in market": b["months"],
            "Total R": b["total_R"], "R per trade": b["R_per_trade"],
            "Final balance (USD)": b["final"], "Return %": b["return_pct"],
            "CAGR %": b["cagr_pct"], "Max drawdown %": b["max_dd_pct"],
            "Sharpe": b["sharpe"], "Win rate": b["win_rate"],
            "Buy & hold return %": bh["return_pct"] if bh else np.nan,
            "Beats buy & hold": (b["return_pct"] > bh["return_pct"]
                                 if bh else np.nan)})
    t3 = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"  TABLE 3  Backtesting by risk tier (starting capital "
          f"{start:,.0f} USD)")
    print("=" * 96)
    print(t3.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  Returns compound MONTHLY and the risk budget is SHARED by the "
          f"positions open in a")
    print(f"  month, not staked per trade. 'Total R' cannot be inflated by any "
          f"sizing choice.")
    if len(t3):
        mm = int(t3["Months in market"].iloc[0])
        print(f"\n  EXPOSURE: the signal is invested in {mm} of {all_months} "
              f"months ({mm/all_months:.0%} of the window).")
        print(f"  Returns compound only in active months, so any arm invested "
              f"in more of them")
        print(f"  compounds more often. Compare percentages ONLY against arms "
              f"with the same")
        print(f"  months-in-market; compare everything else on Total R.")
    if bh:
        print(f"\n  buy & hold: equal weight across {bh['n_assets']} assets over "
              f"{bh['years']:.1f} years -> {bh['return_pct']:,.1f}% "
              f"({bh['cagr_pct']:.1f}% CAGR), invested in all {all_months} "
              f"months.")
    t3.to_csv(f"{out_dir}/table3_backtest.csv", index=False)

    # ---- Table 4 -----------------------------------------------------------
    frac = risk_tiers.get("Moderate", 0.05)
    mb = backtest(te, frac, start, q)
    allb = _compound(te, frac, start)
    nrnd = name_matched_random_backtest(te, frac, q, start, seed, n_draws)
    mrnd = matched_random_backtest(te, frac, q, start, seed, n_draws)
    urnd = unmatched_random_backtest(te, frac, q, start, seed, n_draws)

    def _row(arm, o, months=None, trades=None):
        return {"Arm": arm,
                "Total R": o.get("total_R", o.get("median_total_R", np.nan)),
                "R per trade": o.get("R_per_trade", np.nan),
                "Win rate": o.get("win_rate", o.get("median_win_rate", np.nan)),
                "Trades": trades if trades is not None else o.get("n_trades",
                                                                  np.nan),
                "Months in market": (months if months is not None
                                     else o.get("months",
                                                o.get("mean_months", np.nan))),
                "Return % (illustrative)": o.get("return_pct",
                                                 o.get("median_return_pct",
                                                       np.nan)),
                "Max DD %": o.get("max_dd_pct", np.nan)}

    rows = [_row("Model signal", mb)]
    if nrnd:
        rows.append(_row(f"Random entry, SAME NAME + month (median of "
                         f"{nrnd['n_draws']})", nrnd["median"]))
    if mrnd:
        rows.append(_row(f"Random entry, same month (median of "
                         f"{mrnd['n_draws']})", mrnd["median"]))
    if urnd:
        rows.append(_row(f"Random, UNMATCHED - V71 method (median of "
                         f"{urnd['n_draws']})", urnd["median"],
                         trades=mb["n_trades"]))
    if allb:
        rows.append(_row("Buy every candidate", allb))
    if bh:
        rows.append(_row("Buy & hold (equal weight)",
                         {"return_pct": bh["return_pct"]},
                         months=all_months, trades=bh["n_assets"]))
    t4 = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"  TABLE 4  Baseline comparison (Moderate tier, {frac:.0%} "
          f"portfolio risk)")
    print("=" * 96)
    print(t4.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  Columns are ordered by what the model CLAIMS. A sniper claims "
          f"that the trades it")
    print(f"  picks are better, not that it is invested more or sized better, "
          f"so Total R, R per")
    print(f"  trade and win rate at a FIXED trade count are the result. Return "
          f"% is downstream")
    print(f"  of exposure and position sizing, which the model does not claim "
          f"to control - it is")
    print(f"  an illustration of one sizing rule, not evidence, and it is the "
          f"last column for")
    print(f"  that reason. Buy & hold is in the table for context only: it "
          f"answers 'should I be")
    print(f"  in the market', and the model answers 'is now a better entry "
          f"than another day'.")
    t4.to_csv(f"{out_dir}/table4_baselines.csv", index=False)

    # ---- the verdict: rank inside the right control's distribution ---------
    print("\n" + "=" * 96)
    print("  THE TEST THAT MATCHES THE CLAIM")
    print("=" * 96)
    print(f"  Same names, same months, same number of positions - only the "
          f"entry DAY differs.")
    for lab, ctl in (("same name + month", nrnd), ("same month", mrnd)):
        if not ctl:
            continue
        print(f"\n  vs random entry, {lab}  ({ctl['n_draws']} draws)")
        for stat, mine, draws in (
                ("total R", mb["total_R"], ctl["total_Rs"]),
                ("win rate", mb["win_rate"], ctl["win_rates"]),
                ("return %", mb["return_pct"], ctl["returns"])):
            pr = percentile_rank(mine, draws)
            if not pr:
                continue
            fmt = "{:>9.3f}" if stat == "win rate" else "{:>9,.1f}"
            print(f"    {stat:<9} model " + fmt.format(mine)
                  + "   control median " + fmt.format(float(np.median(draws)))
                  + f"   beat {pr['beat']}/{pr['n']}"
                  f"   p~{pr['p_one_sided']:.3f}")
        if ctl.get("widened"):
            print(f"    ({ctl['widened']} signals had no alternative entry in "
                  f"the same ticker-month; widened)")
    print(f"\n  Rank in the control distribution is the statistic - NOT whether "
          f"the model beat the")
    print(f"  single best draw. The max of {n_draws} draws sits near the "
          f"97.5th percentile, so that")
    print(f"  bar is a ~p<0.025 test carried by one draw, and it moves with the "
          f"seed. V71 used it")
    print(f"  and drew a conclusion it could not support.")

    # ---- concentration -----------------------------------------------------
    conc = concentration(te, q)
    if conc:
        cdf, ctot = conc
        print("\n" + "=" * 96)
        print("  TABLE 5  Where the signal's R actually comes from")
        print("=" * 96)
        print(cdf.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print(f"\n  total R across all fired trades: {ctot:,.1f}")
        print(f"  If most of the R sits in a few trades, the average is real "
              f"but unrepeatable.")
        cdf.to_csv(f"{out_dir}/table5_concentration.csv", index=False)
        _md(cdf, f"{out_dir}/table5_concentration.md")

    for name, df in (("table1_classification", t1), ("table2_confusion", t2),
                     ("table3_backtest", t3), ("table4_baselines", t4)):
        _md(df, f"{out_dir}/{name}.md")
    print(f"\n  wrote CSV and markdown for all four tables to {out_dir}/")
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "baseline_wr": base,
            "matched_random": mrnd, "unmatched_random": urnd, "power": pn}


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default=MODEL_FILE)
    ap.add_argument("--price-cache", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--start", type=float, default=START_CAPITAL)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_tables_v72(c.model_file, c.price_cache, c.out_dir, THRESHOLDS,
                   c.start, None, c.seed, c.n_draws)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v70.joblib"
    RUN_PRICE_CACHE = None          # None uses the cache the model was trained on
    RUN_OUT_DIR     = "thesis_tables_v72"
    RUN_THRESHOLDS  = THRESHOLDS    # confidence levels for Table 1
    RUN_START       = 10_000.0
    RUN_RISK_TIERS  = {"Aggressive": 0.10, "Moderate": 0.05,
                       "Conservative": 0.02}
    RUN_N_DRAWS     = 20            # random draws per control
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_tables_v72(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
                       out_dir=RUN_OUT_DIR, thresholds=RUN_THRESHOLDS,
                       start=RUN_START, risk_tiers=RUN_RISK_TIERS,
                       n_draws=RUN_N_DRAWS, seed=RUN_SEED)
