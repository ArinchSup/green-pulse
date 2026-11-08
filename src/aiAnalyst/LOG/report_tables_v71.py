#!/usr/bin/env python3
"""
report_tables_v71.py - the results tables, in the form a thesis chapter expects.

WHAT THIS PRODUCES

    Table 1  Classification performance at each confidence threshold -
             accuracy, precision, recall, F1, AUC, expectancy, with the
             clustered interval on precision.
    Table 2  Confusion matrix at the deployed threshold.
    Table 3  Backtesting by risk tier, starting at 10,000 USD - the shape the
             reference project's Table 4.3 uses, with one column added.
    Table 4  BASELINE COMPARISON. The model against buying everything, against
             a random signal at matched frequency, and against buy-and-hold.

THE COLUMN THAT MAKES TABLE 3 HONEST

The reference project reports "Best: HOT-USD 222,993%". HOT rose roughly a
thousandfold over that window, so any strategy that held it returns a number
like that. Without a buy-and-hold comparison on the same asset over the same
period, a backtest return measures the ASSET, not the strategy - and reporting
the best of many assets reports a lottery winner, which is why their own table
also shows a worst of -99.91%.

So every backtest row here carries buy-and-hold on the identical universe and
window beside it, and Table 4 puts the strategy against baselines that know
nothing. A return that does not beat buy-and-hold is not a result, however large.

WHY PRECISION CARRIES AN INTERVAL AND ACCURACY DOES NOT

Accuracy over 250,000 overlapping trades looks precise to four decimals and is
not. The block bootstrap on precision is the only number in Table 1 that
reflects how much independent evidence there actually is. Quote that one.

USAGE
  python report_tables_v71.py            # needs entry_model_v70.joblib
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v70 as M70
import evaluate_sniper_v66 as S66
from trade_config import HORIZON_CONFIGS

# =============================================================================
# CONFIG
# =============================================================================
MODEL_FILE = "entry_model_v70.joblib"
PRICE_CACHE = E64.PRICE_CACHE
OUT_DIR = "thesis_tables"

START_CAPITAL = 10_000.0
RISK_TIERS = {"Aggressive": 0.10, "Moderate": 0.05, "Conservative": 0.02}
THRESHOLDS = (0.20, 0.10, 0.05, 0.02, 0.01)
SEED = 71


# =============================================================================
# CLASSIFICATION TABLES
# =============================================================================
def classification_table(te, thresholds=THRESHOLDS, seed=SEED):
    """One row per confidence threshold. Precision carries the honest interval."""
    y = te["label"].to_numpy(int)
    p = te["p"].to_numpy(float)
    base = float(y.mean())
    rows = []
    for q in thresholds:
        cut = float(np.quantile(p, 1.0 - q))
        fire = p >= cut
        tp = int((fire & (y == 1)).sum())
        fp = int((fire & (y == 0)).sum())
        fn = int((~fire & (y == 1)).sum())
        tn = int((~fire & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        sub = te[fire]
        lo, hi = S66.block_bootstrap_mean(sub["label"].to_numpy(float),
                                          sub["date"].to_numpy(), seed=seed)
        elo, ehi = S66.block_bootstrap_mean(sub["r_multiple"].to_numpy(float),
                                            sub["date"].to_numpy(), seed=seed)
        rows.append({
            "Threshold": f"top {q:.0%}",
            "Signals": tp + fp,
            "Coverage": (tp + fp) / len(y),
            "Accuracy": (tp + tn) / len(y),
            "Precision": prec,
            "Precision 95% CI": (f"[{lo:.1%}, {hi:.1%}]"
                                 if np.isfinite(lo) else "n/a"),
            "Recall": rec,
            "F1": f1,
            "Expectancy (R)": float(sub["r_multiple"].mean()),
            # Tables 3 and 4 are built on EXPECTANCY, not on precision, so it
            # needs its own interval. The first version gave a CI on precision
            # only, which left the backtest's headline number unqualified.
            "Expectancy 95% CI": (
                f"[{elo:+.3f}, {ehi:+.3f}]" if np.isfinite(elo) else "n/a"),
            "Lift over blind": prec - base,
        })
    return pd.DataFrame(rows), base


def confusion_table(te, quantile):
    y = te["label"].to_numpy(int)
    p = te["p"].to_numpy(float)
    fire = p >= float(np.quantile(p, 1.0 - quantile))
    tp = int((fire & (y == 1)).sum()); fp = int((fire & (y == 0)).sum())
    fn = int((~fire & (y == 1)).sum()); tn = int((~fire & (y == 0)).sum())
    return pd.DataFrame(
        [{"": "Signal (model fires)", "Actual win": tp, "Actual loss": fp},
         {"": "No signal", "Actual win": fn, "Actual loss": tn}])


# =============================================================================
# BACKTEST
# =============================================================================
def backtest(te, risk_frac, start=START_CAPITAL, quantile=0.10,
             compound="monthly"):
    """
    Compound the fired trades into an equity curve.

    HOW COMPOUNDING IS DONE, AND WHY IT MATTERS MORE THAN THE SIGNAL

    The first version of this function compounded trade by trade in date order,
    staking a fixed fraction of current equity on each. On the full candidate set
    that returned 3.5e49 percent. The number is arithmetic, not skill: 30,000
    trades with a small positive expectancy, compounded sequentially, produce an
    astronomical figure - and it is unachievable, because those trades OVERLAP.
    You cannot stake 5% of equity on a trade, and then 5% of the resulting
    equity on another trade that opened before the first one closed.

    So the default compounds by MONTH. Trades are grouped by the month they
    closed, their R multiples are summed, and that month's total is applied to
    equity once. Concurrent trades therefore share the period rather than
    multiplying each other, which is how a book actually works.

    `total_R` is reported alongside and is the number to trust: it is additive,
    it cannot be inflated by a sizing assumption, and it is the same quantity the
    expectancy column measures.
    """
    p = te["p"].to_numpy(float)
    fire = p >= float(np.quantile(p, 1.0 - quantile))
    t = te[fire].sort_values("date").copy()
    if t.empty:
        return None
    total_R = float(t["r_multiple"].sum())

    if compound == "trade":
        groups = [(i, g) for i, g in t.groupby(np.arange(len(t)))]
    else:
        t["_per"] = pd.PeriodIndex(pd.DatetimeIndex(t["date"]), freq="M")
        groups = list(t.groupby("_per"))

    eq, peak, max_dd, curve = start, start, 0.0, []
    for _, g in groups:
        # risk_frac is the TOTAL portfolio risk in a period, SHARED by the
        # positions open in it - the same budget idea as the portfolio layer's
        # 6% risk-at-stops cap. Summing instead would stake risk_frac PER trade,
        # so a month with fifty signals would put 250% of equity at risk and the
        # account would be wiped by arithmetic rather than by the model.
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
    return {"n_trades": int(len(t)), "n_periods": int(len(curve)),
            "final": float(eq), "total_R": total_R,
            "R_per_trade": total_R / len(t),
            "return_pct": (eq / start - 1) * 100,
            "cagr_pct": ((eq / start) ** (1 / yrs) - 1) * 100 if eq > 0 else -100,
            "max_dd_pct": max_dd * 100, "sharpe": sharpe,
            "win_rate": float(t["label"].mean()), "years": yrs}


def buy_and_hold(price_cache, dates, start=START_CAPITAL, verbose=False):
    """
    Equal-weight buy-and-hold on the same universe over the same window. This is
    the benchmark the reference project's backtest is missing, and it is usually
    the one that decides whether a strategy did anything.
    """
    lo, hi = pd.Timestamp(min(dates)), pd.Timestamp(max(dates))
    rets = []
    for f in sorted(os.listdir(price_cache)):
        if not f.endswith(".pkl"):
            continue
        d = E64.P2.load_prices(os.path.splitext(f)[0], price_cache)
        if d is None:
            continue
        d = d[(d.index >= lo) & (d.index <= hi)]
        if len(d) < 30:
            continue
        c = d["Close"].to_numpy(float)
        if c[0] > 0 and np.isfinite(c[0]) and np.isfinite(c[-1]):
            rets.append(c[-1] / c[0] - 1)
    if not rets:
        return None
    r = float(np.mean(rets))
    yrs = max((hi - lo).days / 365.25, 1e-6)
    return {"n_assets": len(rets), "final": start * (1 + r),
            "return_pct": r * 100,
            "cagr_pct": ((1 + r) ** (1 / yrs) - 1) * 100 if r > -1 else -100,
            "years": yrs}


def random_signal_backtest(te, risk_frac, quantile, seed=SEED, n_draws=20):
    """The same number of trades, chosen at random. What luck alone returns."""
    rng = np.random.default_rng(seed)
    outs = []
    for i in range(n_draws):
        t = te.copy()
        t["p"] = rng.random(len(t))
        r = backtest(t, risk_frac, quantile=quantile)
        if r:
            outs.append(r["return_pct"])
    if not outs:
        return None
    return {"median_return_pct": float(np.median(outs)),
            "best_return_pct": float(np.max(outs)),
            "worst_return_pct": float(np.min(outs)), "n_draws": n_draws}


# =============================================================================
# REPORT
# =============================================================================
def _md(df, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    return path


def run_tables_v71(model_file=MODEL_FILE, price_cache=None, out_dir=OUT_DIR,
                   thresholds=THRESHOLDS, start=START_CAPITAL,
                   risk_tiers=None, seed=SEED, verbose=True):
    risk_tiers = risk_tiers or RISK_TIERS
    os.makedirs(out_dir, exist_ok=True)
    model = M70.load_model(model_file)
    p = model["provenance"]
    price_cache = price_cache or p["price_cache"]
    q = model["quantile"]

    print("=" * 96)
    print("V71 - RESULTS TABLES")
    print("=" * 96)
    M70.describe(model)

    print(f"\n  rebuilding the out-of-sample record (purged walk-forward)")
    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, p["horizon"], E64.STEP,
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    te = E64.walk_forward(d, model["features"], E64.MIN_TRAIN_YEARS, 1, seed,
                          p["horizon"], verbose=False)
    print(f"  {len(te):,} out-of-sample trades, "
          f"{te['year'].nunique()} years")

    # ---- Table 1 -----------------------------------------------------------
    t1, base = classification_table(te, thresholds, seed)
    print("\n" + "=" * 96)
    print("  TABLE 1  Classification performance by confidence threshold")
    print("=" * 96)
    print(f"  baseline (buy every candidate): {base:.1%} win rate")
    print(t1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t1.to_csv(f"{out_dir}/table1_classification.csv", index=False)

    # ---- Table 2 -----------------------------------------------------------
    t2 = confusion_table(te, q)
    print("\n" + "=" * 96)
    print(f"  TABLE 2  Confusion matrix at the deployed threshold (top {q:.0%})")
    print("=" * 96)
    print(t2.to_string(index=False))
    t2.to_csv(f"{out_dir}/table2_confusion.csv", index=False)

    # ---- Table 3 -----------------------------------------------------------
    bh = buy_and_hold(price_cache, te["date"])
    rows = []
    for tier, frac in risk_tiers.items():
        b = backtest(te, frac, start, q)
        if not b:
            continue
        rows.append({
            "Strategy": tier, "Portfolio risk / period": f"{frac:.0%}",
            "Trades": b["n_trades"], "Total R": b["total_R"],
            "R per trade": b["R_per_trade"],
            "Final balance (USD)": b["final"],
            "Return %": b["return_pct"], "CAGR %": b["cagr_pct"],
            "Max drawdown %": b["max_dd_pct"], "Sharpe": b["sharpe"],
            "Win rate": b["win_rate"],
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
    print(f"  month, not staked per trade. Both choices are deliberate: "
          f"per-trade compounding on")
    print(f"  overlapping positions is unachievable and returned 3.5e49% when "
          f"this ran without them.")
    print(f"  'Total R' cannot be inflated by any sizing choice - quote it "
          f"beside every percentage.")
    if bh:
        print(f"\n  buy & hold: equal weight across {bh['n_assets']} assets over "
              f"{bh['years']:.1f} years -> {bh['return_pct']:,.1f}% "
              f"({bh['cagr_pct']:.1f}% CAGR)")
        print(f"  Any strategy return below that line did WORSE than doing "
              f"nothing over the same window.")
    t3.to_csv(f"{out_dir}/table3_backtest.csv", index=False)

    # ---- Table 4 -----------------------------------------------------------
    frac = risk_tiers.get("Moderate", 0.05)
    mb = backtest(te, frac, start, q)
    allb = backtest(te.assign(p=1.0), frac, start, 1.0)
    rnd = random_signal_backtest(te, frac, q, seed)
    rows = [{"Arm": "Model signal", "R per trade": mb["R_per_trade"],
             "Return %": mb["return_pct"], "Trades": mb["n_trades"],
             "Win rate": mb["win_rate"], "Max DD %": mb["max_dd_pct"]}]
    if allb:
        rows.append({"Arm": "Buy every candidate",
                     "R per trade": allb["R_per_trade"],
                     "Return %": allb["return_pct"],
                     "Trades": allb["n_trades"], "Win rate": allb["win_rate"],
                     "Max DD %": allb["max_dd_pct"]})
    if rnd:
        rows.append({"Arm": f"Random signal (median of {rnd['n_draws']})",
                     "R per trade": np.nan,
                     "Return %": rnd["median_return_pct"],
                     "Trades": mb["n_trades"], "Win rate": np.nan,
                     "Max DD %": np.nan})
        rows.append({"Arm": "Random signal (best draw)",
                     "R per trade": np.nan,
                     "Return %": rnd["best_return_pct"],
                     "Trades": mb["n_trades"], "Win rate": np.nan,
                     "Max DD %": np.nan})
    if bh:
        rows.append({"Arm": "Buy & hold (equal weight)", "R per trade": np.nan,
                     "Return %": bh["return_pct"], "Trades": bh["n_assets"],
                     "Win rate": np.nan, "Max DD %": np.nan})
    t4 = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"  TABLE 4  Baseline comparison (Moderate tier, {frac:.0%} portfolio risk)")
    print("=" * 96)
    print(t4.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    t4.to_csv(f"{out_dir}/table4_baselines.csv", index=False)

    if rnd and mb["return_pct"] <= rnd["best_return_pct"]:
        print(f"\n  NOTE the model's return ({mb['return_pct']:,.0f}%) does not "
              f"exceed the best of "
              f"{rnd['n_draws']} RANDOM signals ({rnd['best_return_pct']:,.0f}%).")
        print(f"  Report it as such. A backtest number that luck reaches is not "
              f"evidence of a model.")

    for name, df in (("table1_classification", t1), ("table2_confusion", t2),
                     ("table3_backtest", t3), ("table4_baselines", t4)):
        _md(df, f"{out_dir}/{name}.md")
    print(f"\n  wrote CSV and markdown for all four tables to {out_dir}/")
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "baseline_wr": base}


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default=MODEL_FILE)
    ap.add_argument("--price-cache", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--start", type=float, default=START_CAPITAL)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_tables_v71(c.model_file, c.price_cache, c.out_dir, THRESHOLDS,
                   c.start, None, c.seed)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v70.joblib"
    RUN_PRICE_CACHE = None          # None uses the cache the model was trained on
    RUN_OUT_DIR     = "thesis_tables"
    RUN_THRESHOLDS  = THRESHOLDS    # confidence levels for Table 1
    RUN_START       = 10_000.0      # starting capital, matching the reference
    RUN_RISK_TIERS  = {"Aggressive": 0.10, "Moderate": 0.05,
                       "Conservative": 0.02}
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_tables_v71(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
                       out_dir=RUN_OUT_DIR, thresholds=RUN_THRESHOLDS,
                       start=RUN_START, risk_tiers=RUN_RISK_TIERS,
                       seed=RUN_SEED)
