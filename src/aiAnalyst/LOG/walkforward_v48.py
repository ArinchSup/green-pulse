#!/usr/bin/env python3
"""
walkforward_v48.py - why does every version pass cross-validation and fail out of
sample?

THE PATTERN THIS EXPLAINS

    version                       CV        OOS (2023-2026)
    V43  within-date AUC        0.522            0.507
    V44  IC, date-neutral      +0.029           -0.011
    V47  IC, sector-neutral    +0.030           -0.004

Three different target specifications, three clean in-sample results, three failures on
the same test period. That consistency means the problem is not specific to any one
setup. Two explanations remain, and they point in opposite directions:

  REGIME CHANGE          relationships that held 2005-2023 stopped holding afterwards.
                         If so, a model retrained on a rolling recent window could
                         work and static training is simply the wrong approach.

  CV OPTIMISM            the purged cross-validation is systematically generous, and
                         NO out-of-period test would ever pass. If so, everything is
                         closed and the seven versions share one root cause.

WHAT THIS DOES

Walks forward across the whole 2005-2026 span. For each test window it trains only on
data that ended at least one embargo before the window opens, then measures IC inside
it. Every test window is genuinely out of period, so 2023-2026 stops being a special
case and becomes one row among many.

  EXPANDING   train on everything available up to the cutoff (what V43/V44/V47 did)
  ROLLING     train only on the most recent N years, discarding older data

If relationships decay, rolling beats expanding. If nothing works anywhere, neither
helps and the answer is CV optimism.

It also builds a DECAY CURVE: pooling every walk-forward prediction, it reports IC
against how many years had passed since that model's training data ended. A signal
that decays shows IC falling with distance; a signal that never existed shows noise at
every distance.

REQUIRES  the V44 training dataset, and ideally the V44 OOS dataset too so the span
reaches 2026. Runs offline.

USAGE
  python walkforward_v48.py --datasets dataset_..._cut20230630.csv.gz dataset_...oos....csv.gz
  python walkforward_v48.py --horizon 20 --target sector --mode both
  python walkforward_v48.py --quick          # fewer trees, for a first look
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import pillar2_v43_features as F
from class_xgboost_v44 import PARAMS, ic_per_date, ic_stats, load
from sector_neutral_v47 import add_targets, sector_map


def fmt(x, s="+.4f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{s}}"


# =============================================================================
# DATA
# =============================================================================
def load_all(paths):
    frames = []
    for p in paths:
        d, _ = load(p)
        frames.append(d)
        print(f"    {p}: {len(d):,} rows  {d['signal_date'].min()} -> {d['signal_date'].max()}")
    d = pd.concat(frames, ignore_index=True)
    before = len(d)
    d = d.drop_duplicates(["ticker", "signal_date"], keep="first")
    d = d.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    d["_date"] = pd.to_datetime(d["signal_date"])
    if before != len(d):
        print(f"    dropped {before - len(d):,} rows duplicated across files")
    return d


# =============================================================================
# WALK FORWARD
# =============================================================================
def walk(d, feats, trn, tgt, cfg):
    """
    Yields one dict per test window. Every training set ends at least `embargo_days`
    before its window opens, so a training label cannot resolve inside the window it
    is being graded on.
    """
    start = pd.Timestamp(d["_date"].min())
    end = pd.Timestamp(d["_date"].max())
    first_test = start + pd.DateOffset(years=cfg.min_train_years)
    windows = []
    t0 = first_test
    while t0 < end:
        t1 = min(t0 + pd.DateOffset(years=cfg.window_years), end)
        if (t1 - t0).days > 300:
            windows.append((t0, t1))
        t0 = t1
    if not windows:
        sys.exit("Span too short for the requested windows.")

    y = d[trn].to_numpy(dtype=float)
    a = d[tgt].to_numpy(dtype=float)
    X = d[feats]
    out = []
    for t0, t1 in windows:
        cutoff = t0 - pd.Timedelta(days=cfg.embargo_days)
        # copy: pandas can return a read-only view, and the rolling mask writes to it
        trm = np.asarray((d["_date"] < cutoff).to_numpy(), dtype=bool).copy()
        if cfg.mode_rolling:
            floor = cutoff - pd.DateOffset(years=int(cfg.train_years))
            trm = trm & (d["_date"] >= floor).to_numpy()
        vam = ((d["_date"] >= t0) & (d["_date"] < t1)).to_numpy()
        ok = trm & np.isfinite(y)
        if ok.sum() < 5000 or vam.sum() < 500:
            out.append({"from": t0, "to": t1, "n_train": int(ok.sum()),
                        "n_test": int(vam.sum()), "ic": np.nan, "t": np.nan,
                        "lo": np.nan, "hi": np.nan, "n": 0})
            continue
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[ok], y[ok], verbose=False)
        p = m.predict(X[vam])
        mask = np.isfinite(p) & np.isfinite(a[vam])
        dates = d.loc[vam, "_date"].to_numpy()[mask]
        _, ics = ic_per_date(dates, p[mask], a[vam][mask])
        st = ic_stats(ics)
        st.update({"from": t0, "to": t1, "n_train": int(ok.sum()),
                   "n_test": int(vam.sum()), "train_end": cutoff})
        # keep per-prediction records for the decay curve
        st["_rows"] = pd.DataFrame({
            "date": dates, "pred": p[mask], "actual": a[vam][mask],
            "years_ahead": (pd.to_datetime(dates) - cutoff).days / 365.25})
        out.append(st)
    return out


def report_windows(res, label):
    print(f"\n   {label}")
    print(f"     {'test window':<24}{'train':>9}{'test':>9}{'IC':>10}"
          f"{'95% range':>22}{'t':>8}")
    pos = tot = 0
    for r in res:
        rng = (f"{r['lo']:+.4f} to {r['hi']:+.4f}"
               if np.isfinite(r.get("lo", np.nan)) else "")
        print(f"     {r['from']:%Y-%m}..{r['to']:%Y-%m}{'':<8}"
              f"{r['n_train']:>9,}{r['n_test']:>9,}{fmt(r.get('ic')):>10}"
              f"{rng:>22}{fmt(r.get('t'), '+.2f'):>8}")
        if np.isfinite(r.get("ic", np.nan)):
            tot += 1
            pos += int(r["ic"] > 0)
    if tot:
        ics = [r["ic"] for r in res if np.isfinite(r.get("ic", np.nan))]
        print(f"     mean IC across windows: {np.mean(ics):+.4f}   "
              f"positive in {pos}/{tot} windows")
    return pos, tot


def decay_curve(res, label):
    rows = [r["_rows"] for r in res if "_rows" in r and len(r["_rows"])]
    if not rows:
        return
    allr = pd.concat(rows, ignore_index=True)
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
    print(f"\n   DECAY CURVE - {label}")
    print("     IC against how long after the training data the prediction was made.")
    print(f"     {'years ahead':<14}{'rows':>9}{'dates':>8}{'IC':>10}{'t':>8}")
    for lo, hi in bins:
        g = allr[(allr["years_ahead"] >= lo) & (allr["years_ahead"] < hi)]
        if len(g) < 500:
            continue
        _, ics = ic_per_date(g["date"].to_numpy(), g["pred"].to_numpy(),
                             g["actual"].to_numpy())
        st = ic_stats(ics)
        tag = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"     {tag:<14}{len(g):>9,}{st['n']:>8}{fmt(st['ic']):>10}"
              f"{fmt(st['t'], '+.2f'):>8}")
    print("     A decaying signal falls with distance. Noise is flat and near zero")
    print("     at every distance.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--target", default="sector", choices=["date", "sector", "resid"])
    ap.add_argument("--rank-target", action="store_true", default=True)
    ap.add_argument("--window-years", type=float, default=3.0)
    ap.add_argument("--min-train-years", type=float, default=6.0)
    ap.add_argument("--train-years", type=float, default=5.0,
                    help="lookback for the rolling mode")
    ap.add_argument("--embargo-days", type=int, default=200)
    ap.add_argument("--mode", default="both", choices=["expanding", "rolling", "both"])
    ap.add_argument("--edgar-cache", default="edgar_cache")
    ap.add_argument("--quick", action="store_true", help="fewer trees, faster")
    cfg = ap.parse_args()

    paths = cfg.datasets
    if not paths:
        paths = sorted(p for p in glob.glob("dataset_pillar2_mid_v44*")
                       if p.endswith((".parquet", ".csv.gz", ".csv")))
    if not paths:
        sys.exit("No V44 datasets found. Pass --datasets.")

    if cfg.quick:
        PARAMS["n_estimators"] = 200

    print("=" * 94)
    print("V48 - WALK-FORWARD: IS IT REGIME CHANGE, OR IS THE CV OPTIMISTIC?")
    print("=" * 94)
    print("  loading:")
    d = load_all(paths)
    feats = [c for c in F.FEATURE_NAMES if c in d.columns]
    print(f"\n  combined: {len(d):,} rows | {d['ticker'].nunique()} tickers | "
          f"{d['_date'].min():%Y-%m} -> {d['_date'].max():%Y-%m} | {len(feats)} features")

    sectors, _, _, _, src = sector_map(sorted(d["ticker"].unique()), cfg.edgar_cache)
    d = add_targets(d, [cfg.horizon], sectors, rank_target=cfg.rank_target)
    print(f"  sectors: {d['sector'].nunique()} groups "
          f"({src[0]} from SIC, {src[1]} from the universe lists)")
    trn, tgt = f"trn_{cfg.target}_{cfg.horizon}", f"tgt_{cfg.target}_{cfg.horizon}"
    if trn not in d.columns:
        sys.exit(f"{trn} missing - check --horizon and --target.")
    print(f"  horizon {cfg.horizon} bars | target {cfg.target}-neutral | "
          f"embargo {cfg.embargo_days}d")
    print(f"  test windows of {cfg.window_years:g}y, first after "
          f"{cfg.min_train_years:g}y of training data")

    print("\n" + "=" * 94)
    print("WALK-FORWARD RESULTS  (every window is out of period)")
    print("=" * 94)

    summary = {}
    if cfg.mode in ("expanding", "both"):
        cfg.mode_rolling = False
        res_e = walk(d, feats, trn, tgt, cfg)
        summary["expanding"] = report_windows(res_e, "EXPANDING WINDOW (all history)")
        decay_curve(res_e, "expanding")
    if cfg.mode in ("rolling", "both"):
        cfg.mode_rolling = True
        res_r = walk(d, feats, trn, tgt, cfg)
        summary["rolling"] = report_windows(
            res_r, f"ROLLING WINDOW (most recent {cfg.train_years:g}y only)")
        decay_curve(res_r, "rolling")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)
    for k, (pos, tot) in summary.items():
        print(f"  {k:<10} positive in {pos}/{tot} windows")
    print("\n  Read it like this:")
    print("   most windows positive, only 2023-2026 negative")
    print("       -> regime change. Rolling retrain becomes a real candidate, and the")
    print("          recent period is genuinely different rather than the model broken.")
    print("   roughly half positive, none convincing")
    print("       -> the purged CV has been systematically optimistic all along. That")
    print("          is the single root cause behind all seven versions, and it closes")
    print("          the question properly.")
    print("   rolling clearly beats expanding")
    print("       -> relationships decay; train on recent data only and retrain often.")
    print("   the decay curve falling with years-ahead")
    print("       -> a half-life you can measure, which sets the retraining frequency.")


if __name__ == "__main__":
    main()
