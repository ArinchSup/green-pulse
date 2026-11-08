#!/usr/bin/env python3
"""
size_split_v50.py - was the early out-of-period IC real signal, or survivorship?

THE QUESTION

The walk-forward found IC that decays across fifteen years:

    2011-14  +0.0547     2017-20  +0.0101     2023-26  -0.0129
    2014-17  +0.0261     2020-23  +0.0200

Alpha decay explains that. So does survivorship bias fading as the test window
approaches 2026, the year the ticker list was written. The V49 audit showed the
second explanation has teeth: of companies filing in 2011, only ~41% were still
filing ten years later, and survival depends steeply on size - 23% in the smallest
assets decile against 70% in the largest.

THE TEST

Survivorship selection is SEVERE among small companies and MILD among large ones. A
small company in a 2026-built list beat roughly 3-to-1 odds to be there; a large one
beat 1.4-to-1. So if the early IC comes from the model learning "this company will
survive", the effect has to be concentrated in small caps.

    IC much higher in small caps   -> survivorship. The effect tracks selection
                                      intensity rather than anything economic.
    IC roughly uniform across size -> genuine signal. It does not care how selected
                                      the bucket is.
    IC higher in LARGE caps        -> neither; something else is going on, and the
                                      size story can be dropped.

With --pit it also plots IC against the population survival rate for each window's
horizon, taken from the V49 audit. If IC rises as survival falls, the two move
together and that is the survivorship signature directly.

REQUIRES  both V44 datasets, edgar_cache/ for sector labels, and optionally
pit_universe.csv.gz from survivorship_universe_v49.py. Runs offline.

USAGE
  python size_split_v50.py
  python size_split_v50.py --pit pit_universe.csv.gz
  python size_split_v50.py --terciles 5 --quick
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import pillar2_v43_features as F
from class_xgboost_v44 import PARAMS, ic_per_date, ic_stats
from sector_neutral_v47 import add_targets, sector_map
from walkforward_v48 import load_all

SIZE_COL = "log_dollar_vol"


def fmt(x, s="+.4f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{s}}"


# =============================================================================
# WALK FORWARD, KEEPING SIZE WITH EVERY PREDICTION
# =============================================================================
def walk_sized(d, feats, trn, tgt, cfg):
    start, end = d["_date"].min(), d["_date"].max()
    windows, t0 = [], start + pd.DateOffset(years=int(cfg.min_train_years))
    while t0 < end:
        t1 = min(t0 + pd.DateOffset(years=int(cfg.window_years)), end)
        if (t1 - t0).days > 300:
            windows.append((t0, t1))
        t0 = t1

    y = d[trn].to_numpy(dtype=float)
    a = d[tgt].to_numpy(dtype=float)
    size = pd.to_numeric(d[SIZE_COL], errors="coerce").to_numpy(dtype=float)
    X = d[feats]
    out = []
    for t0, t1 in windows:
        cutoff = t0 - pd.Timedelta(days=cfg.embargo_days)
        trm = np.asarray((d["_date"] < cutoff).to_numpy(), dtype=bool).copy()
        vam = ((d["_date"] >= t0) & (d["_date"] < t1)).to_numpy()
        ok = trm & np.isfinite(y)
        if ok.sum() < 5000 or vam.sum() < 500:
            continue
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[ok], y[ok], verbose=False)
        p = m.predict(X[vam])
        keep = np.isfinite(p) & np.isfinite(a[vam]) & np.isfinite(size[vam])
        out.append({
            "from": t0, "to": t1, "n_train": int(ok.sum()),
            "rows": pd.DataFrame({
                "date": d.loc[vam, "_date"].to_numpy()[keep],
                "pred": p[keep], "actual": a[vam][keep], "size": size[vam][keep]})})
        print(f"    {t0:%Y-%m}..{t1:%Y-%m}  trained on {ok.sum():,}, "
              f"scored {int(keep.sum()):,}")
    return out


# =============================================================================
# IC BY SIZE
# =============================================================================
def ic_of(df):
    if len(df) < 300:
        return np.nan, np.nan, 0
    _, ics = ic_per_date(df["date"].to_numpy(), df["pred"].to_numpy(),
                         df["actual"].to_numpy())
    st = ic_stats(ics)
    return st["ic"], st["t"], st["n"]


def tercile_within_date(rows, k):
    """Size buckets formed INSIDE each date, so the split is not a time trend."""
    r = rows.copy()
    r["b"] = (r.groupby("date")["size"]
               .transform(lambda s: pd.qcut(s.rank(method="first"), k,
                                            labels=False, duplicates="drop")))
    return r


def report(res, k):
    names = {0: "small", k - 1: "large"}
    print("\n" + "=" * 96)
    print(f"IC BY SIZE BUCKET, PER WINDOW  (buckets formed within each date)")
    print("=" * 96)
    hdr = f"   {'window':<22}{'all':>10}"
    for i in range(k):
        hdr += f"{names.get(i, 'mid' + str(i)):>12}"
    hdr += f"{'small-large':>14}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    gaps, per_bucket = [], {i: [] for i in range(k)}
    for r in res:
        rows = tercile_within_date(r["rows"], k)
        all_ic, _, _ = ic_of(rows)
        line = f"   {r['from']:%Y-%m}..{r['to']:%Y-%m}{'':<6}{fmt(all_ic):>10}"
        vals = {}
        for i in range(k):
            ic, _, _ = ic_of(rows[rows["b"] == i])
            vals[i] = ic
            if np.isfinite(ic):
                per_bucket[i].append(ic)
            line += f"{fmt(ic):>12}"
        gap = (vals.get(0, np.nan) - vals.get(k - 1, np.nan))
        if np.isfinite(gap):
            gaps.append(gap)
        line += f"{fmt(gap):>14}"
        print(line)

    print("\n   mean across windows:")
    line = f"   {'':<22}{'':<10}"
    for i in range(k):
        v = np.mean(per_bucket[i]) if per_bucket[i] else np.nan
        line += f"{fmt(v):>12}"
    line += f"{fmt(np.mean(gaps)) if gaps else 'n/a':>14}"
    print(line)
    return per_bucket, gaps


def pooled_gap(res, k, rounds=2000):
    """Small minus large, pooled over every window, with a date-clustered range."""
    rows = pd.concat([tercile_within_date(r["rows"], k) for r in res],
                     ignore_index=True)
    small, large = rows[rows["b"] == 0], rows[rows["b"] == k - 1]
    ds, _ = ic_per_date(small["date"].to_numpy(), small["pred"].to_numpy(),
                        small["actual"].to_numpy())
    dl, _ = ic_per_date(large["date"].to_numpy(), large["pred"].to_numpy(),
                        large["actual"].to_numpy())
    s_ic = pd.Series(_ic_map(small))
    l_ic = pd.Series(_ic_map(large))
    common = s_ic.index.intersection(l_ic.index)
    if len(common) < 20:
        return
    diff = (s_ic[common] - l_ic[common]).to_numpy()
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(rounds)]
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    print("\n" + "=" * 96)
    print("POOLED: SMALL MINUS LARGE")
    print("=" * 96)
    print(f"   small-cap IC {s_ic[common].mean():+.4f} | large-cap IC "
          f"{l_ic[common].mean():+.4f}")
    print(f"   difference {diff.mean():+.4f}   95% {lo:+.4f} to {hi:+.4f}   "
          f"t {t:+.2f}   ({len(common)} dates)")
    if lo > 0:
        print("   -> IC concentrates in SMALL caps, where survivorship selection is")
        print("      most severe. That is the survivorship signature.")
    elif hi < 0:
        print("   -> IC concentrates in LARGE caps, the opposite of what survivorship")
        print("      predicts. The size story does not explain the decay.")
    else:
        print("   -> no size concentration. IC does not track selection intensity,")
        print("      which argues the early result was not simply survivorship.")


def _ic_map(df):
    out = {}
    for dt, g in df.groupby("date"):
        if len(g) < 20:
            continue
        rp, ra = g["pred"].rank(), g["actual"].rank()
        if rp.std() == 0 or ra.std() == 0:
            continue
        out[dt] = float(np.corrcoef(rp, ra)[0, 1])
    return out


def vs_survival(res, k, pit_path):
    """Does IC track how selective the sample had to be for that window?"""
    if not os.path.exists(pit_path):
        print(f"\n   ({pit_path} not found - skipping the survival comparison)")
        return
    pit = pd.read_csv(pit_path)
    last = pit.groupby("cik")["period"].max()
    print("\n" + "=" * 96)
    print("IC AGAINST HOW SELECTIVE THE SAMPLE HAD TO BE")
    print("=" * 96)
    print("   'survival needed' is years from the window to 2026, the year the ticker")
    print("   list was built. 'population survived' is the share of real filers that")
    print("   lasted that long, from the V49 audit.\n")
    print(f"   {'window':<22}{'survival needed':>17}{'population survived':>21}{'IC':>10}")
    end_year = int(pit["y"].max())
    pairs = []
    for r in res:
        rows = tercile_within_date(r["rows"], k)
        ic, _, _ = ic_of(rows)
        wy = int(pd.Timestamp(r["from"]).year)
        need = 2026 - wy
        base = pit.loc[pit["y"] == wy, "cik"].unique()
        target = (wy + min(need, end_year - wy)) * 4
        surv = float((last.reindex(base).fillna(-1) >= target).mean()) if len(base) else np.nan
        print(f"   {r['from']:%Y-%m}..{r['to']:%Y-%m}{'':<6}{need:>15}y"
              f"{surv:>20.0%}{fmt(ic):>10}")
        if np.isfinite(ic) and np.isfinite(surv):
            pairs.append((surv, ic))
    if len(pairs) >= 4:
        s = np.array([p[0] for p in pairs])
        i = np.array([p[1] for p in pairs])
        rho = float(np.corrcoef(s, i)[0, 1])
        print(f"\n   correlation between population survival and IC: {rho:+.2f}")
        print("   Strongly NEGATIVE means IC is highest exactly where the sample had to")
        print("   be most selective - survivorship. Near zero means they are unrelated.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--pit", default="pit_universe.csv.gz")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--target", default="sector", choices=["date", "sector", "resid"])
    ap.add_argument("--terciles", type=int, default=3)
    ap.add_argument("--window-years", type=float, default=3.0)
    ap.add_argument("--min-train-years", type=float, default=6.0)
    ap.add_argument("--embargo-days", type=int, default=200)
    ap.add_argument("--edgar-cache", default="edgar_cache")
    ap.add_argument("--quick", action="store_true")
    cfg = ap.parse_args()

    paths = cfg.datasets or sorted(p for p in glob.glob("dataset_pillar2_mid_v44*")
                                   if p.endswith((".parquet", ".csv.gz", ".csv")))
    if not paths:
        sys.exit("No V44 datasets found. Pass --datasets.")
    if cfg.quick:
        PARAMS["n_estimators"] = 200

    print("=" * 96)
    print("V50 - IS THE EARLY IC SURVIVORSHIP? SPLIT IT BY SIZE")
    print("=" * 96)
    print("  loading:")
    d = load_all(paths)
    feats = [c for c in F.FEATURE_NAMES if c in d.columns]
    if SIZE_COL not in d.columns:
        sys.exit(f"{SIZE_COL} missing from the dataset - cannot split by size.")

    sectors, _, _, _, src = sector_map(sorted(d["ticker"].unique()), cfg.edgar_cache)
    d = add_targets(d, [cfg.horizon], sectors, rank_target=True)
    trn, tgt = f"trn_{cfg.target}_{cfg.horizon}", f"tgt_{cfg.target}_{cfg.horizon}"
    print(f"\n  {len(d):,} rows | horizon {cfg.horizon} | target {cfg.target}-neutral "
          f"| {d['sector'].nunique()} sectors")
    print(f"  size split: {cfg.terciles} buckets on {SIZE_COL}, formed within each date\n")

    res = walk_sized(d, feats, trn, tgt, cfg)
    if not res:
        sys.exit("No usable windows.")

    report(res, cfg.terciles)
    pooled_gap(res, cfg.terciles)
    vs_survival(res, cfg.terciles, cfg.pit)

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  Small caps in a 2026-built list beat roughly 3-to-1 odds to be there;")
    print("  large caps beat about 1.4-to-1. So survivorship contamination is")
    print("  concentrated in the small bucket by construction.")
    print("   IC much higher in small caps -> the early result was survivorship")
    print("   IC uniform across size       -> the early result was a genuine signal")
    print("                                   that has since decayed")


if __name__ == "__main__":
    main()
