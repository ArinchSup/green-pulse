#!/usr/bin/env python3
"""
sector_neutral_v47.py - does removing sector and size from the TARGET reveal a
stock-specific signal that V44 could not see?

THE ARGUMENT

V44 demeaned forward returns within each DATE, which removes the market. It did not
remove sector or size. So when the model ranked 290 names on a given day, much of the
remaining spread was "semiconductors had a good month" rather than anything specific
to a company. Industry-neutralisation is standard in factor models for exactly this
reason, and V44 skipped it.

This does NOT split the sample into per-sector models. Splitting 270k rows into 12
sector-by-size cells would leave ~22k each, and 240 tests would manufacture false
positives faster than it found anything - which is visible in the V41/V42/V43 scope
tables, where every group flips sign between versions. Here the sample stays whole and
only the TARGET changes.

THREE TARGETS, SAME FEATURES, SAME FOLDS

  date         fwd_ret minus the mean of that date           (what V44 used)
  sector       fwd_ret minus the mean of that date x sector
  residual     fwd_ret residualised on sector dummies AND log size, per date, by
               cross-sectional regression. This handles both at once without
               fragmenting into tiny cells.

IC is always measured against the SAME neutralisation the model was trained on, which
is the consistent comparison.

SECTION 1 DECIDES WHETHER THIS IS WORTH RUNNING AT ALL

It reports how much of each date's cross-sectional return variance sector explains. If
sector explains 30% or more, neutralising changes the problem materially. If it
explains 3%, the target barely moves and no result here will differ from V44.

SIC codes come free from the EDGAR submissions files already in edgar_cache/.

USAGE
  python sector_neutral_v47.py
  python sector_neutral_v47.py --horizons 5,20 --folds 5
  python sector_neutral_v47.py --oos dataset_..._oos....csv.gz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import pillar2_v43_features as F
from class_xgboost_v44 import PARAMS, folds, ic_per_date, ic_stats, load

EDGAR_CACHE = "edgar_cache"
MIN_SECTOR_TICKERS = 8      # smaller SIC groups get merged into "other"


# =============================================================================
# SECTORS FROM EDGAR
# =============================================================================
def universe_sectors():
    """
    The hand-written sector lists already in pillar2_v43_universe.py (_TECH, _FINANCE,
    _HEALTH, ...). They cover the whole training universe, whereas cached SIC codes
    only cover whatever tickers the EDGAR audit happened to fetch.
    """
    try:
        import pillar2_v43_universe as U
    except ImportError:
        return {}
    out = {}
    for name in dir(U):
        if not (name.startswith("_") and name[1:].isupper()):
            continue
        val = getattr(U, name)
        if isinstance(val, list) and val and isinstance(val[0], str):
            grp = name[1:].lower()
            for t in val:
                out.setdefault(t, grp)
    return out


def sector_map(tickers, cache_dir):
    """ticker -> 2-digit SIC major group, merging thin groups into 'other'."""
    tick_file = os.path.join(cache_dir, "company_tickers.json")
    if not os.path.exists(tick_file):
        sys.exit(f"{tick_file} missing. Run edgar_audit_v46.py once to populate the cache.")
    with open(tick_file, encoding="utf-8") as f:
        data = json.load(f)
    cik = {}
    for row in (data.values() if isinstance(data, dict) else data):
        t = str(row.get("ticker", "")).upper()
        if t:
            cik[t] = f"{int(row['cik_str']):010d}"

    raw, desc = {}, {}
    missing = []
    for t in tickers:
        c = cik.get(str(t).upper())
        if not c:
            missing.append(t)
            continue
        sub = os.path.join(cache_dir, f"sub_{c}.json")
        if not os.path.exists(sub):
            missing.append(t)
            continue
        try:
            with open(sub, encoding="utf-8") as f:
                js = json.load(f)
            sic = str(js.get("sic") or "")
            if not sic.isdigit():
                missing.append(t)
                continue
            raw[t] = sic[:2].zfill(2)
            desc[raw[t]] = js.get("sicDescription", "") or desc.get(raw[t], "")
        except Exception:
            missing.append(t)

    # Fall back to the universe file's own sector lists wherever SIC is unavailable.
    # Without this, a cache fetched with --universe deploy leaves most of the training
    # universe unlabelled and the "sector" variable stops varying, which makes the
    # whole neutralisation a no-op.
    fallback = universe_sectors()
    n_from_sic, n_from_list = len(raw), 0
    for t in list(missing):
        g = fallback.get(t)
        if g:
            raw[t] = g
            missing.remove(t)
            n_from_list += 1

    counts = pd.Series(raw).value_counts()
    keep = set(counts[counts >= MIN_SECTOR_TICKERS].index)
    out = {t: (g if g in keep else "other") for t, g in raw.items()}
    for t in missing:
        out[t] = "other"
    return out, desc, counts, missing, (n_from_sic, n_from_list)


# =============================================================================
# NEUTRALISATION
# =============================================================================
def size_band(d, q=5):
    """Size proxy already in the feature set, ranked inside each date."""
    col = "log_dollar_vol" if "log_dollar_vol" in d.columns else None
    if col is None:
        return pd.Series("all", index=d.index)
    return (d.groupby("_date")[col]
             .transform(lambda s: pd.qcut(s.rank(method="first"), q,
                                          labels=False, duplicates="drop"))
             .fillna(-1).astype(int).astype(str))


def residualise(group, ycol, sector_col, size_col):
    """
    Cross-sectional regression of the forward return on sector dummies and log size,
    for ONE date. The residual is what is left after both are removed - no cell
    fragmentation, unlike demeaning on sector x size jointly.
    """
    y = group[ycol].to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return pd.Series(np.nan, index=group.index)
    dummies = pd.get_dummies(group[sector_col], drop_first=True).to_numpy(dtype=float)
    size = pd.to_numeric(group[size_col], errors="coerce").to_numpy(dtype=float)
    size = np.nan_to_num(size, nan=float(np.nanmean(size)) if np.isfinite(size).any() else 0.0)
    X = np.column_stack([np.ones(len(group)), dummies, size])
    try:
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=group.index)
    resid = np.full(len(group), np.nan)
    resid[ok] = y[ok] - X[ok] @ beta
    return pd.Series(resid, index=group.index)


def add_targets(d, horizons, sectors, rank_target=False):
    d["sector"] = d["ticker"].map(sectors).fillna("other")
    d["size_b"] = size_band(d)
    if "log_dollar_vol" not in d.columns:
        d["log_dollar_vol"] = 0.0
    for h in horizons:
        base = f"fwd_ret_{h}"
        if base not in d.columns:
            continue
        _ = base
        # date-neutral is what V44 used; recompute so all three are built identically
        d[f"tgt_date_{h}"] = d[base] - d.groupby("_date")[base].transform("mean")
        d[f"tgt_sector_{h}"] = (d[base]
                                - d.groupby(["_date", "sector"])[base].transform("mean"))
        d[f"tgt_resid_{h}"] = (d.groupby("_date", group_keys=False)
                                .apply(lambda g: residualise(g, base, "sector", "log_dollar_vol"),
                                       include_groups=False))
        # The TRAINING target may be a rank (V44's specification, robust to a single
        # +200% move dominating the loss) while IC is always measured against the
        # continuous neutralised return above.
        for kind in ("date", "sector", "resid"):
            tgt = f"tgt_{kind}_{h}"
            d[f"trn_{kind}_{h}"] = (d.groupby("_date")[tgt].rank(pct=True)
                                    if rank_target else d[tgt])
    return d


# =============================================================================
# DIAGNOSTIC: HOW MUCH IS SECTOR?
# =============================================================================
def variance_share(d, horizons):
    print("\n" + "=" * 92)
    print("1) HOW MUCH OF THE CROSS-SECTION IS SECTOR AND SIZE?")
    print("=" * 92)
    print("   Share of each date's return variance explained by sector dummies, and by")
    print("   sector plus size. Neutralising only matters if these are large.\n")
    print(f"   {'horizon':>8}{'sector R2':>12}{'sector+size R2':>17}{'dates':>8}")
    for h in horizons:
        base, rs, rr = f"fwd_ret_{h}", [], []
        if base not in d.columns:
            continue
        for _, g in d.groupby("_date"):
            y = g[base].to_numpy(dtype=float)
            ok = np.isfinite(y)
            if ok.sum() < 30 or np.nanstd(y[ok]) == 0:
                continue
            tot = np.var(y[ok])
            sec = g[f"tgt_sector_{h}"].to_numpy(dtype=float)
            res = g[f"tgt_resid_{h}"].to_numpy(dtype=float)
            if np.isfinite(sec[ok]).all():
                rs.append(1 - np.var(sec[ok]) / tot)
            m = ok & np.isfinite(res)
            if m.sum() > 20:
                rr.append(1 - np.var(res[m]) / np.var(y[m]))
        print(f"   {h:>8}{np.mean(rs) if rs else np.nan:>11.1%}"
              f"{np.mean(rr) if rr else np.nan:>16.1%}{len(rs):>8}")
    print("\n   Above ~25%: sector dominates the cross-section and neutralising changes")
    print("   the problem. Below ~10%: the target barely moves and this will reproduce")
    print("   V44's result whatever the model does.")


# =============================================================================
# CV
# =============================================================================
def run_cv(d, feats, horizon, kind, cfg):
    tcol, acol = f"trn_{kind}_{horizon}", f"tgt_{kind}_{horizon}"
    if tcol not in d.columns or acol not in d.columns:
        return None
    y = d[tcol].to_numpy(dtype=float)
    dates = np.array(sorted(d["_date"].unique()))
    X = d[feats]
    oof = np.full(len(d), np.nan)
    for tr, va in folds(dates, cfg.folds, cfg.embargo_days):
        trm = d["_date"].isin(tr).to_numpy()
        vam = d["_date"].isin(va).to_numpy()
        ok = np.isfinite(y) & trm
        if ok.sum() < 1000:
            continue
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[ok], y[ok], verbose=False)
        oof[vam] = m.predict(X[vam])
    mask = np.isfinite(oof) & np.isfinite(d[acol].to_numpy())
    if mask.sum() < 500:
        return None
    _, ics = ic_per_date(d.loc[mask, "_date"].to_numpy(), oof[mask],
                         d.loc[mask, acol].to_numpy())
    st = ic_stats(ics)
    st["kind"], st["horizon"], st["rows"] = kind, horizon, int(mask.sum())
    st["oof"] = oof
    st["mask"] = mask
    return st


def fmt(x, s="+.4f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{s}}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--oos", default=None,
                    help="out-of-sample V44 dataset; runs the OOS check on the best target")
    ap.add_argument("--horizons", default="5,10,20,60")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=200)
    ap.add_argument("--edgar-cache", default=EDGAR_CACHE)
    ap.add_argument("--rank-target", action="store_true",
                    help="train on the within-group percentile rank instead of the raw "
                         "neutralised return, matching V44's specification. Raw returns "
                         "let one +200% move dominate the loss.")
    cfg = ap.parse_args()

    path = cfg.dataset
    if not path:
        hits = [p for p in glob.glob("dataset_pillar2_mid_v44_*")
                if p.endswith((".parquet", ".csv.gz", ".csv")) and "oos" not in p]
        if not hits:
            sys.exit("No V44 dataset found. Pass --dataset.")
        path = sorted(hits)[-1]
    d, meta = load(path)
    feats = [c for c in F.FEATURE_NAMES if c in d.columns]
    horizons = [int(h) for h in cfg.horizons.split(",") if h.strip()]

    print("=" * 92)
    print("V47 - SECTOR- AND SIZE-NEUTRAL TARGETS")
    print("=" * 92)
    print(f"  dataset: {path}")
    print(f"  {len(d):,} rows | {d['ticker'].nunique()} tickers | "
          f"{d['signal_date'].nunique()} dates | {len(feats)} features")
    print(f"  training target: {'within-date rank (V44 spec)' if cfg.rank_target else 'raw neutralised return'}")

    sectors, desc, counts, missing, src = sector_map(sorted(d["ticker"].unique()),
                                                     cfg.edgar_cache)
    d = add_targets(d, horizons, sectors, rank_target=cfg.rank_target)
    n_sec = d["sector"].nunique()
    print(f"\n  sectors from SIC: {n_sec} groups "
          f"(2-digit SIC, groups under {MIN_SECTOR_TICKERS} tickers merged into 'other')")
    top = d.groupby("sector")["ticker"].nunique().sort_values(ascending=False)
    for s, n in top.head(8).items():
        print(f"    {s:<8}{n:>4} tickers   {desc.get(s, '')[:46]}")
    print(f"    source: {src[0]} from SIC, {src[1]} from the universe file's lists")
    if missing:
        print(f"    {len(missing)} tickers had neither and went to 'other'")
    share_other = float((d["sector"] == "other").mean())
    if share_other > 0.35 or n_sec < 5:
        print(f"\n  !! {share_other:.0%} of rows are in 'other' across only {n_sec} groups.")
        print(f"     The sector variable barely varies, so neutralising it cannot do")
        print(f"     much and section 1 will understate the true sector share. Fetch the")
        print(f"     rest with:  python edgar_audit_v46.py --email <you> --universe all")

    variance_share(d, horizons)

    print("\n" + "=" * 92)
    print("2) CROSS-VALIDATED IC BY TARGET")
    print("=" * 92)
    print(f"  {cfg.folds} folds, {cfg.embargo_days}-day embargo, split by date.")
    print("  IC is measured against the same neutralisation the model trained on.\n")
    print(f"  {'horizon':>8}{'target':>12}{'IC':>10}{'95% range':>22}{'ICIR':>8}{'t':>8}")
    best, results = None, {}
    for h in horizons:
        for kind in ("date", "sector", "resid"):
            st = run_cv(d, feats, h, kind, cfg)
            if st is None:
                continue
            results[(h, kind)] = st
            rng = f"{st['lo']:+.4f} to {st['hi']:+.4f}"
            flag = "  <-- clear of zero" if np.isfinite(st["lo"]) and st["lo"] > 0 else ""
            print(f"  {h:>8}{kind:>12}{fmt(st['ic']):>10}{rng:>22}"
                  f"{fmt(st['icir'], '+.2f'):>8}{fmt(st['t'], '+.2f'):>8}{flag}")
            if best is None or (np.isfinite(st["ic"]) and st["ic"] > results[best]["ic"]):
                best = (h, kind)
        print()

    if best:
        b = results[best]
        print(f"  best: {best[0]} bars, {best[1]}-neutral   IC {fmt(b['ic'])} "
              f"(t {fmt(b['t'], '+.2f')})")
        same_h = [(k, results[(best[0], k)]["ic"]) for k in ("date", "sector", "resid")
                  if (best[0], k) in results]
        print("  at that horizon: " + "  ".join(f"{k} {v:+.4f}" for k, v in same_h))
        print("\n  What matters is whether sector- or resid-neutral BEATS date-neutral.")
        print("  If all three land together, sector was never the thing in the way.")

    # ---- out of sample ----
    if cfg.oos and best:
        print("\n" + "=" * 92)
        print("3) OUT OF SAMPLE  (the gate V43 and V44 both failed)")
        print("=" * 92)
        o, _ = load(cfg.oos)
        o = add_targets(o, [best[0]], sectors, rank_target=cfg.rank_target)
        h, kind = best
        trn, tgt = f"trn_{kind}_{h}", f"tgt_{kind}_{h}"
        ok = np.isfinite(d[trn].to_numpy())
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(d.loc[ok, feats], d.loc[ok, trn], verbose=False)
        p = model.predict(o[feats])
        m = np.isfinite(p) & np.isfinite(o[tgt].to_numpy())
        _, ics = ic_per_date(o.loc[m, "_date"].to_numpy(), p[m], o.loc[m, tgt].to_numpy())
        st = ic_stats(ics)
        print(f"  {int(m.sum()):,} rows | {o['signal_date'].min()} -> {o['signal_date'].max()}")
        print(f"  CV IC  {fmt(results[best]['ic'])}")
        print(f"  OOS IC {fmt(st['ic'])}   95% {fmt(st['lo'])} to {fmt(st['hi'])}"
              f"   t {fmt(st['t'], '+.2f')}   ({st['n']} dates)")
        overlap = max(1.0, h / 5)
        print(f"  overlap-adjusted t: {fmt(st['t'] / np.sqrt(overlap), '+.2f')}")
        if np.isfinite(st["lo"]) and st["lo"] > 0:
            print("\n  HELD. Sector-neutralisation revealed a signal that survives out of")
            print("  sample - the first thing in this project that has.")
        else:
            print("\n  Did not hold. Same outcome as V43 and V44: in-sample structure,")
            print("  nothing out of sample.")

    print("\n" + "=" * 92)
    print("  Read section 1 first. If sector explains little of the cross-section,")
    print("  section 2 cannot differ much from V44 no matter what the model does.")


if __name__ == "__main__":
    main()
