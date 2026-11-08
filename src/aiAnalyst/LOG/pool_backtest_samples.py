#!/usr/bin/env python3
"""
pool_backtest_samples.py - pool the per-sample CSVs from class_ai_pipeline_backtest_v2.py
across seeds and report STRICT win rate, EV and stock-month-clustered 95% ranges.

Why this exists
  1. Resolved WR = W/(W+L) drops breakevens and expiries out of the denominator, which is
     what turned a 40% strict win rate into a 56% headline. Strict WR = W/(W+L+BE+Exp).
  2. One seed gives too few model trades to tell 45% from 60%. Pooling seeds raises n.
  3. Two seeds draw from the SAME pool of ticker-days, so pooling naively double-counts
     rows both seeds happened to draw. This de-duplicates on (ticker, signal_date) and
     tells you how many NEW stock-months each extra seed actually contributed - so you
     can see when adding another seed has stopped buying you anything.
  4. Nearby days on one stock are near-copies, so every 95% range here resamples whole
     stock-months (the CSV's own `cluster` column), not individual rows.

Usage
  python pool_backtest_samples.py
  python pool_backtest_samples.py --model V40
  python pool_backtest_samples.py --model V40 --buckets 0.50,0.60,0.65,0.70,0.75,0.80
  python pool_backtest_samples.py --model V40 --quantiles 5 --out pooled_V40.csv
  python pool_backtest_samples.py --dir backtest_runs --rounds 5000
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# run_id built by the backtest:
#   {model_label}_{universe}_seed{seed}_{YYYYMMDD}_{levels}[_nobe]_samples.csv
RUN_RE = re.compile(
    r"^(?P<model>.+)_(?P<universe>SHAY|TECH_ONLY|SP500)_seed(?P<seed>\d+)"
    # [A-Z_]+ rather than a fixed list, so a new levels mode never breaks the parser
    r"_(?P<end>\d{8})_(?P<levels>[A-Z0-9_]+?)(?P<nobe>_nobe)?_samples\.csv$"
)

WIN, LOSS, BE, EXP = "Win", "Loss", "Breakeven", "Expired"
OUTCOMES = (WIN, LOSS, BE, EXP)


# =============================================================================
# LOADING
# =============================================================================
def load_runs(directory, model_filter=None):
    """Read every *_samples.csv, tag it with its model label and seed."""
    paths = sorted(glob.glob(os.path.join(directory, "*_samples.csv")))
    if not paths:
        sys.exit(f"No *_samples.csv found in '{directory}'. "
                 f"Run the backtest first, or pass --dir.")

    frames, skipped = [], []
    for p in paths:
        m = RUN_RE.match(os.path.basename(p))
        if not m:
            skipped.append(os.path.basename(p))
            continue
        if model_filter and m.group("model") != model_filter:
            continue
        df = pd.read_csv(p)
        df["model_label"] = m.group("model")
        df["universe"] = m.group("universe")
        df["seed"] = int(m.group("seed"))
        df["levels"] = m.group("levels")
        df["breakeven_rule"] = m.group("nobe") is None
        df["source_file"] = os.path.basename(p)
        frames.append(df)

    for name in skipped:
        print(f"  note: filename didn't parse, skipped: {name}")
    if not frames:
        sys.exit(f"No runs matched model '{model_filter}'. "
                 f"Available: {sorted({RUN_RE.match(os.path.basename(p)).group('model') for p in paths if RUN_RE.match(os.path.basename(p))})}")
    return pd.concat(frames, ignore_index=True)


def check_comparable(df):
    """Refuse to pool runs whose exit rules or universe differ - the trades aren't alike."""
    problems = []
    for col, label in [("universe", "universe"), ("levels", "levels mode"),
                       ("breakeven_rule", "breakeven rule")]:
        vals = sorted(df[col].astype(str).unique())
        if len(vals) > 1:
            problems.append(f"{label}: {vals}")
    if problems:
        print("\n!! Runs in this folder are NOT directly comparable:")
        for p in problems:
            print(f"     {p}")
        print("   Filter with --model, or move the odd runs out of the folder.\n")


def dedupe(df):
    """
    Two seeds draw from the same ticker-days. A row both seeds drew is the same trade
    with the same outcome - counting it twice fakes extra evidence.
    """
    before = len(df)
    df = df.sort_values(["seed", "attempt"]).drop_duplicates(
        subset=["model_label", "ticker", "signal_date"], keep="first")
    return df.reset_index(drop=True), before - len(df)


def coverage_report(df):
    """How many NEW stock-months each seed added - the answer to 'do I need more seeds?'."""
    print("\n" + "=" * 78)
    print("SEED COVERAGE  (a seed that adds few new stock-months adds little evidence)")
    print("=" * 78)
    print(f"{'Seed':>6}{'Rows':>8}{'Stock-months':>14}{'New':>7}{'Cumulative':>12}"
          f"{'Model fires':>13}{'New (model)':>13}")

    seen, seen_model = set(), set()
    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed]
        cl = set(sub["cluster"])
        mcl = set(sub.loc[sub["is_model"] == True, "cluster"])  # noqa: E712
        new, new_m = len(cl - seen), len(mcl - seen_model)
        seen |= cl
        seen_model |= mcl
        print(f"{seed:>6}{len(sub):>8}{len(cl):>14}{new:>7}{len(seen):>12}"
              f"{int((sub['is_model'] == True).sum()):>13}{new_m:>13}")  # noqa: E712

    print(f"\n  Pooled: {len(seen)} distinct stock-months, "
          f"{len(seen_model)} of them with a model signal.")
    print("  If the last seed's 'New' column is small, another seed will mostly redraw")
    print("  ticker-days you already have and the 95% ranges below will barely move.")


def training_overlap_warning(df):
    if "in_training_set" not in df.columns:
        return
    n = int(df["in_training_set"].astype(str).str.lower().isin(["true", "1"]).sum())
    if n:
        print(f"\n!! {n} pooled rows are exact training rows (same ticker, same day). "
              f"Those results are not out-of-sample.")


# =============================================================================
# METRICS
# =============================================================================
def strict_wr(outcomes):
    """W / (W + L + BE + Exp) - breakevens and expiries stay in the denominator."""
    n = len(outcomes)
    return float((outcomes == WIN).sum()) / n * 100.0 if n else np.nan


def resolved_wr(outcomes):
    w = int((outcomes == WIN).sum())
    l = int((outcomes == LOSS).sum())
    return w / (w + l) * 100.0 if (w + l) else np.nan


def counts(outcomes):
    return {o: int((outcomes == o).sum()) for o in OUTCOMES}


def bucket_stats(out, ret):
    c = counts(out)
    return {
        "n": len(out),
        **c,
        "strict_wr": strict_wr(out),
        "resolved_wr": resolved_wr(out),
        "mean_ret": float(np.nanmean(ret)) if len(ret) else np.nan,
        "total_ret": float(np.nansum(ret)) if len(ret) else np.nan,
    }


# =============================================================================
# CLUSTERED BOOTSTRAP
# =============================================================================
class ClusterBoot:
    """
    Resamples whole stock-months with replacement. Every bucket is recomputed on the
    SAME resample each round, so bucket-minus-baseline gaps stay paired.
    """

    def __init__(self, df, rounds=2000, seed=0):
        self.rounds = rounds
        self.rng = np.random.default_rng(seed)
        groups = df.groupby("cluster").indices           # cluster -> row positions
        self.keys = list(groups)
        self.idx = [np.asarray(groups[k]) for k in self.keys]

    def resamples(self):
        k = len(self.idx)
        for _ in range(self.rounds):
            pick = self.rng.integers(0, k, size=k)
            yield np.concatenate([self.idx[i] for i in pick])


def ranged(values, lo=2.5, hi=97.5):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def bootstrap_table(df, out_col, ret_col, labels, rounds, seed):
    """95% ranges for strict WR, mean return, and mean return minus the pooled baseline."""
    boot = ClusterBoot(df, rounds=rounds, seed=seed)
    out = df[out_col].to_numpy()
    ret = pd.to_numeric(df[ret_col], errors="coerce").to_numpy(dtype=float)
    lab = labels.to_numpy()
    uniq = bucket_order(labels)

    acc = {u: {"wr": [], "ret": [], "gap": []} for u in uniq}
    for rows in boot.resamples():
        o, r, l = out[rows], ret[rows], lab[rows]
        base = np.nanmean(r) if np.isfinite(r).any() else np.nan
        for u in uniq:
            m = l == u
            if not m.any():
                continue
            acc[u]["wr"].append(strict_wr(o[m]))
            mr = np.nanmean(r[m]) if np.isfinite(r[m]).any() else np.nan
            acc[u]["ret"].append(mr)
            acc[u]["gap"].append(mr - base)

    return {u: {k: ranged(v) for k, v in d.items()} for u, d in acc.items()}


# =============================================================================
# BUCKETING
# =============================================================================
def make_buckets(conf, edges=None, quantiles=None):
    conf = pd.to_numeric(conf, errors="coerce")
    if quantiles:
        q = pd.qcut(conf, quantiles, duplicates="drop")
        return q.cat.rename_categories([str(c) for c in q.cat.categories])
    cuts = [-np.inf] + list(edges) + [np.inf]
    names = []
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        if lo == -np.inf:
            names.append(f"< {hi:.2f}")
        elif hi == np.inf:
            names.append(f">= {lo:.2f}")
        else:
            names.append(f"{lo:.2f}-{hi:.2f}")
    # ordered Categorical, so buckets print low-to-high rather than alphabetically
    return pd.cut(conf, cuts, labels=names, right=False, ordered=True)


def bucket_order(labels):
    """Buckets that actually occur, in numeric order."""
    present = set(labels.dropna().unique())
    if hasattr(labels, "cat"):
        return [c for c in labels.cat.categories if c in present]
    return sorted(present)


# =============================================================================
# REPORTING
# =============================================================================
def fmt(x, spec=".2f", suffix=""):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def print_table(title, note, df, out_col, ret_col, labels, rounds, seed, min_n, sink):
    sink.append("")
    sink.append("=" * 110)
    sink.append(title)
    sink.append("=" * 110)
    sink.append(note)
    sink.append("")

    baseline = pd.to_numeric(df[ret_col], errors="coerce").mean()
    ranges = bootstrap_table(df, out_col, ret_col, labels, rounds, seed)

    header = (f"{'Bucket':<14}{'n':>6}{'W':>5}{'L':>5}{'BE':>5}{'Exp':>5}"
              f"{'StrictWR':>10}{'95% range':>18}{'Resolved':>10}"
              f"{'MeanRet':>10}{'95% range':>18}{'vs base':>10}{'95% range':>18}")
    sink.append(header)
    sink.append("-" * len(header))

    rows_out = []
    for u in bucket_order(labels):
        m = labels == u
        sub = df[m]
        if sub.empty:
            continue
        st = bucket_stats(sub[out_col].to_numpy(),
                          pd.to_numeric(sub[ret_col], errors="coerce").to_numpy(dtype=float))
        rg = ranges.get(u, {})
        wr_lo, wr_hi = rg.get("wr", (np.nan, np.nan))
        rt_lo, rt_hi = rg.get("ret", (np.nan, np.nan))
        gp_lo, gp_hi = rg.get("gap", (np.nan, np.nan))
        gap = st["mean_ret"] - baseline

        flag = "  <- n too small to trust" if st["n"] < min_n else ""
        sink.append(
            f"{u:<14}{st['n']:>6}{st[WIN]:>5}{st[LOSS]:>5}{st[BE]:>5}{st[EXP]:>5}"
            f"{fmt(st['strict_wr'], '.1f', '%'):>10}"
            f"{fmt(wr_lo, '.1f') + ' to ' + fmt(wr_hi, '.1f'):>18}"
            f"{fmt(st['resolved_wr'], '.1f', '%'):>10}"
            f"{fmt(st['mean_ret'], '+.2f'):>10}"
            f"{fmt(rt_lo, '+.2f') + ' to ' + fmt(rt_hi, '+.2f'):>18}"
            f"{fmt(gap, '+.2f'):>10}"
            f"{fmt(gp_lo, '+.2f') + ' to ' + fmt(gp_hi, '+.2f'):>18}{flag}")

        rows_out.append({"bucket": u, **st, "strict_wr_lo": wr_lo, "strict_wr_hi": wr_hi,
                         "mean_ret_lo": rt_lo, "mean_ret_hi": rt_hi,
                         "gap_vs_base": gap, "gap_lo": gp_lo, "gap_hi": gp_hi})

    overall = bucket_stats(df[out_col].to_numpy(),
                           pd.to_numeric(df[ret_col], errors="coerce").to_numpy(dtype=float))
    sink.append("-" * len(header))
    sink.append(f"{'ALL':<14}{overall['n']:>6}{overall[WIN]:>5}{overall[LOSS]:>5}"
                f"{overall[BE]:>5}{overall[EXP]:>5}{fmt(overall['strict_wr'], '.1f', '%'):>10}"
                f"{'':>18}{fmt(overall['resolved_wr'], '.1f', '%'):>10}"
                f"{fmt(overall['mean_ret'], '+.2f'):>10}")
    sink.append("")
    sink.append("  A bucket is only evidence of a sniper if its StrictWR range sits clear of")
    sink.append("  the ALL row, and its 'vs base' range excludes zero.")
    return rows_out


def print_seed_split(df, out_col, ret_col, mask, sink):
    """Run-to-run variance: the same metric computed per seed."""
    sub = df[mask]
    if sub.empty:
        return
    sink.append("")
    sink.append("SEED-BY-SEED (model trades only) - how much this metric moves between runs")
    sink.append(f"  {'Seed':>6}{'n':>6}{'StrictWR':>10}{'Resolved':>10}{'MeanRet':>10}")
    for seed in sorted(sub["seed"].unique()):
        s = sub[sub["seed"] == seed]
        st = bucket_stats(s[out_col].to_numpy(),
                          pd.to_numeric(s[ret_col], errors="coerce").to_numpy(dtype=float))
        sink.append(f"  {seed:>6}{st['n']:>6}{fmt(st['strict_wr'], '.1f', '%'):>10}"
                    f"{fmt(st['resolved_wr'], '.1f', '%'):>10}{fmt(st['mean_ret'], '+.2f'):>10}")
    sink.append("  (these are BEFORE de-duplication, so they are the raw per-run numbers)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Pool backtest v2 sample CSVs across seeds.")
    p.add_argument("--dir", default="backtest_runs")
    p.add_argument("--model", default=None, help="model label, e.g. V40 or V41")
    p.add_argument("--buckets", default="0.50,0.60,0.65,0.70,0.75,0.80",
                   help="confidence cut points")
    p.add_argument("--quantiles", type=int, default=None,
                   help="use N equal-count buckets instead of fixed cut points")
    p.add_argument("--rounds", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--min-n", type=int, default=30,
                   help="flag buckets smaller than this as untrustworthy")
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--out", default=None, help="write the pooled bucket table to this CSV")
    a = p.parse_args()

    raw = load_runs(a.dir, a.model)
    check_comparable(raw)

    print("=" * 110)
    print(f"POOLED BACKTEST SAMPLES   dir={a.dir}   models={sorted(raw['model_label'].unique())}")
    print(f"  files: {len(raw['source_file'].unique())}   raw rows: {len(raw)}")
    print("=" * 110)

    coverage_report(raw)

    raw_model_mask = raw["is_model"] == True  # noqa: E712
    seed_sink = []
    print_seed_split(raw, "model_outcome", "model_ret", raw_model_mask, seed_sink)
    print("\n".join(seed_sink))

    if a.no_dedupe:
        df, dropped = raw, 0
    else:
        df, dropped = dedupe(raw)
    print(f"\n  De-duplicated on (ticker, signal_date): {dropped} duplicate rows dropped, "
          f"{len(df)} kept.")
    training_overlap_warning(df)

    edges = [float(x) for x in a.buckets.split(",") if x.strip()]
    sink = []

    # --- Table A: does the score rank at all? (every sample, model or not) ---
    all_labels = make_buckets(df["confidence"], edges, a.quantiles)
    rows_a = print_table(
        "A) EVERY SAMPLE BY SCORE  -  does a higher score mean a better outcome?",
        "   Uses label_outcome / rand_ret: the trade you'd get taking EVERY sampled day.\n"
        "   This is the ranking test. If StrictWR is flat across buckets, no threshold\n"
        "   anywhere will produce a sniper.",
        df, "label_outcome", "rand_ret", all_labels, a.rounds, a.boot_seed, a.min_n, sink)

    # --- Table B: the sniper's actual record ---
    fired = df[(df["is_model"] == True) & df["model_outcome"].isin(OUTCOMES)]  # noqa: E712
    rows_b = []
    if fired.empty:
        sink.append("\nB) MODEL-FIRED TRADES: none in these files.")
    else:
        fired_labels = make_buckets(fired["confidence"], edges, a.quantiles)
        rows_b = print_table(
            "B) MODEL-FIRED TRADES ONLY  -  the sniper's actual record",
            "   Uses model_outcome / model_ret, the trades the model actually took.",
            fired, "model_outcome", "model_ret", fired_labels,
            a.rounds, a.boot_seed, a.min_n, sink)

        base_all = pd.to_numeric(df["rand_ret"], errors="coerce")
        base_out = df["label_outcome"].to_numpy()
        sink.append("")
        sink.append(f"  Bar to beat (every sampled day, no model): "
                    f"strict WR {fmt(strict_wr(base_out), '.1f', '%')}, "
                    f"mean return {fmt(base_all.mean(), '+.2f')}%")
        sink.append(f"  Selectivity: {len(fired)} trades from {len(df)} samples "
                    f"({len(fired) / len(df) * 100:.1f}% of days)")

    print("\n".join(sink))

    if a.out:
        out = pd.concat([pd.DataFrame(rows_a).assign(table="all_samples"),
                         pd.DataFrame(rows_b).assign(table="model_fired")],
                        ignore_index=True)
        out.to_csv(a.out, index=False)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
