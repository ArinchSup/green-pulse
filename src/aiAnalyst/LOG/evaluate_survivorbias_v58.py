#!/usr/bin/env python3
"""
evaluate_survivorbias_v58.py - put a number on the caveat.

THE PROBLEM, STATED PROPERLY

Every figure in V52 through V57 is calibrated on tickers that still exist. The
companies absent from the price cache are the ones that collapsed, so drawdowns
are under-represented, the probability is understated, and positions sized from it
come out too large. For a risk model that is the unsafe direction, and until now
it has been a paragraph of prose - including in the findings document.

It can be arithmetic instead.

WHAT CAN AND CANNOT BE COMPUTED

Computable, from the SEC filing record V49 already collected: how many companies
existed at each past date, and what share of them were still filing by the end.
That gives the MISSING FRACTION, and it is not one number - it grows the further
back you look. An observation from 2005 has twenty years of subsequent attrition
in front of it; one from 2024 has two.

Not computable from this data: whether the absent companies actually had a 30%
drawdown. Some were acquired, often at a premium and with no drawdown at all;
others went to zero. So the honest output is not a bound, it is a SENSITIVITY: the
true base rate as a function of that unknown rate. A worst-case bound would be
[4%, 63%] and would tell you nothing.

THE SIZE CORRECTION THAT MATTERS

V49 measured survival running from about 23% in the smallest decile to 70% in the
largest. The deployed universe is 337 large and mid caps, so the relevant
counterfactual is not "all companies" - it is "companies like these". Using the
population average would roughly double the apparent bias. The size bucket is
therefore a parameter here, defaulting to the large end, and the population figure
is shown beside it so the difference is visible rather than buried.

THE PART THAT BEARS ON AN OPEN QUESTION

Because the missing fraction grows with age, the bias is WORST IN THE EARLY YEARS
of the sample - which is exactly where V48 found its strongest information
coefficient (+0.0547 in 2011-14, decaying to zero by 2023-26). V50 could not
separate survivorship from alpha decay as the explanation. A bias that is
mechanically larger early is not proof, but it is a quantitative reason to prefer
the survivorship reading, and it is the first handle anything in this project has
had on that question.

USAGE
  python evaluate_survivorbias_v58.py --offline
  python evaluate_survivorbias_v58.py --email you@yourdomain.com --start 2005
  python evaluate_survivorbias_v58.py --offline --survival-bucket population
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# The observed rates being corrected. Defaults are the deployed calibration's.
OBS_BASE_RATE = 0.107           # V53 recalibration, 160,412 observations
OBS_CAL_MAE_PP = 2.15           # V52/V53 out-of-sample calibration error
# Event rates among absent companies to tabulate. Unknowable from this data, so
# the table spans the whole range and the reader picks.
Q_GRID = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]


# =============================================================================
# SURVIVAL
# =============================================================================
def usable_span(d, min_share=0.5):
    """
    The part of the filing record that is actually populated.

    Two edges make the raw record unusable at both ends. XBRL was phased in around
    2009-2011, so the earliest years hold a few hundred filers rather than a
    market. And the most recent quarters are still filling up - a CY2026Q4
    instantaneous fact does not exist in September 2026 - so almost nobody's last
    filing lands there.

    A first version of this took the record's final quarter as the reference for
    "still filing", which required a company to have filed in that barely
    populated quarter. Every survival rate came back 0%, including for the current
    year, which is impossible by construction. Both edges are cut here by filer
    count instead, and the chosen span is printed so the choice is auditable.
    """
    cnt = d.groupby("period")["cik"].nunique().sort_index()
    if cnt.empty:
        return None, None, cnt
    peak = float(cnt.max())
    good = cnt[cnt >= min_share * peak]
    if good.empty:
        return None, None, cnt
    return int(good.index.min()), int(good.index.max()), cnt


def survival_by_year(d, bucket, end_period, min_year_share=0.5, peak_n=None):
    """
    For each year, the share of that year's filers still filing at the END OF THE
    USABLE RECORD - within a size bucket, because attrition depends strongly on
    size and the deployed universe is not a random draw from the population.

    "Still filing" allows a year of slack: a company whose last filing is within
    four quarters of the reference counts as alive, since filing cadence varies
    and an exact match on one quarter measures reporting timing, not survival.
    """
    last = d.groupby("cik")["period"].max()
    rows = []
    for y in sorted(d["y"].unique()):
        sub = d[d["y"] == y]
        n_y = sub["cik"].nunique()
        # a year holding a small fraction of the peak filer count is a phase-in
        # artefact, not a cohort
        if n_y < 50 or (peak_n and n_y < min_year_share * peak_n):
            continue
        med = sub.groupby("cik")["val"].median()
        med = med[med > 0]
        if len(med) < 50:
            continue
        if bucket == "population":
            ciks = med.index
        else:
            # top three deciles by assets - the part of the population the
            # deployed universe actually resembles
            cut = med.quantile(0.70)
            ciks = med[med >= cut].index
        alive = float((last.reindex(ciks).fillna(-1) >= end_period - 3).mean())
        rows.append({"year": int(y), "n": len(ciks), "survived": alive,
                     "missing": 1.0 - alive})
    return pd.DataFrame(rows)


def implied_rate(p_obs, missing, q):
    """
    True population rate, given the observed survivor rate, the missing fraction
    and an assumed event rate among the missing.

        p_true = p_obs x (1 - f) + q x f

    The observed rate applies only to the surviving share; the absent share
    contributes at whatever rate q they had. q = p_obs is the null hypothesis that
    absence is unrelated to drawdowns, and it returns p_obs exactly - which makes
    that row a free check on the arithmetic.
    """
    return p_obs * (1.0 - missing) + q * missing


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_survivorbias_v58() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="offline@localhost")
    ap.add_argument("--cache-dir", default="frames_cache")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--concept", default="Assets")
    ap.add_argument("--survival-bucket", default="large",
                    choices=["large", "population"])
    ap.add_argument("--base-rate", type=float, default=OBS_BASE_RATE)
    ap.add_argument("--cal-mae-pp", type=float, default=OBS_CAL_MAE_PP)
    ap.add_argument("--out", default="survivorbias_v58.json")
    return ap


def run_survivorbias_v58(email=None, cache_dir=None, offline=None, start=None, end=None, concept=None, survival_bucket=None, base_rate=None, cal_mae_pp=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_survivorbias_v58()
        run_survivorbias_v58(email=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"email": email, "cache_dir": cache_dir, "offline": offline, "start": start, "end": end, "concept": concept, "survival_bucket": survival_bucket, "base_rate": base_rate, "cal_mae_pp": cal_mae_pp, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    print("=" * 92)
    print("V58 - SURVIVORSHIP: THE CAVEAT AS ARITHMETIC")
    print("=" * 92)

    try:
        import survivorship_universe_v49 as V49
    except ImportError:
        sys.exit("Run next to survivorship_universe_v49.py.")

    fr = V49.Frames(cfg.email, cfg.cache_dir, offline=cfg.offline)
    d = V49.build_panel(fr, cfg.concept, cfg.start, cfg.end)
    if d is None or d.empty:
        sys.exit("No frames data. Run V49 once online first, or pass --email.")

    p0, p1, cnt = usable_span(d)
    if p0 is None:
        sys.exit("Filing record too sparse to establish a usable span.")
    peak_n = int(cnt.max())
    print(f"\n  usable span: {p0 // 4}Q{p0 % 4 + 1} to {p1 // 4}Q{p1 % 4 + 1} "
          f"(quarters holding at least half the peak of {peak_n:,} filers)")
    print(f"  'still filing' = last filing within 4 quarters of "
          f"{p1 // 4}Q{p1 % 4 + 1}")
    d = d[(d["period"] >= p0) & (d["period"] <= p1)]
    surv = survival_by_year(d, cfg.survival_bucket, p1, peak_n=peak_n)
    surv_pop = survival_by_year(d, "population", p1, peak_n=peak_n)
    if surv.empty:
        sys.exit("Not enough filers per year to compute survival.")
    if surv["survived"].iloc[-1] < 0.9:
        print(f"  WARNING the most recent cohort shows only "
              f"{surv['survived'].iloc[-1]:.0%} survival, which should be near "
              f"100% - the span detection is still wrong, do not trust the rest")

    print(f"\n  filing record: {d['cik'].nunique():,} companies, "
          f"{int(d['y'].min())}-{int(d['y'].max())}")
    print(f"  survival measured within the '{cfg.survival_bucket}' bucket"
          + (" (top 30% by assets)" if cfg.survival_bucket == "large" else ""))
    print(f"  observed survivor-only base rate: {cfg.base_rate:.1%}")

    print("\n" + "=" * 92)
    print("1) HOW MUCH OF THE PAST IS MISSING, BY YEAR")
    print("=" * 92)
    print("  An observation's cohort keeps shrinking after it. The further back the")
    print("  observation, the more of its peers are gone from the price cache.")
    print(f"\n  {'year':>6}{'filers':>9}{'still filing':>14}{'missing':>10}"
          f"{'population missing':>21}")
    pop = surv_pop.set_index("year")["missing"].to_dict()
    for _, r in surv.iterrows():
        pm = pop.get(int(r["year"]))
        print(f"  {int(r['year']):>6}{int(r['n']):>9,}{r['survived']:>13.0%}"
              f"{r['missing']:>10.0%}"
              f"{(f'{pm:.0%}' if pm is not None else '-'):>21}")
    print("\n  The last column is why the size correction matters: using the whole")
    print("  population would overstate the bias for a large-cap universe.")

    print("\n" + "=" * 92)
    print("2) IMPLIED TRUE BASE RATE")
    print("=" * 92)
    print("  Rows are the assumed 30% drawdown rate among absent companies, which")
    print("  this data cannot determine. Columns are the observation's year.")
    print(f"  q = {cfg.base_rate:.1%} is the null that absence is unrelated to")
    print("  drawdowns, and it must return the observed rate unchanged.")
    years = [y for y in surv["year"] if y % 3 == 0 or y == surv["year"].max()]
    if len(years) < 3:
        years = list(surv["year"])
    miss = surv.set_index("year")["missing"].to_dict()
    print(f"\n  {'q':>7}" + "".join(f"{y:>9}" for y in years))
    table = {}
    for q in sorted(set(Q_GRID + [round(cfg.base_rate, 3)])):
        cells, row = [], {}
        for y in years:
            v = implied_rate(cfg.base_rate, miss[y], q)
            row[y] = v
            cells.append(f"{v:>8.1%}")
        tag = "  <- null" if abs(q - cfg.base_rate) < 1e-9 else ""
        table[q] = row
        print(f"  {q:>7.0%}" + "".join(cells) + tag)

    print("\n" + "=" * 92)
    print("3) WHAT IT WOULD TAKE TO MATTER")
    print("=" * 92)
    print(f"  The model's measured out-of-sample calibration error is "
          f"{cfg.cal_mae_pp:.2f}pp. Survivorship")
    print("  only matters if it shifts the true rate by more than that. Below: the")
    print("  absent-company event rate at which the shift exceeds the model's own")
    print("  error, per year.")
    print(f"\n  {'year':>6}{'missing':>10}{'q needed':>11}{'plausible?':>14}")
    need = {}
    for _, r in surv.iterrows():
        f = r["missing"]
        if f <= 1e-6:
            continue
        # p_obs*(1-f) + q*f = p_obs + mae  ->  q = p_obs + mae/f
        q_need = cfg.base_rate + (cfg.cal_mae_pp / 100.0) / f
        need[int(r["year"])] = q_need
        verdict = ("impossible" if q_need > 1.0 else
                   "very likely" if q_need < 0.35 else
                   "likely" if q_need < 0.6 else "possible")
        print(f"  {int(r['year']):>6}{f:>10.0%}"
              f"{(f'{q_need:.0%}' if q_need <= 1 else '>100%'):>11}{verdict:>14}")
    print("\n  A low 'q needed' means very little failure among absent companies")
    print("  would be enough to push the true rate outside the model's error bar.")
    print("  Delisted companies are not a random sample of the market, so treat")
    print("  anything under about 35% as reached.")

    print("\n" + "=" * 92)
    print("4) WHAT THIS DOES TO THE DEPLOYED MODEL")
    print("=" * 92)
    q_mid = 0.5
    y0, y1 = int(surv["year"].min()), int(surv["year"].max())
    for y in (y0, y1):
        f = miss[y]
        p_true = implied_rate(cfg.base_rate, f, q_mid)
        ratio = p_true / cfg.base_rate if cfg.base_rate > 0 else np.nan
        print(f"  {y}: missing {f:.0%}, at q={q_mid:.0%} the true rate is "
              f"{p_true:.1%}, {ratio:.1f}x the observed")
        print(f"        a risk budget sized on the observed probability is then "
              f"about {ratio:.1f}x too large")
    print("\n  The direction is the point. For an alpha model survivorship inflates")
    print("  returns, which is embarrassing. For a RISK model it understates danger")
    print("  and enlarges positions, which is worse. Every probability this project")
    print("  reports should be read as a floor.")

    print("\n" + "=" * 92)
    print("5) THE OPEN QUESTION THIS BEARS ON")
    print("=" * 92)
    early = surv.iloc[0]
    late = surv.iloc[-1]
    print(f"  missing fraction: {early['missing']:.0%} in {int(early['year'])}, "
            f"{late['missing']:.0%} in {int(late['year'])}")
    print("  V48 found its strongest information coefficient in the EARLIEST")
    print("  windows (+0.0547 in 2011-14) decaying to zero by 2023-26, and V50")
    print("  could not tell survivorship from alpha decay as the cause. The bias")
    print("  is mechanically largest exactly where the signal looked strongest.")
    print("  That is not proof - genuine decay would look the same - but it is a")
    print("  quantitative reason to prefer the survivorship reading, and it is")
    print("  the first handle this project has had on the question.")
    print("\n  The test that would settle it: re-run V48 restricted to the years")
    print("  where the missing fraction is small. If the early-period signal")
    print("  survives on a nearly complete cohort it is real; if it disappears it")
    print("  was survivorship. That needs delisted price history, which is why")
    print("  the question stays open.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg),
                   "survival": surv.to_dict("records"),
                   "survival_population": surv_pop.to_dict("records"),
                   "implied": {str(k): v for k, v in table.items()},
                   "q_needed": need}, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_survivorbias_v58(),
# and each value shown is that argument's own default. This one has no QUICK mode - a
# full run takes a while.
#
# Passing any command-line flag still works and takes over, so the old CLI is
# not lost: it just is not the default way in any more.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # Put a REAL address here: SEC fair-access wants a contact in the
    # User-Agent, and they throttle or block requests without one.
    RUN_EMAIL           = 'offline@localhost'
    RUN_CACHE_DIR       = 'frames_cache'
    # False downloads ~21 years of SEC frames on the first run, which is slow
    # and needs network. True uses only what is already in RUN_CACHE_DIR.
    RUN_OFFLINE         = False
    RUN_START           = 2005
    RUN_END             = 2026
    RUN_CONCEPT         = 'Assets'
    RUN_SURVIVAL_BUCKET = 'large'
    RUN_BASE_RATE       = 0.107
    RUN_CAL_MAE_PP      = 2.15
    RUN_OUT             = 'survivorbias_v58.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_survivorbias_v58.py --email ...
    else:
        run_survivorbias_v58(
            email=RUN_EMAIL,
            cache_dir=RUN_CACHE_DIR,
            offline=RUN_OFFLINE,
            start=RUN_START,
            end=RUN_END,
            concept=RUN_CONCEPT,
            survival_bucket=RUN_SURVIVAL_BUCKET,
            base_rate=RUN_BASE_RATE,
            cal_mae_pp=RUN_CAL_MAE_PP,
            out=RUN_OUT,
        )
