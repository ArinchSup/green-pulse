"""
V95 - THE RANDOM BACKTEST, MADE STEADY

WHY V91'S HIT RATE JUMPS FROM RUN TO RUN
----------------------------------------
V91 draws N random (stock, date) picks and trades the ones where V88 says
ENTER. The model fires on about 1% of days, so N = 3,000 picks gives only
about 30 trades. With a true hit rate near 42%, chance alone puts one run
anywhere between about 27% and 57%, and about 1 run in 3 lands below 40%.
No model change can stop that - only more trades per run can.

WHAT V95 DOES
-------------
  [1] V91's draw, repeated: the same 3,000-pick backtest run many times
      (and once more with 30,000 picks per run), so you see the average and
      the spread instead of one lucky or unlucky run.
  [2] Every signal, no random draw: all walk-forward ENTER signals in the
      period - the number the repeated runs are scattered around - with
      90% intervals (calendar-month bootstrap).
  [3] By year: how much the market's year moves the hit rate.
  Core 337 and unseen liquid stocks side by side, because the core stocks
  were hand-picked and flatter every result.

HOW THE HIT RATE IS SHOWN
-------------------------
  Hit %            share of trades that reach the target within 60 days
  Resolved hit %   target / (target + stop): leaves out trades that ended
                   neither way. The break-even rate (37.5% for a target
                   1.67x the stop) is defined for exactly these trades, so
                   this is the fair number to put next to it.
  Avg R            average result per trade in units of the stop distance

Outcomes are the model's own triple-barrier labels (V91 checked that they
match a trade simulated from the prices). Costs (0.10% round trip in V91)
change returns, not which barrier is hit, so hit rates do not depend on them.

Signals come from the walk-forward copies of V88 (each year scored by a model
trained only on earlier years), read from V86's caches - nothing retrained.
Nothing here is a new claim or a new filter: it measures the same frozen
model more carefully.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_model_v81 as M81
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import entry_filters_v93 as V93

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = "thesis_tables_v95"
GOAL = 40.0                                   # the hit rate you asked about
PERIODS = {"2008-2016": list(range(2008, 2017)),
           "2017-2026": list(range(2017, 2027)),
           "2008-2026": list(range(2008, 2027))}


# =============================================================================
# DATA
# =============================================================================
def load_universes(model_in81, core_cache, big_cache, n_seeds, chunk,
                   verbose):
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    out = {}
    for name, frame, mask, tag in (
            (f"core {n_core}", core, np.ones(len(core), bool), "core"),
            ("unseen, liquid", fresh, fresh["liquid"].to_numpy(bool),
             "fresh")):
        U, w = V87.universe(frame, mask, wf, tag)
        out[name] = (U, V85.model_fire(U, *w["combo"]))
    return out


def break_even(U):
    """1 / (1 + reward:risk), with reward:risk read off the target trades."""
    lab = U["label"].to_numpy(float)
    r = U["r_multiple"].to_numpy(float)
    rr = float(np.nanmedian(r[lab == 1]))
    return 100.0 / (1.0 + rr), rr


# =============================================================================
# V91'S DRAW, FAST AND REPEATABLE
# =============================================================================
class Sampler:
    """
    V91's pick: a random stock, then a random scored day of that stock, no
    pick twice in one run. Plus V91's control: for each ENTER pick, a random
    OTHER scored day of the same stock and month.
    """

    def __init__(self, U, years):
        yr = U["year"].to_numpy()
        rows = np.flatnonzero(np.isin(yr, years))
        t = U["ticker"].astype(str).to_numpy()[rows]
        o = np.argsort(t, kind="mergesort")
        self.rows = rows[o]
        _, counts = np.unique(t[o], return_counts=True)
        # a random stock, then a random day of it = each day weighted by
        # 1 / (number of stocks x that stock's number of days)
        w = np.repeat(1.0 / counts, counts)
        self.w = w / w.sum()
        per =pd.PeriodIndex(pd.DatetimeIndex(U["date"]), freq="M") \
            .astype(str).to_numpy()
        key = U["ticker"].astype(str).to_numpy() + "|" + per
        g, _ = pd.factorize(key)
        self.g = g
        self.go = np.argsort(g, kind="mergesort")
        self.gsz = np.bincount(g)
        self.gstart = np.r_[0, np.cumsum(self.gsz)[:-1]]
        self.rank = np.empty(len(g), np.int64)
        self.rank[self.go] = np.arange(len(g)) - self.gstart[g[self.go]]

    def draw(self, rng, n):
        """n picks, none twice - the same as V91's redraw-on-repeat."""
        n = min(n, len(self.rows))
        return self.rows[rng.choice(len(self.rows), size=n, replace=False,
                                    p=self.w)]

    def other_day(self, rng, rows):
        g = self.g[rows]
        sz = self.gsz[g]
        ok = sz > 1
        rows, g, sz = rows[ok], g[ok], sz[ok]
        off = (rng.random(len(rows)) * (sz - 1)).astype(np.int64)
        off = off + (off >= self.rank[rows])
        return self.go[self.gstart[g] + off]


def resolved(tg, sp, m):
    t, s = tg[m].sum(), sp[m].sum()
    return t / (t + s) * 100 if t + s else np.nan


def repeat_runs(U, fires, years, n_picks, n_runs, seed):
    ind = V93.outcome_indicators(U)
    tg, sp, R = ind["tgt"], ind["stp"], ind["R"]
    S = Sampler(U, years)
    rng = np.random.default_rng([seed, n_picks])
    recs = []
    for k in range(n_runs):
        P = S.draw(rng, n_picks)
        E = P[fires[P]]
        C = S.other_day(rng, E)
        recs.append({
            "run": k + 1, "picks": len(P), "ENTER trades": len(E),
            "ENTER hit %": tg[E].mean() * 100 if len(E) else np.nan,
            "ENTER resolved hit %": resolved(tg, sp, E),
            "ENTER avg R": R[E].mean() if len(E) else np.nan,
            "every pick hit %": tg[P].mean() * 100,
            "other day hit %": tg[C].mean() * 100 if len(C) else np.nan})
    return pd.DataFrame(recs)


def spread_row(name, n_picks, D):
    def band(c):
        v = D[c].dropna().to_numpy(float)
        return (v.mean(), np.percentile(v, 5), np.percentile(v, 95),
                (v >= GOAL).mean() * 100) if v.size else (np.nan,) * 4
    h, r = band("ENTER hit %"), band("ENTER resolved hit %")
    both = D[["ENTER hit %", "other day hit %"]].dropna()
    return {"Universe": name, "Picks per run": n_picks,
            "Runs": len(D), "ENTER trades per run": D["ENTER trades"].mean(),
            "Hit % avg": h[0], "Hit % lo": h[1], "Hit % hi": h[2],
            "Runs >= goal %": h[3],
            "Resolved % avg": r[0], "Resolved % lo": r[1],
            "Resolved % hi": r[2], "Resolved runs >= goal %": r[3],
            "Every pick hit %": D["every pick hit %"].mean(),
            "Other day hit %": D["other day hit %"].mean(),
            "Runs ENTER beat other day %":
                (both["ENTER hit %"] > both["other day hit %"]).mean() * 100
                if len(both) else np.nan}


# =============================================================================
# EVERY SIGNAL, NO DRAW
# =============================================================================
def every_signal(name, U, fires, st, years, label):
    ind = st.ind
    yr = np.isin(st.ev.year, years)
    r = st.row(fires, years)
    b = st.baselines(fires, years)
    sm = b["all days of the same stock & month"]
    ae = b["average entry, same month (all stocks)"]
    return {"Universe": name, "Years": label, "Signals": r["Signals"],
            "Hit %": r.get("Target %", np.nan),
            "Stop %": r.get("Stop %", np.nan),
            "Expired %": r.get("Expired %", np.nan),
            "Resolved hit %": resolved(ind["tgt"], ind["stp"], fires & yr),
            "Avg R": r.get("Avg R", np.nan),
            "Same stock-month hit %": sm["Target %"],
            "Average entry hit %": ae["Target %"],
            "Any day hit %": ind["tgt"][yr].mean() * 100,
            "vs same stock-month pp": r.get("Target vs same stock-month pp"),
            "vs same stock-month lo": r.get("Target vs same stock-month lo"),
            "vs same stock-month hi": r.get("Target vs same stock-month hi"),
            "vs average entry pp": r.get("Target vs average entry pp"),
            "vs average entry lo": r.get("Target vs average entry lo"),
            "vs average entry hi": r.get("Target vs average entry hi")}


def by_year(name, U, fires, st, years):
    ind = st.ind
    tg, sp, R = ind["tgt"], ind["stp"], ind["R"]
    yr = st.ev.year
    rows = []
    for y in years:
        m = fires & (yr == y)
        n = int(m.sum())
        rows.append({"Universe": name, "Year": y, "Signals": n,
                     "Hit %": tg[m].mean() * 100 if n else np.nan,
                     "Resolved hit %": resolved(tg, sp, m),
                     "Avg R": R[m].mean() if n else np.nan,
                     "Same stock-month hit %":
                         st.pool["tgt"][m].mean() * 100 if n else np.nan,
                     "Any day hit %": tg[yr == y].mean() * 100})
    return pd.DataFrame(rows)


# =============================================================================
# PRINTING
# =============================================================================
def f1(v):
    return f"{v:.1f}" if v is not None and np.isfinite(v) else "-"


def fci(d, lo, hi):
    if d is None or not np.isfinite(d):
        return "-"
    return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def banner(title, W):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)


def show_spread(T, W):
    banner(f"[1] THE RANDOM BACKTEST, REPEATED - ENTER trades' hit rate "
           f"across runs (goal: {GOAL:.0f}%+)", W)
    A = pd.DataFrame({
        "Universe": T["Universe"],
        "Picks/run": T["Picks per run"].map(lambda v: f"{v:,}"),
        "Trades/run": T["ENTER trades per run"].map(lambda v: f"{v:.0f}"),
        "Hit % (avg)": T["Hit % avg"].map(f1),
        "90% of runs": [f"{a:.1f} - {b:.1f}" for a, b in
                        zip(T["Hit % lo"], T["Hit % hi"])],
        f"Runs >= {GOAL:.0f}%": T["Runs >= goal %"].map(lambda v: f"{v:.0f}%"),
        "Resolved % (avg)": T["Resolved % avg"].map(f1),
        "90% of runs ": [f"{a:.1f} - {b:.1f}" for a, b in
                         zip(T["Resolved % lo"], T["Resolved % hi"])],
        f"Runs >= {GOAL:.0f}% ": T["Resolved runs >= goal %"].map(
            lambda v: f"{v:.0f}%")})
    print(A.to_string(index=False))
    print("\n  the same runs, against V91's two comparisons:")
    B = pd.DataFrame({
        "Universe": T["Universe"],
        "Picks/run": T["Picks per run"].map(lambda v: f"{v:,}"),
        "ENTER hit %": T["Hit % avg"].map(f1),
        "same stock & month, other day": T["Other day hit %"].map(f1),
        "every pick": T["Every pick hit %"].map(f1),
        "runs where ENTER beat the other day": T[
            "Runs ENTER beat other day %"].map(lambda v: f"{v:.0f}%")})
    print(B.to_string(index=False))


def show_every(T, W, be):
    banner(f"[2] EVERY SIGNAL, NO RANDOM DRAW - what the runs above are "
           f"scattered around (break-even, resolved trades: {be:.1f}%)", W)
    A = pd.DataFrame({
        "Universe": T["Universe"], "Years": T["Years"],
        "Signals": T["Signals"].map(lambda v: f"{int(v):,}"),
        "Hit %": T["Hit %"].map(f1), "Stop %": T["Stop %"].map(f1),
        "Expired %": T["Expired %"].map(f1),
        "Resolved hit %": T["Resolved hit %"].map(f1),
        "Avg R": T["Avg R"].map(lambda v: f"{v:+.3f}"
                                if np.isfinite(v) else "-")})
    print(A.to_string(index=False))
    print("\n  hit % of the comparison groups, for the same signals "
          "(90% intervals, calendar-month bootstrap):")
    B = pd.DataFrame({
        "Universe": T["Universe"], "Years": T["Years"],
        "ENTER": T["Hit %"].map(f1),
        "same stock-month": T["Same stock-month hit %"].map(f1),
        "average entry": T["Average entry hit %"].map(f1),
        "any day": T["Any day hit %"].map(f1),
        "ENTER vs same stock-month, pp": [
            fci(d, lo, hi) for d, lo, hi in zip(
                T["vs same stock-month pp"], T["vs same stock-month lo"],
                T["vs same stock-month hi"])],
        "ENTER vs average entry, pp": [
            fci(d, lo, hi) for d, lo, hi in zip(
                T["vs average entry pp"], T["vs average entry lo"],
                T["vs average entry hi"])]})
    print(B.to_string(index=False))


def show_years(Y, W):
    banner("[3] BY YEAR - every signal (descriptive)", W)
    wide = None
    for name, g in Y.groupby("Universe", sort=False):
        tag = name.split(",")[0].split(" ")[0]
        part = pd.DataFrame({
            "Year": g["Year"].to_numpy(),
            f"{tag} signals": g["Signals"].to_numpy(),
            f"{tag} hit %": g["Hit %"].map(f1).to_numpy(),
            f"{tag} resolved %": g["Resolved hit %"].map(f1).to_numpy(),
            f"{tag} same stock-month %":
                g["Same stock-month hit %"].map(f1).to_numpy()})
        wide = part if wide is None else wide.merge(part, on="Year")
    print(wide.to_string(index=False))


def takeaways(E, T, Y, be, n_ref):
    lines = []
    full = list(PERIODS)[-1]
    for _, r in E[E["Years"] == full].iterrows():
        lines.append(f"  {r['Universe']}, every signal {full}: hit "
                     f"{r['Hit %']:.1f}%, resolved {r['Resolved hit %']:.1f}% "
                     f"(break-even {be:.1f}%), same stock-month "
                     f"{r['Same stock-month hit %']:.1f}%")
    for _, r in T[T["Picks per run"] == n_ref].iterrows():
        lines.append(f"  {r['Universe']}, {n_ref:,} picks per run: about "
                     f"{r['ENTER trades per run']:.0f} trades; hit % lands "
                     f"between {r['Hit % lo']:.1f} and {r['Hit % hi']:.1f} in "
                     f"90% of runs; {100 - r['Runs >= goal %']:.0f}% of runs "
                     f"fall below {GOAL:.0f}% by chance")
    for name, g in Y.groupby("Universe", sort=False):
        g = g[g["Signals"] >= 30]
        if len(g):
            lo, hi = g.loc[g["Hit %"].idxmin()], g.loc[g["Hit %"].idxmax()]
            k = int((g["Hit %"] >= GOAL).sum())
            lines.append(f"  {name}: {k} of {len(g)} years (30+ signals) at "
                         f"{GOAL:.0f}%+; lowest {int(lo['Year'])} "
                         f"{lo['Hit %']:.1f}%, highest {int(hi['Year'])} "
                         f"{hi['Hit %']:.1f}%")
    return lines


# =============================================================================
# RUNNER
# =============================================================================
def run_v95(n_list=(3000, 30000), n_runs=200, seed=7, years=(2008, 2026),
            n_boot=4000, model_in81="entry_model_v81.joblib",
            core_cache=None, big_cache="price_cache_v68", n_seeds=5,
            chunk=250, out_dir=OUT_DIR, verbose=False):
    t0 = time.time()
    W = 118
    os.makedirs(out_dir, exist_ok=True)
    yrs = list(range(years[0], years[1] + 1))
    print("=" * W)
    print(f"V95 - THE RANDOM BACKTEST, MADE STEADY: {n_runs} runs per row, "
          f"{years[0]}-{years[1]}, core and unseen stocks")
    print("=" * W)
    print("  reading V86's data and walk-forward models (cached) ...")
    unis = load_universes(model_in81, core_cache, big_cache, n_seeds, chunk,
                          verbose)
    first = next(iter(unis.values()))[0]
    be, rr = break_even(first)
    print(f"  target = {rr:.2f} x the stop  ->  break-even hit rate for "
          f"resolved trades {be:.1f}%")

    spread, every, year_rows, examples = [], [], [], {}
    for name, (U, fires) in unis.items():
        pool = np.isin(U["year"].to_numpy(), yrs)
        print(f"  {name}: {U.loc[pool, 'ticker'].nunique():,} stocks, "
              f"{int(pool.sum()):,} scored days, ENTER on "
              f"{fires[pool].mean():.2%} of them")
        for n in n_list:
            D = repeat_runs(U, fires, yrs, n, n_runs, seed)
            D.to_csv(f"{out_dir}/runs_{name.split(',')[0].replace(' ', '')}"
                     f"_{n}.csv", index=False)
            spread.append(spread_row(name, n, D))
            if n == n_list[0]:
                examples[name] = D["ENTER hit %"].head(10).to_numpy()
        st = V93.Stats(U, n_boot)
        for label, py in PERIODS.items():
            py = [y for y in py if y in yrs]
            if py:
                every.append(every_signal(name, U, fires, st, py, label))
        year_rows.append(by_year(name, U, fires, st, yrs))

    T, E = pd.DataFrame(spread), pd.DataFrame(every)
    Y = pd.concat(year_rows, ignore_index=True)

    banner(f"[0] WHY ONE RUN IS NOT ENOUGH - ENTER hit % in the first 10 runs "
           f"of {n_list[0]:,} picks", W)
    for name, v in examples.items():
        print(f"  {name:<16} " + "  ".join(f1(x) for x in v))

    show_spread(T, W)
    show_every(E, W, be)
    show_years(Y, W)
    T.to_csv(f"{out_dir}/repeated_runs.csv", index=False)
    E.to_csv(f"{out_dir}/every_signal.csv", index=False)
    Y.to_csv(f"{out_dir}/by_year.csv", index=False)

    print("\n" + "=" * W + "\n  WHAT THIS MEANS\n" + "=" * W)
    print("\n".join(takeaways(E, T, Y, be, n_list[0])))
    print(f"\n  tables in {out_dir}/ | {(time.time() - t0) / 60:.1f} min")
    return T, E, Y


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_N_LIST    = (3000, 30000)       # picks per run (3,000 = V91's run)
    RUN_N_RUNS    = 200                 # how many times each run is repeated
    RUN_SEED      = 7
    RUN_YEARS     = (2008, 2026)        # walk-forward years to draw from
    RUN_N_BOOT    = 4000
    RUN_BIG_CACHE = "price_cache_v68"   # needed to read V86's caches
    RUN_CHUNK     = 250                 # same as V86 (cache key)
    RUN_OUT_DIR   = "thesis_tables_v95"
    # -------------------------------------------------------------------------

    run_v95(n_list=RUN_N_LIST, n_runs=RUN_N_RUNS, seed=RUN_SEED,
            years=RUN_YEARS, n_boot=RUN_N_BOOT, big_cache=RUN_BIG_CACHE,
            chunk=RUN_CHUNK, out_dir=RUN_OUT_DIR)
