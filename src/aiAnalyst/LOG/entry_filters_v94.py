"""
V94 - MAKING V93'S TREND FILTER TESTABLE: TREND RULES THAT SURVIVE A DIP

WHY V93 COULD NOT CONFIRM ITS FILTER
------------------------------------
V93's uptrend rule ("close above the 200-day EMA") pointed the right way on
the unseen stocks, but it removed almost every signal. The model fires on the
deepest dips, and a deep dip usually drags the price below its own 200-day
line, so the rule vetoes the very days the model looks for. On the unseen
stocks in 2022-2026 it kept 25 of 1,017 signals: far too few to confirm
anything. On the core stocks in 2022-2026 it lowered the average R. Verdict:
not confirmed.

WHAT V94 CHANGES
----------------
1. Trend rules that do not depend on today's price, so a dip inside a longer
   uptrend still counts as an uptrend. All are textbook definitions; none is
   tuned:
     T1 golden-cross regime   50-day EMA above the 200-day EMA
     T2 rising long trend     200-day EMA higher than 20 trading days ago
     T3 12-month momentum     close 21 trading days ago above the close 252
                              trading days ago: the past year's move,
                              skipping the last month, where the dip sits
     F1 V93's rule            close above the 200-day EMA (for reference)
   V93's market filter is dropped. It did not help on the unseen stocks
   (-0.026 R vs no filter) and hurt on the core in 2022-2026 (-0.106 R,
   interval below zero).
2. A filter must keep enough signals to be testable: at least 25% of them on
   the choice data, and at least 200 on the confirmation data, counted
   before any outcome there is read. Otherwise the answer is INCONCLUSIVE,
   not a pass.
3. Fresh confirmation data. Every filter result so far came from 2017-2026.
   The unseen stocks' 2008-2016 signals have never been used for any filter
   question. So the claim is tested there: on stocks the model was not
   trained on, and in years the filter was not chosen on.

WHAT IS TESTED (fixed before any result is seen)
------------------------------------------------
A filter only removes ENTER signals; the model and its 1% cut do not change.

  CHOOSE   unseen liquid stocks, 2017-2026. These were already seen in V93,
           so they are used only to choose. Among F1 and T1-T3, keep the
           filters that pass at least 25% of the signals. Choose the one with
           the highest average R per kept trade, if that beats no filter.
           Otherwise no filter is adopted, and the confirmation data are not
           read, so they stay fresh for a later test.
  CONFIRM  unseen liquid stocks, 2008-2016. Power gate: the chosen filter
           must keep at least 200 signals there, else INCONCLUSIVE (outcomes
           not read). CONFIRMED only if all four hold:
             1. kept signals earn more than the signals it skips: average R
                kept minus average R skipped, 90% lower bound > 0
             2. kept signals hit the target more often than an average entry
                in the same month (all stocks): 90% lower bound > 0
                (V93's criterion 1, unchanged)
             3. the timing edge survives: kept signals hit the target more
                often than all days of their own stock-month: 90% lower
                bound > 0 (V93's criterion 3, unchanged)
             4. not only a crash effect: criterion 1's estimate is also
                above 0 in 2010-2016, without the 2008-09 crisis
           V93's criterion 2 (average R at least no filter's) follows from
           criterion 1, which is stricter (an interval, not a point). So
           V94 is at least as strict as V93.

Signals with less than one year (252 trading days) of price history before
the signal day are left out of every row, F0 included: the trend rules need
a year of prices.

Signals come from the walk-forward copies of V88 (each year scored by a model
trained only on earlier years of the core stocks). They are read from V86's
caches; nothing is retrained.

READING THE TABLES
------------------
  Kept / Skipped            ENTER signals the filter lets through / vetoes
  Target / Stop / Expired   share of trades ending each way (within 60 days)
  Avg R                     average result per trade, in units of the stop
                            distance (+1.67 R = target, -1 R = stop)
  Kept - skipped, R         how much more a kept trade earns than a skipped
                            one: the filter's own value
  vs same stock-month       kept signal minus the average of ALL days of the
                            same stock and month: the timing claim
  vs avg entry              kept signal minus the average of all stocks'
                            days in the same month: "better than buying
                            anything that month"

A note on the core stocks (shown as secondary only): about 45 of them were
picked in 2026 because they had done well, so their past falls tended to
recover. That can make dips in falling stocks look better on the core than
they are, which works against any trend filter there.
"""

import os
import json
import time
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import entry_filters_v93 as V93

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = "thesis_tables_v94"
CHOOSE_YEARS = list(range(2017, 2027))
CONFIRM_YEARS = list(range(2008, 2017))
CALM_YEARS = list(range(2010, 2017))          # criterion 4: without 2008-09
MIN_KEPT_SHARE = 0.25
MIN_KEPT_CONFIRM = 200
MIN_HISTORY = 252
SLOPE_BARS = 20
MOM_SKIP, MOM_LOOK = 21, 252
BEAR, BULL = V82.BREADTH_CUTS

F0 = "F0 none (V88 as is)"
FILTERS = {                                   # name -> trend column
    F0: None,
    "F1 close > 200-day EMA (V93)": "f1",
    "T1 50-day EMA > 200-day EMA": "t1",
    "T2 200-day EMA rising (20 days)": "t2",
    "T3 12-month momentum > 0": "t3",
}
TREND_COLS = ("f1", "t1", "t2", "t3")
OUTCOME_COLS = ("Signals", "Target %", "Stop %", "Expired %", "Avg R")

PREREG = """PRE-REGISTRATION - V94 trend filters that survive a dip
written {now}, before any V94 result was computed{replaces}.

FILTERS   applied to the walk-forward V88 ENTER signals; the model and its 1%
          cut are unchanged. Signals with less than {hist} trading days of
          price history before the signal day are left out of every row.
            F1 close > 200-day EMA (V93's rule)
            T1 50-day EMA > 200-day EMA
            T2 200-day EMA above its value {slope} trading days earlier
            T3 close {skip} trading days ago > close {look} trading days ago
CHOOSE    unseen liquid stocks, {s0}-{s1} (already seen in V93; used only to
          choose). Among filters keeping >= {share:.0%} of the signals, the
          highest average R per kept trade, only if it beats no filter.
          Otherwise no filter is adopted and the confirmation outcomes are
          not read.
CONFIRM   unseen liquid stocks, {c0}-{c1} (never used for any filter
          question). Power gate: >= {gate} kept signals, else INCONCLUSIVE
          and the outcomes are not read. CONFIRMED only if all hold:
            1. average R kept minus average R skipped, 90% lower bound > 0
            2. target hit rate minus the same-month all-stock average, 90%
               lower bound > 0
            3. target hit rate minus the same stock-month average, 90% lower
               bound > 0
            4. criterion 1's estimate > 0 in {q0}-{q1} (without 2008-09)
METHOD    calendar-month bootstrap, {nb} draws; kept and skipped signals
          paired by month.
"""


# =============================================================================
# DATA
# =============================================================================
def add_trend(U, mask, cache):
    """
    Trend flags on the rows in `mask`, from each stock's own prices up to and
    including the signal day's close (the entry price) - no look-ahead.
    Also `bars`: how many trading days of history came before the signal.
    Returns the number of signal rows whose date was not found in the prices.
    """
    out = {k: np.full(len(U), np.nan) for k in TREND_COLS + ("bars",)}
    sub = U.loc[np.asarray(mask, bool), ["ticker", "date"]]
    unmatched = 0
    for t, g in sub.groupby("ticker", sort=False):
        df = E64.P2.load_prices(t, cache)
        if df is None or "Close" not in df:
            unmatched += len(g)
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")].sort_index()
        v = c.to_numpy()
        c = c[np.isfinite(v) & (v > 0)]
        e50, e200 = E64._ema(c, 50), E64._ema(c, 200)
        flags = {"f1": c > e200,
                 "t1": e50 > e200,
                 "t2": e200 > e200.shift(SLOPE_BARS),
                 "t3": c.shift(MOM_SKIP) > c.shift(MOM_LOOK)}
        pos = c.index.get_indexer(pd.DatetimeIndex(g["date"]).normalize())
        ok = pos >= 0
        unmatched += int((~ok).sum())
        rows, p = g.index.to_numpy()[ok], pos[ok]
        for k, s in flags.items():
            out[k][rows] = s.to_numpy(float)[p]
        out["bars"][rows] = p
    for k, v in out.items():
        U[k] = v
    return unmatched


def masks(U, fires):
    """valid = ENTER signals with a year of history; one kept-mask per filter."""
    bars = np.nan_to_num(U["bars"].to_numpy(float), nan=-1.0)
    valid = np.asarray(fires, bool) & (bars >= MIN_HISTORY)
    for k in TREND_COLS:
        valid &= np.isfinite(U[k].to_numpy(float))
    M = {}
    for name, col in FILTERS.items():
        M[name] = valid.copy() if col is None \
            else valid & (U[col].to_numpy(float) > 0.5)
    return valid, M


# =============================================================================
# TABLES
# =============================================================================
def pct(v, d=1):
    return f"{v:.{d}f}" if np.isfinite(v) else "-"


def sgn(v, d=3):
    return f"{v:+.{d}f}" if np.isfinite(v) else "-"


def with_ci(d, lo, hi, d_fmt=3):
    return f"{sgn(d, d_fmt)} {V93.ci(lo, hi)}" if np.isfinite(d) else "-"


def table(st, valid, M, years):
    """One row per filter: outcomes of the kept signals, of the skipped ones,
    and the paired kept-minus-skipped differences."""
    n0 = int((valid & np.isin(st.ev.year, years)).sum())
    R, tg = st.ind["R"], st.ind["tgt"]
    rows = []
    for name, kept in M.items():
        r = st.row(kept, years)
        rec = {"Filter": name, **r}
        rec["Kept %"] = r["Signals"] / n0 * 100 if n0 else np.nan
        if name != F0:
            skip = valid & ~kept
            rs = st.row(skip, years)
            rec["Skipped"] = rs["Signals"]
            rec["Target % skipped"] = rs.get("Target %", np.nan)
            rec["Avg R skipped"] = rs.get("Avg R", np.nan)
            d, lo, hi = st.paired(R, kept, R, skip, years)
            rec.update({"R kept-skipped": d, "R kept-skipped lo": lo,
                        "R kept-skipped hi": hi})
            d, lo, hi = st.paired(tg, kept, tg, skip, years)
            rec.update({"Target kept-skipped pp": d * 100,
                        "Target kept-skipped lo": lo * 100,
                        "Target kept-skipped hi": hi * 100})
        rows.append(rec)
    T = pd.DataFrame(rows)
    for c in ("Target %", "Stop %", "Expired %", "Avg R", "Skipped",
              "Target % skipped", "Avg R skipped", "R kept-skipped",
              "R kept-skipped lo", "R kept-skipped hi",
              "Target vs average entry pp", "Target vs average entry lo",
              "Target vs average entry hi", "Target vs same stock-month pp",
              "Target vs same stock-month lo",
              "Target vs same stock-month hi"):
        if c not in T:
            T[c] = np.nan
    return T


def show(T, title, W=118):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    A = pd.DataFrame({"Filter": T["Filter"]})
    A["Kept"] = T["Signals"].map(lambda v: f"{int(v):,}")
    A["Kept %"] = T["Kept %"].map(lambda v: pct(v, 0))
    for c in ("Target %", "Stop %", "Expired %"):
        A[c] = T[c].map(pct)
    A["Avg R"] = T["Avg R"].map(sgn)
    A["Skipped"] = T["Skipped"].map(
        lambda v: f"{int(v):,}" if np.isfinite(v) else "-")
    A["Target % skipped"] = T["Target % skipped"].map(pct)
    A["Avg R skipped"] = T["Avg R skipped"].map(sgn)
    print(A.to_string(index=False))
    B = pd.DataFrame({"Filter": T["Filter"]})
    B["Kept - skipped, R [90%]"] = [
        with_ci(d, lo, hi) for d, lo, hi in zip(
            T["R kept-skipped"], T["R kept-skipped lo"],
            T["R kept-skipped hi"])]
    B["Target vs avg entry, pp [90%]"] = [
        with_ci(d, lo, hi, 1) for d, lo, hi in zip(
            T["Target vs average entry pp"], T["Target vs average entry lo"],
            T["Target vs average entry hi"])]
    B["Target vs same stock-month, pp [90%]"] = [
        with_ci(d, lo, hi, 1) for d, lo, hi in zip(
            T["Target vs same stock-month pp"],
            T["Target vs same stock-month lo"],
            T["Target vs same stock-month hi"])]
    print()
    print(B.to_string(index=False))


def coverage(unis, plan):
    """Signal counts only - no outcomes - for every universe and period."""
    rows = []
    for name, years in plan:
        U, fires, valid, M, _ = unis[name]
        yr = np.isin(U["year"].to_numpy(), years)
        n_f, n_v = int((fires & yr).sum()), int((valid & yr).sum())
        rec = {"Universe": name, "Years": f"{years[0]}-{years[-1]}",
               "ENTER": f"{n_f:,}", ">= 1 yr history": f"{n_v:,}"}
        for fname, m in M.items():
            if fname == F0:
                continue
            k = int((m & yr).sum())
            rec[fname.split(" ")[0] + " kept"] = (
                f"{k:,} ({k / n_v * 100:.0f}%)" if n_v else "-")
        rows.append(rec)
    return pd.DataFrame(rows)


def presentable(st, M, valid, chosen, years):
    """Outcome shares for V88 as is, for the kept signals, for their two
    comparison groups, and for the signals the filter skips."""
    rows = []
    for key, label in ((F0, "ENTER, no filter (V88 as is)"),
                       (chosen, f"ENTER + {chosen.split(' ', 1)[1]}")):
        r = st.row(M[key], years)
        rows.append({"Entry": label,
                     **{k: r.get(k, np.nan) for k in OUTCOME_COLS}})
        for k, v in st.baselines(M[key], years).items():
            rows.append({"Entry": f"    vs {k}", **v})
    r = st.row(valid & ~M[chosen], years)
    rows.append({"Entry": "ENTER signals the filter skips",
                 **{k: r.get(k, np.nan) for k in OUTCOME_COLS}})
    return pd.DataFrame(rows)


def show_outcomes(P, title, W=118):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    D = P.copy()
    D["Signals"] = P["Signals"].map(
        lambda v: f"{int(v):,}" if np.isfinite(v) else "-")
    for c in ("Target %", "Stop %", "Expired %"):
        D[c] = P[c].map(pct)
    D["Avg R"] = P["Avg R"].map(sgn)
    print(D.to_string(index=False))


def by_group(st, valid, kept, years, groups, with_boot=False):
    """Descriptive: kept vs skipped signals inside each group (year, regime)."""
    yr = np.isin(st.ev.year, years)
    R, tg = st.ind["R"], st.ind["tgt"]
    rows = []
    for name, g in groups:
        k, s = kept & g & yr, valid & ~kept & g & yr
        nk, ns = int(k.sum()), int(s.sum())
        rec = {"Group": name, "ENTER": nk + ns, "Kept": nk,
               "Kept %": nk / (nk + ns) * 100 if nk + ns else np.nan,
               "Target % kept": tg[k].mean() * 100 if nk else np.nan,
               "Target % skipped": tg[s].mean() * 100 if ns else np.nan,
               "Avg R kept": R[k].mean() if nk else np.nan,
               "Avg R skipped": R[s].mean() if ns else np.nan}
        if with_boot:
            d, lo, hi = st.paired(R, kept & g, R, valid & ~kept & g, years) \
                if nk and ns else (np.nan, np.nan, np.nan)
            rec.update({"R kept-skipped": d, "lo": lo, "hi": hi})
        rows.append(rec)
    return pd.DataFrame(rows)


def show_groups(T, title, W=118):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    D = pd.DataFrame({"Group": T["Group"]})
    D["ENTER"] = T["ENTER"].map(lambda v: f"{int(v):,}")
    D["Kept"] = T["Kept"].map(lambda v: f"{int(v):,}")
    D["Kept %"] = T["Kept %"].map(lambda v: pct(v, 0))
    D["Target % kept"] = T["Target % kept"].map(pct)
    D["Target % skipped"] = T["Target % skipped"].map(pct)
    D["Avg R kept"] = T["Avg R kept"].map(sgn)
    D["Avg R skipped"] = T["Avg R skipped"].map(sgn)
    if "R kept-skipped" in T:
        D["Kept - skipped, R [90%]"] = [
            with_ci(d, lo, hi) for d, lo, hi in zip(
                T["R kept-skipped"], T["lo"], T["hi"])]
    print(D.to_string(index=False))


# =============================================================================
# PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, n_boot):
    settings = {"filters": list(FILTERS), "choose": CHOOSE_YEARS,
                "confirm": CONFIRM_YEARS, "calm": CALM_YEARS,
                "share": MIN_KEPT_SHARE, "gate": MIN_KEPT_CONFIRM,
                "hist": MIN_HISTORY, "slope": SLOPE_BARS,
                "mom": [MOM_SKIP, MOM_LOOK], "n_boot": n_boot}
    sig = hashlib.sha256(json.dumps(settings, sort_keys=True)
                         .encode()).hexdigest()[:16]
    os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "preregistration.json")
    tp = os.path.join(out_dir, "preregistration.txt")
    status = "fresh"
    if os.path.exists(mp):
        meta = json.load(open(mp))
        if meta.get("sig") == sig:
            print(f"  pre-registered {meta['written']} - unchanged since")
            return meta["written"], meta.get("status") != \
                "changed-after-result"
        status = ("changed-after-result" if os.path.exists(
            os.path.join(out_dir, "verdict.txt")) else "changed")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = PREREG.format(
        now=now, replaces="" if status == "fresh"
        else " (REPLACES an earlier pre-registration)",
        hist=MIN_HISTORY, slope=SLOPE_BARS, skip=MOM_SKIP, look=MOM_LOOK,
        s0=CHOOSE_YEARS[0], s1=CHOOSE_YEARS[-1], share=MIN_KEPT_SHARE,
        c0=CONFIRM_YEARS[0], c1=CONFIRM_YEARS[-1], gate=MIN_KEPT_CONFIRM,
        q0=CALM_YEARS[0], q1=CALM_YEARS[-1], nb=n_boot)
    if os.path.exists(tp):
        with open(tp) as fh, open(os.path.join(
                out_dir, "preregistration_history.txt"), "a") as h:
            h.write(fh.read() + "\n" + "-" * 80 + "\n")
    open(tp, "w").write(txt)
    json.dump({"sig": sig, "written": now, "status": status}, open(mp, "w"))
    print(f"  wrote {tp} ({now})")
    if status == "changed-after-result":
        print("  *** settings changed AFTER a result was written - this "
              "verdict is exploratory")
    return now, status != "changed-after-result"


# =============================================================================
# RUNNER
# =============================================================================
def choose(T):
    """V94's choice rule, applied to the choice table. Returns (chosen, notes)."""
    r0 = float(T.loc[T["Filter"] == F0, "Avg R"].iloc[0])
    notes, best = [], None
    for _, r in T[T["Filter"] != F0].iterrows():
        if not r["Kept %"] >= MIN_KEPT_SHARE * 100:
            why = f"keeps {r['Kept %']:.0f}% of signals (needs " \
                  f"{MIN_KEPT_SHARE:.0%}) - not eligible"
        elif not r["Avg R"] > r0:
            why = f"avg R {r['Avg R']:+.3f} does not beat no filter " \
                  f"({r0:+.3f}) - not eligible"
        else:
            why = f"eligible: avg R {r['Avg R']:+.3f} vs {r0:+.3f}, keeps " \
                  f"{r['Kept %']:.0f}%"
            if best is None or r["Avg R"] > best[1]:
                best = (r["Filter"], r["Avg R"])
        notes.append(f"    {r['Filter']:<34} {why}")
    return (best[0] if best else None), notes


def run_v94(model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
            out_dir=OUT_DIR, verbose=False):
    t0 = time.time()
    W = 118
    print("=" * W)
    print("V94 - TREND FILTERS THAT SURVIVE A DIP: choose on unseen 2017-2026, "
          "confirm on unseen 2008-2016 (pre-registered)")
    print("=" * W)
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    print("  reading V86's data and walk-forward models (cached) ...")
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    print("  market breadth from the core stocks (regime table only) ...")
    b, _ = V82.full_breadth(core_cache, V86.cache_names(core_cache))
    CORE = f"core {n_core}"
    UNSEEN = "unseen, liquid"
    unis = {}
    for name, frame, mask, tag, cache in (
            (UNSEEN, fresh, fresh["liquid"].to_numpy(bool), "fresh",
             big_cache),
            (CORE, core, np.ones(len(core), bool), "core", core_cache)):
        U, w = V87.universe(frame, mask, wf, tag)
        fires = V85.model_fire(U, *w["combo"])
        U["breadth"] = pd.DatetimeIndex(U["date"]).normalize().map(b) \
            .to_numpy(float)
        print(f"  {name}: trend measures on {int(fires.sum()):,} ENTER "
              f"rows ...")
        miss = add_trend(U, fires, cache)
        valid, M = masks(U, fires)
        if miss:
            print(f"      {miss:,} signal dates not found in the price files "
                  f"(left out)")
        unis[name] = (U, fires, valid, M, V93.Stats(U, n_boot))

    print("\n  PRE-REGISTRATION")
    when, prereg_ok = preregister(out_dir, n_boot)

    # ---- [0] coverage: counts only ----------------------------------------
    C = coverage(unis, ((UNSEEN, CHOOSE_YEARS), (UNSEEN, CONFIRM_YEARS),
                        (CORE, CHOOSE_YEARS), (CORE, CONFIRM_YEARS)))
    print("\n" + "-" * W)
    print("  [0] HOW MANY SIGNALS EACH FILTER KEEPS - counts only, no "
          "outcomes read")
    print("-" * W)
    print(C.to_string(index=False))
    C.to_csv(f"{out_dir}/coverage.csv", index=False)

    # ---- [1] choose on unseen 2017-2026 -------------------------------------
    U, fires, valid, M, st = unis[UNSEEN]
    T_ch = table(st, valid, M, CHOOSE_YEARS)
    show(T_ch, f"[1] CHOOSE - {UNSEEN}, {CHOOSE_YEARS[0]}-"
               f"{CHOOSE_YEARS[-1]} (seen in V93; used only to choose)")
    T_ch.to_csv(f"{out_dir}/choose_unseen_{CHOOSE_YEARS[0]}_"
                f"{CHOOSE_YEARS[-1]}.csv", index=False)
    chosen, notes = choose(T_ch)
    print("\n  choice rule:")
    print("\n".join(notes))
    print(f"\n  chosen: {chosen or 'none'}")

    # ---- [2] confirm on unseen 2008-2016 ------------------------------------
    lines, ran = [], False
    tag = f"{CONFIRM_YEARS[0]}-{CONFIRM_YEARS[-1]}"
    if chosen is None:
        lines.append("  VERDICT: NOT ADOPTED - no trend filter that keeps at "
                     f"least {MIN_KEPT_SHARE:.0%} of the signals beat V88 as "
                     "is on the choice data.\n"
                     f"  The {tag} outcomes were not read, so they stay "
                     "fresh for a later test.")
    else:
        yr = np.isin(st.ev.year, CONFIRM_YEARS)
        n_conf = int((M[chosen] & yr).sum())
        n_all = int((valid & yr).sum())
        print(f"\n  power gate: '{chosen}' keeps {n_conf:,} of {n_all:,} "
              f"signals in {tag} (needs {MIN_KEPT_CONFIRM})"
              f" -> {'OK' if n_conf >= MIN_KEPT_CONFIRM else 'TOO FEW'}")
        if n_conf < MIN_KEPT_CONFIRM:
            lines.append(f"  VERDICT: INCONCLUSIVE - '{chosen}' keeps only "
                         f"{n_conf} signals in {tag} (needs "
                         f"{MIN_KEPT_CONFIRM}).\n  The outcomes there were "
                         "not read; judge the filter on live data instead.")
        else:
            ran = True
            T_cf = table(st, valid, M, CONFIRM_YEARS)
            show(T_cf, f"[2] CONFIRM - {UNSEEN}, {tag} (all filters shown; "
                       f"only '{chosen}' is the claim)")
            T_cf.to_csv(f"{out_dir}/confirm_unseen_{CONFIRM_YEARS[0]}_"
                        f"{CONFIRM_YEARS[-1]}.csv", index=False)
            r = T_cf[T_cf["Filter"] == chosen].iloc[0]
            kept, skip = M[chosen], valid & ~M[chosen]
            d4 = st.paired(st.ind["R"], kept, st.ind["R"], skip,
                           CALM_YEARS)[0]
            c1 = r["R kept-skipped lo"] > 0
            c2 = r["Target vs average entry lo"] > 0
            c3 = r["Target vs same stock-month lo"] > 0
            c4 = bool(np.isfinite(d4) and d4 > 0)
            ok = c1 and c2 and c3 and c4
            pf = lambda c: "PASS" if c else "FAIL"
            lines += [
                f"  chosen on unseen {CHOOSE_YEARS[0]}-{CHOOSE_YEARS[-1]}: "
                f"{chosen}; confirmed on unseen {tag} ({n_conf:,} kept, "
                f"{int(r['Skipped']):,} skipped)",
                f"  1. avg R kept minus skipped: {r['R kept-skipped']:+.3f} "
                f"{V93.ci(r['R kept-skipped lo'], r['R kept-skipped hi'])}"
                f"  -> {pf(c1)}",
                f"  2. target vs an average entry in the same month: "
                f"{r['Target vs average entry pp']:+.1f} pp "
                f"{V93.ci(r['Target vs average entry lo'], r['Target vs average entry hi'])}"
                f"  -> {pf(c2)}",
                f"  3. target vs all days of the same stock-month: "
                f"{r['Target vs same stock-month pp']:+.1f} pp "
                f"{V93.ci(r['Target vs same stock-month lo'], r['Target vs same stock-month hi'])}"
                f"  -> {pf(c3)}",
                f"  4. criterion 1 without 2008-09 ({CALM_YEARS[0]}-"
                f"{CALM_YEARS[-1]}): {d4:+.3f}  -> {pf(c4)}"]
            if ok:
                verdict = (f"CONFIRMED - adopt '{chosen}' in the recommender:"
                           " ENTER only when V88 fires AND the stock passes "
                           "the trend rule; otherwise WAIT (downtrend).")
            elif c1 and c3 and c4 and not c2:
                verdict = ("NOT CONFIRMED - the filter improves the trades "
                           "(criterion 1), but its hit rate is not clearly "
                           "above an average entry (criterion 2). Keep V88 "
                           "unfiltered.")
            else:
                verdict = "NOT CONFIRMED - keep V88's signals unfiltered."
            lines.append(f"\n  VERDICT: {verdict}")

    if ran:
        # ---- [3] the table to present -------------------------------------
        P = presentable(st, M, valid, chosen, CONFIRM_YEARS)
        show_outcomes(P, f"[3] THE TABLE TO PRESENT - {UNSEEN}, {tag}. "
                         "Each 'vs' row uses the same stock-months / months "
                         "as the ENTER row above it.")
        P.to_csv(f"{out_dir}/presentable_unseen_{CONFIRM_YEARS[0]}_"
                 f"{CONFIRM_YEARS[-1]}.csv", index=False)

        # ---- [4] where the filter helps: regimes and years -----------------
        bb = U["breadth"].to_numpy(float)
        G = by_group(st, valid, M[chosen], CONFIRM_YEARS,
                     ((f"bull (breadth > {BULL:.2f})", bb > BULL),
                      ("neutral", (bb >= BEAR) & (bb <= BULL)),
                      (f"bear (breadth < {BEAR:.2f})", bb < BEAR)),
                     with_boot=True)
        show_groups(G, f"[4a] BY MARKET REGIME - {UNSEEN}, {tag}, "
                       f"'{chosen}' (descriptive)")
        G.to_csv(f"{out_dir}/regimes_unseen.csv", index=False)
        yrs = st.ev.year
        Y = by_group(st, valid, M[chosen], CONFIRM_YEARS,
                     [(str(y), yrs == y) for y in CONFIRM_YEARS])
        show_groups(Y, f"[4b] BY YEAR - {UNSEEN}, '{chosen}' (descriptive)")
        Y.to_csv(f"{out_dir}/per_year_unseen.csv", index=False)

        # ---- [5] secondary: the core --------------------------------------
        Uc, fc, vc, Mc, stc = unis[CORE]
        for yrs_, key in ((CONFIRM_YEARS, "confirm"), (CHOOSE_YEARS,
                                                        "choose")):
            T = table(stc, vc, Mc, yrs_)
            show(T, f"[5] secondary - {CORE}, {yrs_[0]}-{yrs_[-1]} "
                    "(hand-picked stocks - see the note at the top)")
            T.to_csv(f"{out_dir}/secondary_core_{yrs_[0]}_{yrs_[-1]}.csv",
                     index=False)

    if not prereg_ok:
        lines.append("  (settings changed after an earlier result - "
                     "exploratory)")
    print("\n" + "=" * W + "\n  THE PRE-REGISTERED VERDICT\n" + "=" * W)
    print("\n".join(lines))
    print(f"\n  pre-registered {when}; tables in {out_dir}/ | "
          f"{(time.time() - t0) / 60:.1f} min")
    open(os.path.join(out_dir, "verdict.txt"), "w").write(
        "\n".join(lines) + "\n")


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_BIG_CACHE = "price_cache_v68"
    RUN_CHUNK     = 250                 # same as V86 (cache key)
    RUN_N_BOOT    = 4000
    RUN_OUT_DIR   = "thesis_tables_v94"
    # -------------------------------------------------------------------------

    run_v94(big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK, n_boot=RUN_N_BOOT,
            out_dir=RUN_OUT_DIR)
