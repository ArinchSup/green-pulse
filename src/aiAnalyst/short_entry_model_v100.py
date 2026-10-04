"""
V100 - A SHORT-HORIZON ENTRY MODEL: V88's recipe on a two-week trade
(pre-registered, tested on stocks it has never seen)

WHY
---
V88 times entries for a 60-day trade. The system's SHORT horizon is about two
weeks (trade_config: 14 calendar days). Short-term reversal - a stock that fell
over the last few days tends to bounce over the next weeks - is one of the
best-documented effects in stock returns, and V88's own gain over V81 came from
its short-horizon features (2-day RSI, 1- and 3-day returns, distance to the
10-day low). So the question is whether the same recipe, retrained on a
two-week trade, picks good entry days on stocks it has never seen.

Free hourly data only goes back about two years, so this model uses daily bars
like V88. That keeps 2008-2026 for testing.

WHAT IS FIXED BEFORE ANY RESULT (nothing here was tuned on short trades)
------------------------------------------------------------------------
  TRADE     enter at the signal day's close. Target = 0.5 x the stock's own
            expected move over the hold (ATR% x sqrt(hold)), kept between
            V88's 8% and 40% scaled by sqrt(hold / 60) - for 10 days, 3.27%
            to 16.33% - so the limits bind for the same share of stocks as
            in V88. Stop = 0.6 x target (reward:risk 1.67, break-even 37.5% of
            trades that end at the target or the stop). Neither hit within the
            hold: exit at that day's close. A day that touches both counts as
            the stop.
  MODEL     V88's recipe unchanged: a classifier on the same ten "low is good"
            features, shallow trees, 5 seeds; fold models retrained every year
            on the core stocks only.
  SIGNAL    ENTER when the score is strictly above the 99th percentile of the
            universe's own scores over the trailing 252 trading days, refit
            monthly - V88's rule.
  OPPONENT  the simple rule a researcher would have used: each of the ten
            features alone, with the same 1% rule. The one with the best lift
            on the CORE stocks, 2008-2016, is chosen in this run, before any
            unseen-stock result is computed.

THE TEST
--------
  PRIMARY   unseen liquid stocks (V86's list and liquidity floor), 2008-2026.
            Power gate (counted before any outcome is read): at least 500
            ENTER signals, else INCONCLUSIVE.
     1. TIMING  ENTER days hit the target more often than all days of the same
                stock and month: 90% lower bound above 0 -> CONFIRMED.
     2. RULE    model minus the opponent, paired by month: lower bound above 0
                -> BEATS; upper bound below 0 -> WORSE; otherwise AHEAD BUT NOT
                PROVEN (estimate above 0) or DOES NOT BEAT.
  REPORTED  what the trades look like (hit, stop, expiry, days held, R as
            labelled and with real fills - a day that OPENS beyond the stop or
            target exits at the open - minus costs), market regimes, years,
            both halves, all unseen stocks and the core - descriptive. Real
            fills matter more here than in V88: the stop is only about one
            day's typical move away, so gaps through it are common.

Same evaluation as V86-V99: calendar-month bootstrap, comparisons paired by
month. The first run builds the short-trade data and fold models (30-50
minutes, cached - an interrupted run resumes); later runs take a few minutes.
"""

import contextlib
import hashlib
import json
import math
import os
import textwrap
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import trade_config as TC
import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import measure_v83 as V83
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import entry_filters_v93 as V93

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = "thesis_tables_v100"
CACHE_DIR = "short_cache_v100"
BUILD = "v100.1"
HOLD = 10                          # trading days, about two weeks
STEP = E64.STEP                    # every 5th trading day, as in V88
MID_BARS = TC.HORIZON_CONFIGS["MID"]["lookahead_bars"]
COST = 0.001                       # round trip, as in the trade simulations
CFG = V86.COMBO_CFG                # V88's recipe
FEATS, MONO = V85.features_of(CFG)
N_SEEDS = 5
MIN_TRAIN_YEARS = 3                # as V81/V88's walk-forward
YEARS = list(range(2008, 2027))
CHOOSE_YEARS = list(range(2008, 2017))
HALVES = {"2008-2016": list(range(2008, 2017)),
          "2017-2026": list(range(2017, 2027))}
MIN_SIGNALS = 500
SEED = 100
BEAR, BULL = V82.BREADTH_CUTS
KEEP = (["ticker", "date", "label", "r_multiple", "log_dollar_vol",
         "stop_pct", "target_pct", "bars_held"] + V86.BASE_FEATS)

LIQ, ALL, MODEL, RANDOM = ("unseen, liquid", "unseen, all", "model",
                           "random 1% (yardstick)")

PREREG = """PRE-REGISTRATION - V100 short-horizon entry model
written {now}{when}.

TRADE      daily bars, entry at the signal day's close. Target = {k} x ATR% x
           sqrt({hold}), kept between {floor:.2%} and {ceil:.2%} (V88's limits x
           sqrt({hold}/{mid})); stop = {sl} x target (break-even 37.5% of resolved
           trades); exit at the close after {hold} trading days if neither is
           hit; a day touching both counts as the stop.
MODEL      V88's recipe unchanged: a classifier on ten features, all "low is
           good":
           {feats};
           shallow trees, {seeds} seed(s); fold models retrained each year on the
           {n_core} core stocks only, with a {emb}-calendar-day gap before each
           test year.
SIGNAL     score strictly above the top {q:.0%} cut of the universe's own scores
           over the trailing {win} trading days, refit monthly.
OPPONENT   each of the ten features alone with the same 1% rule; the one with
           the best lift over the same stock-month on the core stocks,
           {c0}-{c1}, chosen in this run before any unseen-stock result.
PRIMARY    unseen liquid stocks ({n_fresh:,} unseen stocks; V86's liquidity floor),
           {y0}-{y1}. Power gate: >= {gate} ENTER signals, else INCONCLUSIVE.
           1. TIMING: lift over the same stock-month, 90% lower bound > 0
              -> CONFIRMED.
           2. RULE: model minus opponent, paired by month: lower bound > 0 ->
              BEATS; upper bound < 0 -> WORSE; else AHEAD BUT NOT PROVEN
              (estimate > 0) or DOES NOT BEAT.
METHOD     calendar-month bootstrap, {nb} draws, paired by month. Costs
           ({cost:.2%} round trip) and gap fills (a day that opens beyond the
           stop or target exits at the open) are reported in R, not used in
           the decisions.
SECONDARY  halves, all unseen stocks, the core, regimes, years - descriptive.
NAMES      sha256 of the unseen list: {names_sig}
"""


# =============================================================================
# 1. THE SHORT TRADE
# =============================================================================
def geometry(hold=HOLD):
    """V88's barrier rule over a shorter hold: same K and stop ratio, limits
    scaled by sqrt(hold / 60) so they bind for the same share of stocks."""
    s = math.sqrt(hold / MID_BARS)
    return {"key": f"SHORT_D{hold}", "hold": int(hold), "k": TC.VOL_K,
            "sl": TC.SL_TP_RATIO, "floor": round(TC.VOL_MIN_TP_PCT * s, 6),
            "ceil": round(TC.VOL_MAX_TP_PCT * s, 6)}


def register(g):
    """Make the short horizon known to trade_config (and so to V64's label
    builder and V86's walk-forward gap). Adds a key; changes nothing else."""
    if TC.GEOMETRY_MODE != "VOL_SCALED":
        raise SystemExit("V100 assumes trade_config.GEOMETRY_MODE = "
                         "'VOL_SCALED', as V88 was built.")
    atr_fields = {k: TC.HORIZON_CONFIGS["SHORT"][k]
                  for k in ("atr_stop_mult", "atr_target_mult",
                            "max_stop_pct", "min_profit_pct", "min_rr")}
    TC.HORIZON_CONFIGS[g["key"]] = {
        "interval": "1d", "lookahead_bars": g["hold"],
        "eval_days": int(round(g["hold"] * 365.25 / 252)), **atr_fields}


@contextlib.contextmanager
def short_limits(g):
    """trade_config's target limits are module-wide (V88's 8%-40%); use the
    short ones only while short-trade labels are being built."""
    old = (TC.VOL_MIN_TP_PCT, TC.VOL_MAX_TP_PCT)
    TC.VOL_MIN_TP_PCT, TC.VOL_MAX_TP_PCT = g["floor"], g["ceil"]
    try:
        yield
    finally:
        TC.VOL_MIN_TP_PCT, TC.VOL_MAX_TP_PCT = old


# =============================================================================
# 2. DATA
# =============================================================================
def _slim(d, cache, names):
    d = E64.apply_label_mode(d, "barrier")
    d = d[KEEP].copy()
    d["date_n"] = pd.DatetimeIndex(d["date"]).normalize()
    return d.merge(V86._short(cache, names), on=["ticker", "date_n"],
                   how="left")


def build(names, cache, g, tag, chunk, verbose=True):
    """Candidate days with short-trade labels, cached in chunks of names."""
    chunks = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    parts, t0, built = [], time.time(), 0
    for k, ch in enumerate(chunks, 1):
        key = V86._hash([tag, cache, ch, g, STEP, BUILD])
        path = os.path.join(CACHE_DIR, f"{tag}_{key}.pkl")
        if os.path.exists(path):
            part, how = pd.read_pickle(path), "cached"
        else:
            t = time.time()
            with short_limits(g):
                d = E64.build_dataset(ch, cache, g["key"], STEP,
                                      verbose=False)
            part = (_slim(d, cache, ch) if len(d) else pd.DataFrame(
                columns=KEEP + ["date_n"] + V86.SHORT_FEATS))
            part.to_pickle(path)
            built += 1
            how = f"{time.time() - t:4.0f}s"
        parts.append(part)
        if verbose:
            left = len(chunks) - k
            eta = (time.time() - t0) / built * left if built else 0
            print(f"    {tag} {k}/{len(chunks)}  {len(ch)} stocks  "
                  f"{len(part):>9,} rows  {how:>6}"
                  + (f"  ~{eta / 60:.0f} min left" if built and left else ""))
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d["date"] = pd.DatetimeIndex(d["date"])
    d = d.sort_values(["date", "ticker"], kind="mergesort") \
        .reset_index(drop=True)
    d["year"] = d["date"].dt.year
    d["row_id"] = np.arange(len(d))
    d["date_n"] = d["date"].dt.normalize()
    return d


def stock_lists(core_cache, big_cache):
    """V86's lists: the core, and the unseen stocks minus near-copies of core
    stocks (share classes, renames). Reuses V86's cached duplicate check."""
    core_names = V86.cache_names(core_cache)
    core_set = set(core_names)
    cand = [t for t in V86.cache_names(big_cache) if t not in core_set]
    dkey = V86._hash(["dups", core_cache, sorted(core_set), big_cache, cand,
                      V86.DUP_CORR, V86.DUP_MIN_OVERLAP])
    for folder in (V86.CACHE_DIR, CACHE_DIR):
        p = os.path.join(folder, f"dups_{dkey}.pkl")
        if os.path.exists(p):
            dups = pd.read_pickle(p)
            break
    else:
        print("  (V86's duplicate check is not cached here - computing it)")
        dups = V86.find_duplicates(core_cache, sorted(core_set), big_cache,
                                   cand, thr=V86.DUP_CORR)
        dups.to_pickle(os.path.join(CACHE_DIR, f"dups_{dkey}.pkl"))
    drop = set(dups["fresh"])
    return core_names, [t for t in cand if t not in drop], len(drop)


def prepare(core_cache, big_cache, g, n_seeds, chunk, verbose=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    register(g)
    core_names, fresh_names, n_dup = stock_lists(core_cache, big_cache)
    print(f"  core {len(core_names)} stocks | unseen {len(fresh_names):,} "
          f"stocks ({n_dup} near-copies of core stocks left out)")
    print(f"  short-trade candidate days (built once, then cached in "
          f"{CACHE_DIR}/) ...")
    core = build(core_names, core_cache, g, "core", chunk, verbose)
    fresh = build(fresh_names, big_cache, g, "fresh", chunk, verbose)
    floor = V86.liquidity_floor(core, fresh["date"], V86.LIQ_PCT)
    fresh["liquid"] = fresh["log_dollar_vol"].to_numpy(float) >= floor
    fold_years = sorted(core["year"].unique())[MIN_TRAIN_YEARS:]
    sig = V86._hash([len(core), core_names, len(fresh), fresh_names,
                     str(fresh["date"].max()), g, STEP, BUILD])
    print("  walk-forward: V88's recipe retrained each year on the core, "
          "scoring both sets (cached after the first run) ...")
    wf = {"combo": V86.walk_forward_oou(
        CFG, core, fresh, fold_years, n_seeds, g["key"],
        "short model (V88's recipe)", sig, cache_dir=CACHE_DIR,
        verbose=verbose)}
    return core, fresh, wf, core_names, fresh_names


def rule(U, score):
    """The same strict 1% rule as V88, on a plain score."""
    score = np.asarray(score, float)
    return V83.apply_rule(score, V83.rolling_cut(U["date"], score),
                          strict=True)


def rule_name(f, m):
    return f"{'low' if m < 0 else 'high'} {f}"


# =============================================================================
# 3. TABLES
# =============================================================================
def pct(v, d=1):
    return f"{v:.{d}f}" if v is not None and np.isfinite(v) else "-"


def ci_text(p, lo, hi, d=1):
    if p is None or not np.isfinite(p):
        return "-"
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def banner(t, W=118):
    print("\n" + "-" * W + f"\n  {t}\n" + "-" * W)


def gap_walk(o, h, l, c, entry, stop, target, n):
    """R of one trade with realistic fills. A day that OPENS beyond the stop
    or the target exits at that open (a gap); otherwise as the label - a day
    touching both counts as the stop; no exit within n days -> last close."""
    risk = entry - stop
    n = min(n, len(c))
    if risk <= 0 or n <= 0:
        return np.nan
    for i in range(n):
        if o[i] <= stop or o[i] >= target:
            return (o[i] - entry) / risk
        if l[i] <= stop:
            return -1.0
        if h[i] >= target:
            return (target - entry) / risk
    return (c[n - 1] - entry) / risk


def gap_fill_r(U, m, cache, g):
    """gap_walk for every row in mask m, rebuilt from the price files with the
    same levels as the labels (entry at the close, trade_config's levels)."""
    out = np.full(len(U), np.nan)
    rows = np.flatnonzero(m)
    if not rows.size:
        return out
    sub = U.iloc[rows]
    with short_limits(g):
        for t, grp in sub.groupby("ticker", sort=False):
            df = E64.P2.load_prices(t, cache)
            if df is None or not len(df):
                continue
            atr = E64._atr(df, 14).to_numpy(float)
            o, h, lo, c = (df[k].to_numpy(float)
                           for k in ("Open", "High", "Low", "Close"))
            pos = pd.Index(df.index).get_indexer(pd.DatetimeIndex(grp["date"]))
            for i, p in zip(grp.index, pos):
                if p < 0:
                    continue
                lv = TC.compute_levels(c[p], atr[p], g["key"])
                if lv.get("valid"):
                    out[i] = gap_walk(o[p + 1:], h[p + 1:], lo[p + 1:],
                                      c[p + 1:], c[p], lv["stop"],
                                      lv["target"], g["hold"])
    return out


def trade_row(label, st, U, m, years, rgap):
    """Every ENTER trade in mask m: outcomes, days held, R as labelled and
    with real fills (gaps) and costs, and the two matched comparisons."""
    r = st.row(m, years)
    mm = m & np.isin(U["year"].to_numpy(), years)
    if not r.get("Signals"):
        return {"": label, "Signals": "0"}
    ind = st.ind
    res = ind["tgt"][mm].sum() + ind["stp"][mm].sum()
    cost_r = COST / (U["stop_pct"].to_numpy(float)[mm] / 100.0)
    real = rgap[mm] - cost_r
    return {"": label,
            "Signals": f"{r['Signals']:,}",
            "Hit target %": pct(r["Target %"]),
            "Stop %": pct(r["Stop %"]),
            "Expired %": pct(r["Expired %"]),
            "Resolved hit %": pct(ind["tgt"][mm].sum() / res * 100
                                  if res else np.nan),
            "Days held": f"{U['bars_held'].to_numpy(float)[mm].mean():.1f}",
            "Avg R": f"{r['Avg R']:+.3f}",
            "Avg R, gaps + costs": (f"{np.nanmean(real):+.3f}"
                                    if np.isfinite(real).any() else "-"),
            "vs same stock-month, pp [90%]": ci_text(
                r["Target vs same stock-month pp"],
                r["Target vs same stock-month lo"],
                r["Target vs same stock-month hi"]),
            "vs average entry, pp [90%]": ci_text(
                r["Target vs average entry pp"],
                r["Target vs average entry lo"],
                r["Target vs average entry hi"])}


def group_rows(st, U, m, groups, years):
    """Signals, hit rate and lift over the same stock-month inside groups."""
    ind, pool = st.ind, st.pool
    yr = np.isin(U["year"].to_numpy(), years)
    out = []
    for name, g in groups:
        mm = m & g & yr
        n = int(mm.sum())
        row = {"Group": name, "Signals": f"{n:,}"}
        if n:
            row["Hit target %"] = pct(ind["tgt"][mm].mean() * 100)
            row["Same stock-month %"] = pct(pool["tgt"][mm].mean() * 100)
            if n >= 30:
                p, lo, hi = st.mean_ci((ind["tgt"] - pool["tgt"]) * 100,
                                       m & g, years)
                row["Lift, pp [90%]"] = ci_text(p, lo, hi)
            else:
                row["Lift, pp [90%]"] = "(fewer than 30)"
        out.append(row)
    return pd.DataFrame(out)


# =============================================================================
# 4. PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, g, n_boot, n_seeds, core_names, fresh_names):
    settings = {"g": g, "step": STEP, "cfg": CFG, "seeds": n_seeds,
                "q": V83.QUANTILE, "win": V82.WINDOW_DAYS,
                "liq": V86.LIQ_PCT, "dup": V86.DUP_CORR, "years": YEARS,
                "choose": CHOOSE_YEARS, "gate": MIN_SIGNALS, "n_boot": n_boot,
                "cost": COST, "build": BUILD, "core": core_names,
                "fresh": fresh_names}
    sig = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str)
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
    when = {"fresh": ", before any unseen-stock result was computed",
            "changed": ", before any unseen-stock result was computed "
                       "(REPLACES an earlier pre-registration)",
            "changed-after-result": " - REPLACES an earlier "
                                    "pre-registration AFTER a V100 result "
                                    "was seen (exploratory)"}[status]
    feats = textwrap.fill(", ".join(FEATS), width=66,
                          subsequent_indent=" " * 11)
    txt = PREREG.format(
        now=now, when=when, k=g["k"], hold=g["hold"], floor=g["floor"],
        ceil=g["ceil"], mid=MID_BARS, sl=g["sl"], feats=feats,
        emb=int(g["hold"] * 365.25 / 252) + 5,
        seeds=n_seeds, n_core=len(core_names), q=V83.QUANTILE,
        win=V82.WINDOW_DAYS, c0=CHOOSE_YEARS[0], c1=CHOOSE_YEARS[-1],
        n_fresh=len(fresh_names), y0=YEARS[0], y1=YEARS[-1],
        gate=MIN_SIGNALS, nb=n_boot, cost=COST,
        names_sig=hashlib.sha256(",".join(fresh_names).encode())
        .hexdigest()[:16])
    if os.path.exists(tp):
        with open(tp) as fh, open(os.path.join(
                out_dir, "preregistration_history.txt"), "a") as h:
            h.write(fh.read() + "\n" + "-" * 80 + "\n")
    open(tp, "w").write(txt)
    json.dump({"sig": sig, "written": now, "status": status}, open(mp, "w"))
    print(f"  wrote {tp} ({now})")
    if status == "changed-after-result":
        print("  *** settings changed AFTER a result was written - exploratory")
    return now, status != "changed-after-result"


# =============================================================================
# 5. RUNNER
# =============================================================================
def run_v100(hold=HOLD, model_in81="entry_model_v81.joblib", core_cache=None,
             big_cache="price_cache_v68", n_seeds=N_SEEDS, chunk=250,
             n_boot=4000, out_dir=OUT_DIR, verbose=True):
    t0 = time.time()
    W = 118
    g = geometry(hold)
    print("=" * W)
    print(f"V100 - A SHORT-HORIZON ENTRY MODEL: V88's recipe on a {g['hold']}-"
          f"day trade (pre-registered, unseen stocks)")
    print("=" * W)
    core_cache = core_cache or M81.load_model(model_in81)["provenance"][
        "price_cache"]
    print(f"  trade: target {g['k']} x ATR% x sqrt({g['hold']}), kept within "
          f"{g['floor']:.2%}-{g['ceil']:.2%}; stop {g['sl']} x target; exit "
          f"after {g['hold']} trading days")
    core, fresh, wf, core_names, fresh_names = prepare(
        core_cache, big_cache, g, n_seeds, chunk, verbose)
    print("  market breadth from the core stocks (regime table only) ...")
    b, _ = V82.full_breadth(core_cache, core_names)

    CORE = f"core {len(core_names)}"
    unis = {}
    for name, frame, mask, tag in (
            (LIQ, fresh, fresh["liquid"].to_numpy(bool), "fresh"),
            (ALL, fresh, np.ones(len(fresh), bool), "fresh"),
            (CORE, core, np.ones(len(core), bool), "core")):
        U, w = V87.universe(frame, mask, wf, tag)
        U["breadth"] = U["date_n"].map(b).to_numpy(float)
        unis[name] = (U, {MODEL: V85.model_fire(U, *w["combo"])})

    print("\n  PRE-REGISTRATION")
    when, prereg_ok = preregister(out_dir, g, n_boot, n_seeds, core_names,
                                  fresh_names)

    # ---- [0] the data ------------------------------------------------------
    banner(f"[0] THE DATA - candidate days every {STEP} trading days; "
           f"outcomes shown for the core only", W)
    rows = []
    for name in (CORE, LIQ, ALL):
        U, F = unis[name]
        yr = np.isin(U["year"].to_numpy(), YEARS)
        rec = {"data": name, "stocks": f"{U.loc[yr, 'ticker'].nunique():,}",
               f"candidate days {YEARS[0]}-{YEARS[-1]}": f"{int(yr.sum()):,}",
               "ENTER signals": f"{int((F[MODEL] & yr).sum()):,}"}
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))
    Uc = unis[CORE][0]
    ic = V93.outcome_indicators(Uc)
    tp_ = Uc["target_pct"].to_numpy(float)
    print(f"\n  core, every candidate day: target {ic['tgt'].mean():.1%}, stop "
          f"{ic['stp'].mean():.1%}, expired {ic['exp'].mean():.1%}; median "
          f"target {np.median(tp_):.2f}%, stop {np.median(Uc['stop_pct']):.2f}"
          f"%; at the lower limit {np.mean(tp_ <= g['floor'] * 100 + 0.02):.0%}"
          f", at the upper {np.mean(tp_ >= g['ceil'] * 100 - 0.02):.1%}")

    # ---- [1] the opponent, chosen on the core 2008-2016 --------------------
    Uc, Fc = unis[CORE]
    ev_c = V85.Evaluator(Uc, n_boot, seed=SEED)
    rows = []
    for f, m in zip(FEATS, MONO):
        nm = rule_name(f, m)
        Fc[nm] = rule(Uc, m * Uc[f].to_numpy(float))
        s = ev_c.summary(Fc[nm], CHOOSE_YEARS)
        rows.append({"Rule": nm, "Signals": s["signals"] if s else 0,
                     "Lift pp": s["lift"] if s else np.nan,
                     "lo": s["lo"] if s else np.nan,
                     "hi": s["hi"] if s else np.nan})
    T1 = pd.DataFrame(rows)
    T1.to_csv(os.path.join(out_dir, "opponent_choice.csv"), index=False)
    ok = T1["Lift pp"].notna()
    opp = T1.loc[T1.loc[ok, "Lift pp"].idxmax(), "Rule"] if ok.any() else None
    banner(f"[1] THE SIMPLE RULE TO BEAT - each feature alone, top 1%; chosen "
           f"on {CORE}, {CHOOSE_YEARS[0]}-{CHOOSE_YEARS[-1]}", W)
    D = pd.DataFrame({"Rule": T1["Rule"],
                      "Signals": T1["Signals"].map(lambda v: f"{int(v):,}"),
                      "Lift over same stock-month, pp [90%]": [
                          ci_text(p, lo, hi) for p, lo, hi in
                          zip(T1["Lift pp"], T1["lo"], T1["hi"])]})
    print(D.to_string(index=False))
    if (T1["Signals"] == 0).any():
        print("  a rule with 0 signals never fired: too many days tie exactly "
              "at its cut (e.g. a close at the 10-day low), and the rule needs "
              "a score strictly above it")
    print(f"\n  chosen opponent: {opp or 'none (no rule fired enough)'}")

    for name in (LIQ, ALL):
        U, F = unis[name]
        if opp:
            f, m = next((f, m) for f, m in zip(FEATS, MONO)
                        if rule_name(f, m) == opp)
            F[opp] = rule(U, m * U[f].to_numpy(float))
        F[RANDOM] = rule(U, np.random.default_rng(SEED).random(len(U)))

    # ---- [2] primary --------------------------------------------------------
    lines, decided = [], False
    U, F = unis[LIQ]
    yr = np.isin(U["year"].to_numpy(), YEARS)
    n_sig = int((F[MODEL] & yr).sum())
    banner(f"[2] PRIMARY - {LIQ} stocks, {YEARS[0]}-{YEARS[-1]} (pre-registered)",
           W)
    print(f"  power gate: {n_sig:,} ENTER signals (needs {MIN_SIGNALS:,}) -> "
          f"{'OK' if n_sig >= MIN_SIGNALS else 'TOO FEW'}")
    ev = V85.Evaluator(U, n_boot, seed=SEED)
    st = V93.Stats(U, n_boot)
    if n_sig < MIN_SIGNALS:
        lines.append(f"  VERDICT: INCONCLUSIVE - only {n_sig:,} ENTER signals; "
                     "the outcomes were not read.")
    else:
        decided = True
        rows = []
        for arm in [MODEL] + ([opp] if opp else []) + [RANDOM]:
            s = ev.summary(F[arm], YEARS)
            both = int((F[arm] & F[MODEL] & yr).sum())
            rows.append({"Arm": arm, "Signals": f"{s['signals']:,}" if s
                         else "0",
                         "Lift over same stock-month, pp [90%]":
                             ci_text(s["lift"], s["lo"], s["hi"]) if s
                             else "-",
                         "Excess R": f"{s['expR']:+.3f}" if s else "-",
                         "Shared with the model": (
                             f"{both / max(s['signals'], 1):.0%}"
                             if s and arm != MODEL else "")})
        T2 = pd.DataFrame(rows)
        print(T2.to_string(index=False))
        T2.to_csv(os.path.join(out_dir, "primary.csv"), index=False)
        sm = ev.summary(F[MODEL], YEARS)
        c1 = bool(sm and sm["lo"] > 0)
        lines += [f"  1. TIMING - ENTER days vs all days of the same stock and "
                  f"month: {ci_text(sm['lift'], sm['lo'], sm['hi'])} pp "
                  f"-> {'CONFIRMED' if c1 else 'NOT CONFIRMED'}"]
        worse = False
        if opp:
            p = ev.paired(F[MODEL], F[opp], YEARS)
            if p is None:
                v2 = "NOT MEASURABLE (the rule fired fewer than 30 times)"
                txt = "-"
            else:
                txt = ci_text(p["diff"], p["lo"], p["hi"])
                worse = p["hi"] < 0
                v2 = ("BEATS the rule" if p["lo"] > 0 else
                      "WORSE than the rule" if worse else
                      "AHEAD BUT NOT PROVEN" if p["diff"] > 0 else
                      "DOES NOT BEAT the rule")
            lines.append(f"  2. RULE - model minus '{opp}', paired by month: "
                         f"{txt} pp -> {v2}")
        else:
            lines.append("  2. RULE - no opponent could be chosen")
        lines.append("\n  VERDICT: " + (
            "NOT CONFIRMED - no timing edge for the two-week trade with this "
            "recipe." if not c1 else
            "TIMING CONFIRMED, but the simple rule did better - improve the "
            "model before serving it." if worse else
            "CONFIRMED - the short model times entries on unseen stocks. Next: "
            "train the final short model and add it to the recommender."))

    # ---- [3] what the trades look like --------------------------------------
    if decided:
        banner("[3] WHAT THE TRADES LOOK LIKE - every ENTER signal "
               "(break-even 37.5% of resolved trades; descriptive)", W)
        print("  re-walking the ENTER trades from the price files for real "
              "fills (gaps) ...")
        rows, stats, gaps = [], {LIQ: st}, {}
        for name in (LIQ, CORE, ALL):
            Ux, Fx = unis[name]
            stx = stats.get(name) or V93.Stats(Ux, n_boot)
            stats[name] = stx
            m_all = Fx[MODEL] | (Fx[opp] if opp and opp in Fx else False)
            gaps[name] = gap_fill_r(Ux, m_all, core_cache if name == CORE
                                    else big_cache, g)
            periods = ([("2008-2026", YEARS)] + list(HALVES.items())
                       if name != ALL else [("2008-2026", YEARS)])
            for lab, yrs in periods:
                rows.append(trade_row(f"{name}, {lab}", stx, Ux, Fx[MODEL],
                                      yrs, gaps[name]))
        if opp:
            rows.append(trade_row(f"{LIQ}, 2008-2026: rule '{opp}'", st, U,
                                  F[opp], YEARS, gaps[LIQ]))
        T3 = pd.DataFrame(rows)
        T3.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        cols = list(T3.columns)
        print(T3[cols[:7]].to_string(index=False))
        print()
        print(T3[[cols[0]] + cols[7:]].to_string(index=False))
        mm = F[MODEL] & yr
        stopped = mm & (st.ind["stp"] > 0)
        gapped = stopped & (gaps[LIQ] < -1 - 1e-9)
        if stopped.any():
            print(f"\n  real fills, {LIQ} ENTER trades: {gapped.sum() / stopped.sum():.0%}"
                  f" of the stopped trades opened below the stop"
                  + (f" (average exit {np.nanmean(gaps[LIQ][gapped]):+.2f}R)"
                     if gapped.any() else "")
                  + f"; costs take about "
                  f"{np.nanmean(COST / (U['stop_pct'].to_numpy(float)[mm] / 100)):.3f}"
                  f"R a trade")

        # ---- [4] regimes and years ----------------------------------------
        bb = U["breadth"].to_numpy(float)
        groups = [(f"bull (breadth > {BULL:.2f})", bb > BULL),
                  ("neutral", (bb >= BEAR) & (bb <= BULL)),
                  (f"bear (breadth < {BEAR:.2f})", bb < BEAR)]
        T4 = group_rows(st, U, F[MODEL], groups, YEARS)
        banner(f"[4] BY MARKET REGIME - {LIQ}, {YEARS[0]}-{YEARS[-1]} "
               f"(descriptive)", W)
        print(T4.to_string(index=False))
        T4.to_csv(os.path.join(out_dir, "regimes.csv"), index=False)

        banner("[5] BY YEAR - ENTER signals (descriptive)", W)
        yrs = U["year"].to_numpy()
        Ty = group_rows(st, U, F[MODEL], [(str(y), yrs == y) for y in YEARS],
                        YEARS).rename(columns={"Group": "Year"})
        Ucx, Fcx = unis[CORE]
        stc = stats[CORE]
        Tc = group_rows(stc, Ucx, Fcx[MODEL],
                        [(str(y), Ucx["year"].to_numpy() == y) for y in YEARS],
                        YEARS)
        Ty = Ty.merge(Tc.rename(columns={
            "Group": "Year", "Signals": "core signals",
            "Hit target %": "core hit %",
            "Same stock-month %": "core same stock-month %",
            "Lift, pp [90%]": "core lift, pp [90%]"}), on="Year", how="left")
        print(Ty.fillna("").to_string(index=False))
        Ty.to_csv(os.path.join(out_dir, "years.csv"), index=False)

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
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_HOLD      = 10                  # trading days; fix BEFORE the first run
    RUN_BIG_CACHE = "price_cache_v68"
    RUN_CHUNK     = 250
    RUN_N_SEEDS   = 5
    RUN_N_BOOT    = 4000
    RUN_OUT_DIR   = "thesis_tables_v100"
    # -------------------------------------------------------------------------

    run_v100(hold=RUN_HOLD, big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS,
             chunk=RUN_CHUNK, n_boot=RUN_N_BOOT, out_dir=RUN_OUT_DIR)
