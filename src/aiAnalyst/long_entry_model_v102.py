"""
CHANGED IN V106: ci() is copied in from fundamental_filter_v98 (same four
lines), so this file no longer needs that module. The original is in
log/originals_v106/long_entry_model_v102.py.

V102 - A LONG-HORIZON ENTRY MODEL: V88's recipe on a one-year trade
(pre-registered, tested on stocks it has never seen)

WHY
---
The system has three horizons. V88 times entries for a 60-day trade and V100/
V101 for a 10-day trade. trade_config's LONG horizon holds for 250 trading
days - about a year. The question is the same as V100's: retrained on a
one-year trade, does V88's recipe pick better entry days, on stocks it has
never seen, than the other days of the same stock and month?

Expect a smaller effect than at 10 or 60 days: a dip of a few percent matters
less against a year of price moves. A clear "no" would be a result too: it
would say that for a one-year hold, which day of the month you buy matters
little.

WHAT IS FIXED BEFORE ANY RESULT (nothing here was tuned on long trades)
-----------------------------------------------------------------------
  TRADE     enter at the signal day's close. Target = 0.5 x the stock's own
            expected move over the hold (ATR% x sqrt(250)), kept between V88's
            8% and 40% scaled by sqrt(250 / 60) - 16.33% to 81.65% - so the
            limits bind for the same share of stocks as in V88 and V100.
            Stop = 0.6 x target (break-even 37.5% of trades that end at the
            target or the stop). Neither hit within 250 trading days: exit at
            that close. A day that touches both counts as the stop.
  MODEL     V88's recipe unchanged: a classifier on the same ten "low is good"
            features, shallow trees, 5 seeds; fold models retrained every year
            on the core stocks only, with a gap of a year and a few days before
            each test year (a label needs a year of future prices).
  SIGNAL    ENTER when the score is strictly above the 99th percentile of the
            universe's own scores over the trailing 252 trading days, refit
            monthly - V88's rule.
  OPPONENT  each of the ten features alone with the same 1% rule; the one with
            the best lift on the CORE stocks, 2008-2016, is chosen in this run,
            before any unseen-stock result is computed.

THE TEST
--------
  PRIMARY   unseen liquid stocks (V86's list and liquidity floor), 2008-2025
            (a label needs a year of future prices, so the data stops about a
            year before the price files do). Power gate (counted before any
            outcome is read): at least 500 ENTER signals, else INCONCLUSIVE.
     1. TIMING  ENTER days hit the target more often than all days of the same
                stock and month: 90% lower bound above 0 -> CONFIRMED.
     2. RULE    model minus the opponent, paired: lower bound above 0 -> BEATS;
                upper bound below 0 -> WORSE; otherwise AHEAD BUT NOT PROVEN
                (estimate above 0) or DOES NOT BEAT.
  INTERVALS one-year trades started a month apart share most of their path, so
            calendar months are resampled in BLOCKS OF 12 CONSECUTIVE MONTHS
            (circular, paired across arms) - the length of the trade. The plain
            month bootstrap of V86-V100 would make the intervals too narrow; it
            is printed beside, for reference only.
  REPORTED  what the trades look like (hit, stop, expiry, days held, R as
            labelled and with real fills and costs), market regimes, years,
            both halves, all unseen stocks and the core - descriptive.

The first run builds the one-year-trade data and fold models (30-60 minutes,
cached - an interrupted run resumes); later runs take a few minutes.
"""

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
import short_entry_model_v100 as V100

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = "thesis_tables_v102"
CACHE_DIR = "long_cache_v102"
BUILD = "v102.1"
HOLD = TC.HORIZON_CONFIGS["LONG"]["lookahead_bars"]     # 250 trading days
STEP = E64.STEP
MID_BARS = TC.HORIZON_CONFIGS["MID"]["lookahead_bars"]
COST = V100.COST
CFG = V100.CFG                     # V88's recipe
FEATS, MONO = V100.FEATS, V100.MONO
N_SEEDS = 5
MIN_TRAIN_YEARS = V100.MIN_TRAIN_YEARS
YEARS = list(range(2008, 2026))
CHOOSE_YEARS = list(range(2008, 2017))
HALVES = {"2008-2016": list(range(2008, 2017)),
          "2017-2025": list(range(2017, 2026))}
BLOCK = 12                         # months per resampled block (one trade)
MIN_SIGNALS = 500
SEED = 102
BEAR, BULL = V82.BREADTH_CUTS

LIQ, ALL, MODEL, RANDOM = ("unseen, liquid", "unseen, all", "model",
                           "random 1% (yardstick)")

PREREG = """PRE-REGISTRATION - V102 long-horizon entry model
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
           2. RULE: model minus opponent, paired: lower bound > 0 -> BEATS;
              upper bound < 0 -> WORSE; else AHEAD BUT NOT PROVEN (estimate
              > 0) or DOES NOT BEAT.
METHOD     intervals from a circular block bootstrap of calendar months,
           blocks of {block} consecutive months, {nb} draws, the same resampled
           months for every arm (paired). The plain month bootstrap is printed
           for reference only. Costs ({cost:.2%} round trip) and gap fills are
           reported in R, not used in the decisions.
SECONDARY  halves, all unseen stocks, the core, regimes, years - descriptive.
NAMES      sha256 of the unseen list: {names_sig}
"""


# =============================================================================
# 1. THE ONE-YEAR TRADE
# =============================================================================
def geometry(hold=HOLD):
    """V88's barrier rule over a one-year hold: same K and stop ratio, limits
    scaled by sqrt(hold / 60) so they bind for the same share of stocks."""
    s = math.sqrt(hold / MID_BARS)
    return {"key": f"LONG_D{hold}", "hold": int(hold), "k": TC.VOL_K,
            "sl": TC.SL_TP_RATIO, "floor": round(TC.VOL_MIN_TP_PCT * s, 6),
            "ceil": round(TC.VOL_MAX_TP_PCT * s, 6)}


def register(g):
    """Make the one-year trade known to trade_config under its own key (so
    V64's label builder and V86's walk-forward gap use 250 bars). Adds a key;
    trade_config's LONG entry and everything else stay as they are."""
    if TC.GEOMETRY_MODE != "VOL_SCALED":
        raise SystemExit("V102 assumes trade_config.GEOMETRY_MODE = "
                         "'VOL_SCALED', as V88 was built.")
    atr_fields = {k: TC.HORIZON_CONFIGS["LONG"][k]
                  for k in ("atr_stop_mult", "atr_target_mult",
                            "max_stop_pct", "min_profit_pct", "min_rr")}
    TC.HORIZON_CONFIGS[g["key"]] = {
        "interval": "1d", "lookahead_bars": g["hold"],
        "eval_days": int(round(g["hold"] * 365.25 / 252)), **atr_fields}


# =============================================================================
# 2. DATA (V100's builder, in this version's own cache folder)
# =============================================================================
def build(names, cache, g, tag, chunk, verbose=True):
    """Candidate days with one-year-trade labels, cached in chunks of names."""
    chunks = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    parts, t0, built = [], time.time(), 0
    for k, ch in enumerate(chunks, 1):
        key = V86._hash([tag, cache, ch, g, STEP, BUILD])
        path = os.path.join(CACHE_DIR, f"{tag}_{key}.pkl")
        if os.path.exists(path):
            part, how = pd.read_pickle(path), "cached"
        else:
            t = time.time()
            with V100.short_limits(g):        # the trade's own target limits
                d = E64.build_dataset(ch, cache, g["key"], STEP,
                                      verbose=False)
            part = (V100._slim(d, cache, ch) if len(d) else pd.DataFrame(
                columns=V100.KEEP + ["date_n"] + V86.SHORT_FEATS))
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
    stocks. Reuses V86's (or V100's) cached duplicate check."""
    core_names = V86.cache_names(core_cache)
    core_set = set(core_names)
    cand = [t for t in V86.cache_names(big_cache) if t not in core_set]
    dkey = V86._hash(["dups", core_cache, sorted(core_set), big_cache, cand,
                      V86.DUP_CORR, V86.DUP_MIN_OVERLAP])
    for folder in (V86.CACHE_DIR, V100.CACHE_DIR, CACHE_DIR):
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
    print(f"  one-year-trade candidate days (built once, then cached in "
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
        "long model (V88's recipe)", sig, cache_dir=CACHE_DIR,
        verbose=verbose)}
    return core, fresh, wf, core_names, fresh_names


# =============================================================================
# 3. BLOCK BOOTSTRAP (V87's circular blocks, with access to the draws)
# =============================================================================
class Blocks:
    """Calendar months resampled in circular blocks of L consecutive months.
    One set of weights per period, shared by every comparison, so differences
    are paired. L = 1 is the plain month bootstrap used up to V100."""

    def __init__(self, dates, years, n_boot, L, seed=SEED):
        per = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
        self.mcode, self.months = pd.factorize(per)
        self.year = np.asarray(years)
        self.n_boot, self.L, self.seed, self._W = n_boot, int(L), seed, {}

    def _w(self, years):
        key = tuple(years)
        if key not in self._W:
            my = np.array([m.year for m in self.months])
            idx = np.flatnonzero(np.isin(my, years))
            idx = idx[np.argsort(self.months[idx].to_timestamp().values)]
            n, L = len(idx), self.L
            rng = np.random.default_rng([self.seed, L, years[0], years[-1]])
            k = int(np.ceil(n / L)) if n else 0
            if n:
                starts = rng.integers(0, n, size=(self.n_boot, k))
                flat = ((starts[:, :, None] + np.arange(L)) % n) \
                    .reshape(self.n_boot, -1)[:, :n]
                flat = flat + np.arange(self.n_boot)[:, None] * n
                W = np.bincount(flat.ravel(), minlength=self.n_boot * n) \
                    .reshape(self.n_boot, n).astype(float)
            else:
                W = np.zeros((self.n_boot, 0))
            self._W[key] = (idx, W)
        return self._W[key]

    def mean(self, x, m, years):
        idx, W = self._w(years)
        x = np.asarray(x, float)
        mm = m & np.isin(self.year, years) & np.isfinite(x)
        K = len(self.months)
        s = np.bincount(self.mcode[mm], x[mm], K)[idx]
        c = np.bincount(self.mcode[mm], minlength=K)[idx].astype(float)
        if c.sum() < 1:
            return np.nan, np.full(self.n_boot, np.nan), 0
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (W @ s) / (W @ c)
        return s.sum() / c.sum(), draws, int(c.sum())

    def diff(self, xa, ma, xb, mb, years):
        pa, da, na = self.mean(xa, ma, years)
        pb, db, nb = self.mean(xb, mb, years)
        return pa - pb, da - db, na, nb


def ci(draws):
    """5-95% interval of bootstrap draws (was fundamental_filter_v98.ci)."""
    d = draws[np.isfinite(draws)]
    return (np.percentile(d, 5), np.percentile(d, 95)) if d.size else \
        (np.nan, np.nan)


def ci_text(p, draws, d=1):
    if p is None or not np.isfinite(p):
        return "-"
    lo, hi = ci(draws)
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def excess(U):
    """Per row: target hit minus its stock-month pool, and minus the same
    month's average entry (all stocks) - in percentage points."""
    ind = V93.outcome_indicators(U)
    pool, base = V93.group_means(U, ind["tgt"])
    return ind, {"sm": (ind["tgt"] - pool) * 100,
                 "avg": (ind["tgt"] - base) * 100, "pool": pool * 100}


# =============================================================================
# 4. TABLES
# =============================================================================
def trade_row(label, U, ind, ex, m, years, rgap, B):
    mm = m & np.isin(U["year"].to_numpy(), years)
    n = int(mm.sum())
    if not n:
        return {"": label, "Signals": "0"}
    res = ind["tgt"][mm].sum() + ind["stp"][mm].sum()
    cost_r = COST / (U["stop_pct"].to_numpy(float)[mm] / 100.0)
    real = rgap[mm] - cost_r
    p1, d1, _ = B.mean(ex["sm"], m, years)
    p2, d2, _ = B.mean(ex["avg"], m, years)
    return {"": label, "Signals": f"{n:,}",
            "Hit target %": V100.pct(ind["tgt"][mm].mean() * 100),
            "Stop %": V100.pct(ind["stp"][mm].mean() * 100),
            "Expired %": V100.pct(ind["exp"][mm].mean() * 100),
            "Resolved hit %": V100.pct(ind["tgt"][mm].sum() / res * 100
                                       if res else np.nan),
            "Days held": f"{U['bars_held'].to_numpy(float)[mm].mean():.0f}",
            "Avg R": f"{np.nanmean(ind['R'][mm]):+.3f}",
            "Avg R, gaps + costs": (f"{np.nanmean(real):+.3f}"
                                    if np.isfinite(real).any() else "-"),
            "vs same stock-month, pp [90%]": ci_text(p1, d1),
            "vs average entry, pp [90%]": ci_text(p2, d2)}


def group_rows(U, ind, ex, m, groups, years, B, with_ci=True):
    yr = np.isin(U["year"].to_numpy(), years)
    out = []
    for name, g in groups:
        mm = m & g & yr
        n = int(mm.sum())
        row = {"Group": name, "Signals": f"{n:,}"}
        if n:
            row["Hit target %"] = V100.pct(ind["tgt"][mm].mean() * 100)
            row["Same stock-month %"] = V100.pct(ex["pool"][mm].mean())
            p, d, _ = B.mean(ex["sm"], m & g, years)
            row["Lift, pp" + (" [90%]" if with_ci else "")] = (
                ci_text(p, d) if with_ci and n >= 30 else f"{p:+.1f}")
        out.append(row)
    return pd.DataFrame(out)


# =============================================================================
# 5. PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, g, n_boot, n_seeds, core_names, fresh_names):
    settings = {"g": g, "step": STEP, "cfg": CFG, "seeds": n_seeds,
                "q": V83.QUANTILE, "win": V82.WINDOW_DAYS,
                "liq": V86.LIQ_PCT, "dup": V86.DUP_CORR, "years": YEARS,
                "choose": CHOOSE_YEARS, "gate": MIN_SIGNALS, "block": BLOCK,
                "n_boot": n_boot, "cost": COST, "build": BUILD,
                "core": core_names, "fresh": fresh_names}
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
                                    "pre-registration AFTER a V102 result "
                                    "was seen (exploratory)"}[status]
    feats = textwrap.fill(", ".join(FEATS), width=66,
                          subsequent_indent=" " * 11)
    txt = PREREG.format(
        now=now, when=when, k=g["k"], hold=g["hold"], floor=g["floor"],
        ceil=g["ceil"], mid=MID_BARS, sl=g["sl"], feats=feats,
        emb=int(g["hold"] * 365.25 / 252) + 5, seeds=n_seeds,
        n_core=len(core_names), q=V83.QUANTILE, win=V82.WINDOW_DAYS,
        c0=CHOOSE_YEARS[0], c1=CHOOSE_YEARS[-1], n_fresh=len(fresh_names),
        y0=YEARS[0], y1=YEARS[-1], gate=MIN_SIGNALS, block=BLOCK, nb=n_boot,
        cost=COST, names_sig=hashlib.sha256(",".join(fresh_names).encode())
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
# 6. RUNNER
# =============================================================================
def run_v102(hold=HOLD, model_in81="entry_model_v81.joblib", core_cache=None,
             big_cache="price_cache_v68", n_seeds=N_SEEDS, chunk=250,
             n_boot=4000, out_dir=OUT_DIR, verbose=True):
    t0 = time.time()
    W = 118
    g = geometry(hold)
    print("=" * W)
    print(f"V102 - A LONG-HORIZON ENTRY MODEL: V88's recipe on a {g['hold']}-"
          f"day (one-year) trade (pre-registered, unseen stocks)")
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
    V100.banner(f"[0] THE DATA - candidate days every {STEP} trading days; "
                f"outcomes shown for the core only", W)
    rows = []
    for name in (CORE, LIQ, ALL):
        U, F = unis[name]
        yr = np.isin(U["year"].to_numpy(), YEARS)
        rows.append({"data": name,
                     "stocks": f"{U.loc[yr, 'ticker'].nunique():,}",
                     f"candidate days {YEARS[0]}-{YEARS[-1]}":
                         f"{int(yr.sum()):,}",
                     "ENTER signals": f"{int((F[MODEL] & yr).sum()):,}",
                     "last candidate day": str(U["date"].max().date())})
    print(pd.DataFrame(rows).to_string(index=False))
    Uc = unis[CORE][0]
    ic = V93.outcome_indicators(Uc)
    tp_ = Uc["target_pct"].to_numpy(float)
    print(f"\n  core, every candidate day: target {ic['tgt'].mean():.1%}, stop "
          f"{ic['stp'].mean():.1%}, expired {ic['exp'].mean():.1%}; median "
          f"target {np.median(tp_):.1f}%, stop {np.median(Uc['stop_pct']):.1f}"
          f"%; at the lower limit {np.mean(tp_ <= g['floor'] * 100 + 0.02):.0%}"
          f", at the upper {np.mean(tp_ >= g['ceil'] * 100 - 0.02):.1%}; days "
          f"held {Uc['bars_held'].mean():.0f} on average")

    # ---- [1] the opponent, chosen on the core 2008-2016 --------------------
    Uc, Fc = unis[CORE]
    ind_c, ex_c = excess(Uc)
    B1c = Blocks(Uc["date"], Uc["year"], n_boot, 1)
    rows = []
    for f, m in zip(FEATS, MONO):
        nm = V100.rule_name(f, m)
        Fc[nm] = V100.rule(Uc, m * Uc[f].to_numpy(float))
        p, d, n = B1c.mean(ex_c["sm"], Fc[nm], CHOOSE_YEARS)
        rows.append({"Rule": nm, "Signals": n,
                     "Lift pp": p if n >= 30 else np.nan, "draws": d})
    T1 = pd.DataFrame(rows)
    T1.drop(columns="draws").to_csv(os.path.join(out_dir,
                                                 "opponent_choice.csv"),
                                    index=False)
    ok = T1["Lift pp"].notna()
    opp = T1.loc[T1.loc[ok, "Lift pp"].idxmax(), "Rule"] if ok.any() else None
    V100.banner(f"[1] THE SIMPLE RULE TO BEAT - each feature alone, top 1%; "
                f"chosen on {CORE}, {CHOOSE_YEARS[0]}-{CHOOSE_YEARS[-1]} "
                f"(month intervals; the choice uses the estimate)", W)
    D = pd.DataFrame({"Rule": T1["Rule"],
                      "Signals": T1["Signals"].map(lambda v: f"{int(v):,}"),
                      "Lift over same stock-month, pp [90%]": [
                          ci_text(p, d) for p, d in
                          zip(T1["Lift pp"], T1["draws"])]})
    print(D.to_string(index=False))
    if (T1["Signals"] == 0).any():
        print("  a rule with 0 signals never fired: too many days tie exactly "
              "at its cut (e.g. a close at the 10-day low)")
    print(f"\n  chosen opponent: {opp or 'none (no rule fired enough)'}")

    for name in (LIQ, ALL):
        U, F = unis[name]
        if opp:
            f, m = next((f, m) for f, m in zip(FEATS, MONO)
                        if V100.rule_name(f, m) == opp)
            F[opp] = V100.rule(U, m * U[f].to_numpy(float))
        F[RANDOM] = V100.rule(U, np.random.default_rng(SEED).random(len(U)))

    # ---- [2] primary --------------------------------------------------------
    lines, decided = [], False
    U, F = unis[LIQ]
    yr = np.isin(U["year"].to_numpy(), YEARS)
    n_sig = int((F[MODEL] & yr).sum())
    V100.banner(f"[2] PRIMARY - {LIQ} stocks, {YEARS[0]}-{YEARS[-1]} "
                f"(pre-registered; {BLOCK}-month blocks)", W)
    print(f"  power gate: {n_sig:,} ENTER signals (needs {MIN_SIGNALS:,}) -> "
          f"{'OK' if n_sig >= MIN_SIGNALS else 'TOO FEW'}")
    ind, ex = excess(U)
    B = Blocks(U["date"], U["year"], n_boot, BLOCK)
    B1 = Blocks(U["date"], U["year"], n_boot, 1)
    if n_sig < MIN_SIGNALS:
        lines.append(f"  VERDICT: INCONCLUSIVE - only {n_sig:,} ENTER signals; "
                     "the outcomes were not read.")
    else:
        decided = True
        rows = []
        for arm in [MODEL] + ([opp] if opp else []) + [RANDOM]:
            p, d, n = B.mean(ex["sm"], F[arm], YEARS)
            _, d1, _ = B1.mean(ex["sm"], F[arm], YEARS)
            both = int((F[arm] & F[MODEL] & yr).sum())
            rows.append({"Arm": arm, "Signals": f"{n:,}",
                         f"Lift, pp [90%, {BLOCK}-month blocks]":
                             ci_text(p, d),
                         "same, month intervals (reference)": ci_text(p, d1),
                         "Shared with the model": (f"{both / max(n, 1):.0%}"
                                                   if arm != MODEL else "")})
        T2 = pd.DataFrame(rows)
        print(T2.to_string(index=False))
        T2.to_csv(os.path.join(out_dir, "primary.csv"), index=False)
        pm, dm, _ = B.mean(ex["sm"], F[MODEL], YEARS)
        lo_m = ci(dm)[0]
        c1 = bool(np.isfinite(lo_m) and lo_m > 0)
        lines.append(f"  1. TIMING - ENTER days vs all days of the same stock "
                     f"and month: {ci_text(pm, dm)} pp ({BLOCK}-month blocks) "
                     f"-> {'CONFIRMED' if c1 else 'NOT CONFIRMED'}")
        worse = False
        if opp:
            pd_, dd, na, nb = B.diff(ex["sm"], F[MODEL], ex["sm"], F[opp],
                                     YEARS)
            if na < 30 or nb < 30:
                v2, txt = ("NOT MEASURABLE (an arm fired fewer than 30 "
                           "times)"), "-"
            else:
                lo, hi = ci(dd)
                txt = ci_text(pd_, dd)
                worse = hi < 0
                v2 = ("BEATS the rule" if lo > 0 else
                      "WORSE than the rule" if worse else
                      "AHEAD BUT NOT PROVEN" if pd_ > 0 else
                      "DOES NOT BEAT the rule")
            lines.append(f"  2. RULE - model minus '{opp}', paired: {txt} pp "
                         f"-> {v2}")
        else:
            lines.append("  2. RULE - no opponent could be chosen")
        lines.append("\n  VERDICT: " + (
            "NOT CONFIRMED - no timing edge for the one-year trade with this "
            "recipe: for a one-year hold, the day of the month matters "
            "little." if not c1 else
            "TIMING CONFIRMED, but the simple rule did better - improve the "
            "model before serving it." if worse else
            "CONFIRMED - the long model times entries on unseen stocks. Next: "
            "train the final long model and add it to the recommender."))

    # ---- [3] what the trades look like --------------------------------------
    if decided:
        V100.banner(f"[3] WHAT THE TRADES LOOK LIKE - every ENTER signal "
                    f"(break-even 37.5% of resolved trades; {BLOCK}-month "
                    f"blocks; descriptive)", W)
        print("  re-walking the ENTER trades from the price files for real "
              "fills (gaps) ...")
        rows, gaps, cache_of = [], {}, {}
        for name in (LIQ, CORE, ALL):
            Ux, Fx = unis[name]
            if name == LIQ:
                indx, exx, Bx = ind, ex, B
            else:
                indx, exx = excess(Ux)
                Bx = Blocks(Ux["date"], Ux["year"], n_boot, BLOCK)
            cache_of[name] = (indx, exx, Bx)
            m_all = Fx[MODEL] | (Fx[opp] if opp and opp in Fx else False)
            gaps[name] = V100.gap_fill_r(Ux, m_all, core_cache if name == CORE
                                         else big_cache, g)
            periods = ([(f"{YEARS[0]}-{YEARS[-1]}", YEARS)]
                       + list(HALVES.items())
                       if name != ALL else [(f"{YEARS[0]}-{YEARS[-1]}",
                                             YEARS)])
            for lab, yrs in periods:
                rows.append(trade_row(f"{name}, {lab}", Ux, indx, exx,
                                      Fx[MODEL], yrs, gaps[name], Bx))
        if opp:
            rows.append(trade_row(f"{LIQ}, {YEARS[0]}-{YEARS[-1]}: rule "
                                  f"'{opp}'", U, ind, ex, F[opp], YEARS,
                                  gaps[LIQ], B))
        T3 = pd.DataFrame(rows)
        T3.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        cols = list(T3.columns)
        print(T3[cols[:7]].to_string(index=False))
        print()
        print(T3[[cols[0]] + cols[7:]].to_string(index=False))
        mm = F[MODEL] & yr
        stopped = mm & (ind["stp"] > 0)
        gapped = stopped & (gaps[LIQ] < -1 - 1e-9)
        if stopped.any():
            print(f"\n  real fills, {LIQ} ENTER trades: "
                  f"{gapped.sum() / stopped.sum():.0%} of the stopped trades "
                  f"opened below the stop"
                  + (f" (average exit {np.nanmean(gaps[LIQ][gapped]):+.2f}R)"
                     if gapped.any() else "")
                  + "; costs take about "
                  f"{np.nanmean(COST / (U['stop_pct'].to_numpy(float)[mm] / 100)):.3f}"
                  f"R a trade")

        # ---- [4] regimes and years -----------------------------------------
        bb = U["breadth"].to_numpy(float)
        groups = [(f"bull (breadth > {BULL:.2f})", bb > BULL),
                  ("neutral", (bb >= BEAR) & (bb <= BULL)),
                  (f"bear (breadth < {BEAR:.2f})", bb < BEAR)]
        T4 = group_rows(U, ind, ex, F[MODEL], groups, YEARS, B)
        V100.banner(f"[4] BY MARKET REGIME - {LIQ}, {YEARS[0]}-{YEARS[-1]} "
                    f"({BLOCK}-month blocks; descriptive)", W)
        print(T4.to_string(index=False))
        T4.to_csv(os.path.join(out_dir, "regimes.csv"), index=False)

        V100.banner("[5] BY YEAR - ENTER signals (estimates only: one year is "
                    "about one block; descriptive)", W)
        yrs = U["year"].to_numpy()
        Ty = group_rows(U, ind, ex, F[MODEL], [(str(y), yrs == y)
                                               for y in YEARS], YEARS, B,
                        with_ci=False).rename(columns={"Group": "Year"})
        Ucx, Fcx = unis[CORE]
        indc, exc, Bc = cache_of[CORE]
        Tc = group_rows(Ucx, indc, exc, Fcx[MODEL],
                        [(str(y), Ucx["year"].to_numpy() == y)
                         for y in YEARS], YEARS, Bc, with_ci=False)
        Ty = Ty.merge(Tc.rename(columns={
            "Group": "Year", "Signals": "core signals",
            "Hit target %": "core hit %",
            "Same stock-month %": "core same stock-month %",
            "Lift, pp": "core lift, pp"}), on="Year", how="left")
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
    RUN_HOLD      = 250                 # trade_config's LONG; fix BEFORE the
                                        # first run
    RUN_BIG_CACHE = "price_cache_v68"
    RUN_CHUNK     = 250
    RUN_N_SEEDS   = 5
    RUN_N_BOOT    = 4000
    RUN_OUT_DIR   = "thesis_tables_v102"
    # -------------------------------------------------------------------------

    run_v102(hold=RUN_HOLD, big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS,
             chunk=RUN_CHUNK, n_boot=RUN_N_BOOT, out_dir=RUN_OUT_DIR)
