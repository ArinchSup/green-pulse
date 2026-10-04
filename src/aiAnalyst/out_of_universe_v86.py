"""
V86 - THE OUT-OF-UNIVERSE TEST.

WHY THIS TEST
-------------
V85 left the model level with the simple rule on the 337 names it was built on:
V81 ahead of `low bb_position` by about one point, interval straddling zero. On
those names two things limit any verdict:
  * every choice so far (features, directions, the 1% cut, the strict rule)
    was made looking at them, and
  * they give ~180 signals a year, so the paired interval is about +/-4 pp.

The ~2,000 names in price_cache_v68 that are NOT in the core cache fix both.
No row from them was ever used to train, select a feature, pick a direction or
set the rule. And a universe several times larger fires several times as
often, which is the only thing that can turn a one-point edge into a proven
one - or show that it was never there.

WHAT IS HELD FIXED
------------------
  MODEL     V81 exactly: expectancy XGBoost, the six features, monotone "low is
            good", 5 seeds (81-85). Fold models are retrained year by year on
            the CORE names only - the same models V83/V85 measured (checked:
            the core fire count must reproduce V83's).
  OPPONENT  `low bb_position`, fixed in V85 before this test existed.
  RULE      strict > the 99th percentile of the trailing 252 days, refit
            monthly. On fresh names the trailing window is the FRESH
            universe's own scores - what score() does when pointed at a new
            cache - and the model's window is scored by the same fold model.
  METRIC    precision lift over the same stock-month pool (only the day
            differs), paired against the rule by a month bootstrap.

WHICH NAMES COUNT AS FRESH
--------------------------
  1. In the big cache, not in the core cache (by ticker).
  2. Not a near-copy of a core name. A different ticker can be the same
     company - GOOG/GOOGL, BRK-A/BRK-B, a rename like FB/META - and its
     signals would be the core's signals again. Any fresh name whose daily
     returns correlate above 0.95 with a core name over 250+ shared days is
     dropped and listed.
  3. PRIMARY universe only: rows where the stock's 20-day dollar volume is at
     least the 10th percentile of the CORE universe's trailing year (causal,
     monthly). Short-horizon reversal in thin stocks is partly bid-ask bounce;
     this keeps the fresh names comparable in liquidity to the names the
     model was built on. The unfiltered universe is reported as secondary.

PRE-REGISTRATION
----------------
The primary test and its decision rule are written to
thesis_tables_v86/preregistration.txt BEFORE any fresh-name result is
computed. If the settings that define it change after a result was written,
the run says so - and that result is no longer pre-registered.

    PRIMARY   V81 vs low bb_position, liquid fresh names, 2017-2026.
              V81 BEATS the rule if the lower bound of the paired 90%
              interval is above zero (one-sided 5%).

Everything else - V85's chosen combo, the blend, all fresh names, 2008-2026,
the validation/test halves, bull markets, per year - is secondary and labelled
so. V85's combo is the one other model with a claim to test; if its lower
bound is quoted, use the Bonferroni column (k=2).

WHAT THIS TEST CANNOT FIX
-------------------------
  * Survivorship: the big cache holds names that could still be downloaded.
    Dips recover more often among survivors. This inflates BOTH arms' lifts.
    The paired difference is exposed only as far as one arm leans harder into
    stocks that later died - smaller, not zero - which is why it is the claim.
  * One market: fresh names share the core's market factor, so this is
    out-of-universe, not out-of-market.

COST
----
First run: building ~2,000 names takes 15-30 minutes (cached in chunks - a
crash resumes where it stopped). Fold training is on the core only: ~3 minutes
for V81 over 2008-2026, ~5 more for the combo. Later runs read the caches.
"""

import os
import sys
import time
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import measure_v83 as V83
import sweep_options_v85 as V85

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered")

SEED = V85.SEED
OUT_DIR = "thesis_tables_v86"
CACHE_DIR = "oou_cache_v86"
BUILD_VERSION = "v86.1"

VAL_YEARS = V85.VAL_YEARS
TEST_YEARS = V85.TEST_YEARS
HELD_YEARS = V85.HELD_YEARS
OPPONENT = V85.OPPONENT
BASE_FEATS = V85.BASE_FEATS
BASE_MONO = V85.BASE_MONO
SHORT_FEATS = [f for f, _ in V85.SHORT]
MIN_ROWS = V83.MIN_ROWS
BULL = V82.BREADTH_CUTS[1]

DUP_CORR = 0.95
DUP_MIN_OVERLAP = 250
LIQ_PCT = 0.10

KEEP = (["ticker", "date", "label", "r_multiple", "log_dollar_vol"]
        + BASE_FEATS)

COMBO_NAME = "combo top-3: 6 + 7 + 11a"
_stage = dict(V85.STAGE1)
COMBO_CFG, _, _ = V85.apply_deltas(
    [(n, _stage[n]) for n in ("6 classify objective", "7 + short horizon",
                              "11a hyperparams shallow")])
V81_CFG = V85.cfg()


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def cache_names(path):
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(path) if f.endswith(".pkl"))


# =============================================================================
# 1. WHICH NAMES ARE FRESH
# =============================================================================
def _returns(cache, names, start):
    cols = {}
    for t in names:
        df = E64.P2.load_prices(t, cache)
        if df is None or len(df) < 30:
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")]
        c = c[(c.index >= start) & (c > 0)]
        cols[t] = np.log(c).diff().clip(-0.5, 0.5)
    return pd.DataFrame(cols)


def find_duplicates(core_cache, core_names, big_cache, fresh_names,
                    thr=DUP_CORR, min_overlap=DUP_MIN_OVERLAP,
                    start="2004-01-01", block=400):
    """
    Fresh names whose daily returns are near-identical to some core name's:
    share classes, renames, dual listings. Pairwise-complete correlation for
    every fresh x core pair, from masked matrix products.
    """
    cols = ["fresh", "core match", "corr", "shared days"]
    Y = _returns(core_cache, core_names, pd.Timestamp(start))
    X = _returns(big_cache, fresh_names, pd.Timestamp(start))
    if Y.empty or X.empty:
        return pd.DataFrame(columns=cols)
    idx = Y.index.union(X.index)
    ynames, xnames = list(Y.columns), list(X.columns)
    Y = Y.reindex(idx).to_numpy(float)
    X = X.reindex(idx).to_numpy(float)
    My = np.isfinite(Y).astype(float)
    Y0 = np.where(My > 0, Y, 0.0)
    Y2 = Y0 * Y0
    rows = []
    for a in range(0, X.shape[1], block):
        x = X[:, a:a + block]
        Mx = np.isfinite(x).astype(float)
        X0 = np.where(Mx > 0, x, 0.0)
        n = Mx.T @ My
        sx, sy = X0.T @ My, Mx.T @ Y0
        sxx, syy, sxy = (X0 * X0).T @ My, Mx.T @ Y2, X0.T @ Y0
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / n
            vx, vy = sxx - sx * sx / n, syy - sy * sy / n
            r = cov / np.sqrt(vx * vy)
        r = np.where((n >= min_overlap) & np.isfinite(r), r, -np.inf)
        j = r.argmax(axis=1)
        best = r[np.arange(len(j)), j]
        for i in np.flatnonzero(best > thr):
            rows.append({"fresh": xnames[a + i], "core match": ynames[j[i]],
                         "corr": float(best[i]),
                         "shared days": int(n[i, j[i]])})
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# 2. DATA - core exactly as V85 built it; fresh in cached chunks, slim
# =============================================================================
def _short(cache, names):
    try:
        _, sh = V85.price_features(cache, names)
    except ValueError:                              # nothing loadable
        return pd.DataFrame(columns=["ticker", "date_n"] + SHORT_FEATS)
    return sh


def build_core(prov, price_cache, cache_dir=CACHE_DIR, verbose=True):
    names = cache_names(price_cache)
    key = _hash(["core", price_cache, names, prov["horizon"], prov["step"],
                 prov["label_mode"], BUILD_VERSION])
    path = os.path.join(cache_dir, f"core_{key}.pkl")
    if os.path.exists(path):
        if verbose:
            print(f"  core dataset: cached ({path})")
        return pd.read_pickle(path), names
    d = E64.build_dataset(names, price_cache, prov["horizon"], prov["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year
    # V85's exact sequence, so the V81 fold models reproduce
    d = d.sort_values("date").reset_index(drop=True)
    d["row_id"] = np.arange(len(d))
    d["date_n"] = pd.DatetimeIndex(d["date"]).normalize()
    d = d.merge(_short(price_cache, names), on=["ticker", "date_n"],
                how="left")
    d = d.sort_values("row_id").reset_index(drop=True)
    d = d[KEEP + SHORT_FEATS + ["year", "row_id", "date_n"]]
    d.to_pickle(path)
    return d, names


def _build_slim(cache, names, prov):
    d = E64.build_dataset(names, cache, prov["horizon"], prov["step"],
                          verbose=False)
    if d.empty:
        return pd.DataFrame(columns=KEEP + SHORT_FEATS)
    d = E64.apply_label_mode(d, prov["label_mode"])
    d = d[KEEP].copy()          # xs_/mkt_ columns were computed per chunk:
    d["date_n"] = pd.DatetimeIndex(d["date"]).normalize()  # dropped on purpose
    d = d.merge(_short(cache, names), on=["ticker", "date_n"], how="left")
    return d.drop(columns="date_n")


def build_fresh(big_cache, names, prov, chunk=250, cache_dir=CACHE_DIR,
                verbose=True):
    chunks = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    parts, t0, built = [], time.time(), 0
    for k, ch in enumerate(chunks, 1):
        key = _hash(["fresh", big_cache, ch, prov["horizon"], prov["step"],
                     prov["label_mode"], BUILD_VERSION])
        path = os.path.join(cache_dir, f"fresh_{key}.pkl")
        if os.path.exists(path):
            part, how = pd.read_pickle(path), "cached"
        else:
            t = time.time()
            part = _build_slim(big_cache, ch, prov)
            part.to_pickle(path)
            built += 1
            how = f"{time.time() - t:4.0f}s"
        parts.append(part)
        if verbose:
            left = len(chunks) - k
            eta = ((time.time() - t0) / max(built, 1)) * left if built else 0
            print(f"    chunk {k}/{len(chunks)}  {len(ch)} names  "
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


def liquidity_floor(core, dates, pct=LIQ_PCT, min_rows=MIN_ROWS):
    """Per row: the pct-quantile of the CORE universe's log dollar volume over
    the trailing window, refit monthly. Causal - uses only earlier months."""
    win = V82._win()
    di = pd.DatetimeIndex(core["date"])
    v = core["log_dollar_vol"].to_numpy(float)
    o = np.argsort(di.values, kind="mergesort")
    ds, vs = di.values[o], v[o]
    codes, uniq = pd.factorize(pd.PeriodIndex(pd.DatetimeIndex(dates),
                                              freq="M"))
    out = np.full(len(dates), np.nan)
    for k, m in enumerate(uniq):
        t = m.to_timestamp()
        lo = np.searchsorted(ds, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(ds, t.to_datetime64(), "left")
        w = vs[lo:hi]
        w = w[np.isfinite(w)]
        if w.size >= min_rows:
            out[codes == k] = float(np.quantile(w, pct))
    return out


# =============================================================================
# 3. FOLD MODELS - trained on the core only, scoring both universes
# =============================================================================
def _fit_predict(c, tr, sets, seed, start):
    live = [i for i, X in enumerate(sets) if len(X)]
    got = V85.fit_predict(c, tr, [sets[i] for i in live], seed, start)
    out = [np.array([], float) for _ in sets]
    for i, p in zip(live, got):
        out[i] = np.asarray(p, float)
    return out


def walk_forward_oou(c, core, fresh, years, n_seeds, horizon, label, sig,
                     cache_dir=CACHE_DIR, resume=True, verbose=True):
    """
    Exactly V85.walk_forward on the core, plus: the same fold models score
    the fresh names' test year and trailing window. Returns
    {"core": (te, refs), "fresh": (te, refs)} in V85's format. `sig`
    identifies the two datasets, so a changed cache is never read stale.
    """
    key = _hash([c, n_seeds, years, SEED, sig, BUILD_VERSION])
    path = os.path.join(cache_dir, f"wf_{key}.pkl")
    if resume and os.path.exists(path):
        if verbose:
            print(f"    {label:<34} cached")
        return pd.read_pickle(path)
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    win = V82._win()
    dc = pd.DatetimeIndex(core["date"])
    dfr = pd.DatetimeIndex(fresh["date"])
    base_seed = SEED + int(c["seed_offset"])
    acc = {"core": ([], {}), "fresh": ([], {})}
    t0 = time.time()
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        tr = core[dc < start - embargo]
        te = core[core["year"] == y]
        ref = core[(dc >= start - win) & (dc < start)]
        if len(tr) < 2000 or len(te) < 200:
            continue
        fte = fresh[fresh["year"] == y]
        fref = fresh[(dfr >= start - win) & (dfr < start)]
        sets = [te, ref, fte, fref]
        P = [[] for _ in sets]
        for s in range(n_seeds):
            for i, p in enumerate(_fit_predict(c, tr, sets, base_seed + s,
                                               start)):
                P[i].append(p)
        P = [np.mean(p, axis=0) if len(p[0]) else np.array([], float)
             for p in P]
        for tag, a, b, pa, pb in (("core", te, ref, P[0], P[1]),
                                  ("fresh", fte, fref, P[2], P[3])):
            if not len(a):
                continue
            acc[tag][0].append(pd.DataFrame(
                {"row_id": a["row_id"].to_numpy(), "p": pa}))
            acc[tag][1][y] = pd.DataFrame({
                "row_id": np.r_[b["row_id"].to_numpy(), a["row_id"].to_numpy()],
                "date": np.r_[b["date"].to_numpy(), a["date"].to_numpy()],
                "p": np.r_[pb, pa]})
        if verbose:
            print(f"      {label} {y}: train {len(tr):,} core rows, scored "
                  f"{len(te):,} core + {len(fte):,} fresh  "
                  f"[{time.time() - t0:4.0f}s]")
    out = {k: (pd.concat(v[0], ignore_index=True) if v[0]
               else pd.DataFrame(columns=["row_id", "p"]), v[1])
           for k, v in acc.items()}
    pd.to_pickle(out, path)
    return out


def restrict(te, refs, pos):
    """Keep only rows inside a universe and renumber them to its positions."""
    def m(df):
        p = pos[df["row_id"].to_numpy(int)]
        k = p >= 0
        o = df[k].copy()
        o["row_id"] = p[k]
        return o.reset_index(drop=True)
    return m(te), {y: m(r) for y, r in refs.items()}


# =============================================================================
# 4. ARMS AND TABLES
# =============================================================================
def arms_for(U, wf, n_random, seed=SEED):
    """Every arm on one universe frame U (RangeIndex, row_id == position)."""
    def rule(score):
        score = np.asarray(score, float)
        return V83.apply_rule(score, V83.rolling_cut(U["date"], score),
                              strict=True)
    fires = {}
    te, refs = wf["V81"]
    fires["V81"] = V85.model_fire(U, te, refs)
    if "combo" in wf:
        fires[COMBO_NAME] = V85.model_fire(U, *wf["combo"])
    good_bb = 1.0 - V82.trailing_pct(U, "bb_position")
    fires["blend V81 + bb_position"] = V85.blend_fire(U, refs, good_bb)
    fires[OPPONENT] = rule(-U["bb_position"].to_numpy(float))
    fires["low rsi_14"] = rule(-U["rsi_14"].to_numpy(float))
    fires["low range60_position"] = rule(-U["range60_position"]
                                         .to_numpy(float))
    fires["composite v2"] = rule(V82.composite_v2(
        U, list(zip(BASE_FEATS, BASE_MONO))))
    rng = np.random.default_rng(seed)
    for k in range(n_random):
        fires[f"random #{k + 1}"] = rule(rng.random(len(U)))
    return fires


def concentration(U, f, years):
    t = U["ticker"].to_numpy()[f & np.isin(U["year"].to_numpy(), years)]
    if not t.size:
        return 0, np.nan
    vc = pd.Series(t).value_counts()
    return int(vc.size), float(vc.head(10).sum() / t.size)


def arm_table(U, fires, ev, all_years, k_bonf):
    opp = fires[OPPONENT]
    rows = []
    for a, f in fires.items():
        row = {"Arm": a}
        s = ev.summary(f, HELD_YEARS)
        if s:
            row.update({"Signals": s["signals"], "Lift pp": s["lift"],
                        "Lift lo": s["lo"], "Lift hi": s["hi"],
                        "ExpR": s["expR"], "Cond yrs": s["cond"],
                        "Worst yr": s["worst"], "Max/yr": s["max_year"]})
        n, top = concentration(U, f, HELD_YEARS)
        row.update({"Names": n, "Top-10 share": top})
        if a != OPPONENT:
            for tag, yrs in (("", HELD_YEARS), ("val ", VAL_YEARS),
                             ("test ", TEST_YEARS), ("all-yrs ", all_years)):
                p = ev.paired(f, opp, yrs, k_tests=k_bonf)
                if p:
                    row[f"{tag}vs bb"] = p["diff"]
                    row[f"{tag}vs bb lo"] = p["lo"]
                    row[f"{tag}vs bb hi"] = p["hi"]
                    if tag == "":
                        row["vs bb lo (Bonf)"] = p["lo_bonf"]
                        row["P(diff<=0)"] = p["p_le0"]
        sa = ev.summary(f, all_years)
        if sa:
            row["All-yrs lift"] = sa["lift"]
            row["All-yrs signals"] = sa["signals"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_year(U, fires, ev, years, arms):
    yr = U["year"].to_numpy()
    rows = []
    for y in years:
        r = {"Year": y}
        for a in arms:
            if a not in fires:
                continue
            m = fires[a] & (yr == y)
            r[f"{a} n"] = int(m.sum())
            r[f"{a} lift"] = ev.ex_w[m].mean() * 100 if m.sum() else np.nan
        if "V81 lift" in r and f"{OPPONENT} lift" in r:
            r["V81 - bb"] = r["V81 lift"] - r[f"{OPPONENT} lift"]
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# 5. PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, settings, fresh_names, n_core, verbose=True):
    sig = _hash([settings, fresh_names])
    meta_p = os.path.join(out_dir, "preregistration.json")
    txt_p = os.path.join(out_dir, "preregistration.txt")
    seen_result = os.path.exists(os.path.join(out_dir, "verdict.txt"))
    status = "fresh"
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            meta = json.load(fh)
        if meta.get("sig") == sig:
            if verbose:
                print(f"  pre-registered {meta['written']} - settings "
                      f"unchanged since")
            ok = meta.get("status") != "changed-after-result"
            return meta["written"], ok
        status = "changed-after-result" if seen_result else "changed"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = settings
    replaces = ("" if status == "fresh"
                else " (REPLACES an earlier pre-registration)")
    names_sig = hashlib.sha256(",".join(fresh_names).encode()).hexdigest()[:16]
    text = f"""PRE-REGISTRATION - V86 out-of-universe test
written {now}, before any fresh-name result was computed{replaces}

MODEL      V81 as served: expectancy XGBoost on {', '.join(s['features'])},
           monotone low-is-good, {s['n_seeds']} seeds from {SEED}. Fold models are
           retrained year by year on the {n_core} CORE names only. No row from a
           fresh name is used to train, choose a feature, a direction or the rule.
OPPONENT   {OPPONENT} (fixed in V85, before this test).
UNIVERSE   names in {s['big_cache']} that are not in {s['core_cache']}, minus any
           whose daily returns correlate above {s['dup_corr']} with a core name over
           {s['dup_overlap']}+ shared days; {len(fresh_names)} names remain.
           PRIMARY keeps a row only when the stock's 20-day log dollar volume is at
           least the {s['liq_pct'] * 100:.0f}th percentile of the core universe's
           trailing 252 trading days (causal, refit monthly).
RULE       each arm fires when its score is STRICTLY above the {(1 - s['quantile']) * 100:.0f}th
           percentile of the universe's own trailing-252-day scores, refit monthly.
           The model's window is scored by the same fold model.
METRIC     win-rate lift over the same stock-month pool, in pp; the paired
           difference V81 - {OPPONENT}; 90% interval from a calendar-month
           bootstrap with weights shared by both arms ({s['n_boot']} draws).
YEARS      {HELD_YEARS[0]}-{HELD_YEARS[-1]}.
DECISION   V81 BEATS the rule if the lower 90% bound of the paired difference
           is above zero (one-sided 5%). WORSE if the upper bound is below zero.
           Otherwise: AHEAD BUT NOT PROVEN if the point estimate is above zero,
           else DOES NOT BEAT.
SECONDARY  (not decisions) V85's chosen combo [{COMBO_NAME}] - quote its
           Bonferroni k=2 bound if claimed; the V81+bb blend; all fresh names
           without the liquidity filter; all years; validation/test halves;
           bull markets; per year; the core names side by side.
NAMES      sha256 of the fresh list: {names_sig}
"""
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(txt_p):
        with open(txt_p) as fh, \
                open(os.path.join(out_dir, "preregistration_history.txt"),
                     "a") as hist:
            hist.write(fh.read() + "\n" + "-" * 80 + "\n")
    with open(txt_p, "w") as fh:
        fh.write(text)
    with open(meta_p, "w") as fh:
        json.dump({"sig": sig, "written": now, "status": status}, fh)
    if verbose:
        print(f"  wrote {txt_p} ({now})")
        if status == "changed-after-result":
            print("  *** the settings that define the primary test changed "
                  "AFTER a result was written.\n      This run's primary "
                  "result is NOT pre-registered - report it as exploratory.")
        elif status == "changed":
            print("  (settings changed before any result was written - the new"
                  " pre-registration stands)")
    return now, status != "changed-after-result"


# =============================================================================
# RUNNER
# =============================================================================
def run_v86(model_in="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, n_boot=4000,
            with_combo=True, first_year=None, n_random=3, chunk=250,
            liq_pct=LIQ_PCT, dup_corr=DUP_CORR, resume=True,
            out_dir=OUT_DIR, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    m81 = M81.load_model(model_in)
    prov = m81["provenance"]
    core_cache = core_cache or prov["price_cache"]
    t0 = time.time()
    W = 104
    print("=" * W)
    print("V86 - OUT-OF-UNIVERSE TEST: names the model has never seen")
    print("=" * W)
    if not os.path.isdir(big_cache):
        raise RuntimeError(f"{big_cache} not found - point RUN_BIG_CACHE at "
                           f"the expanded price cache")

    # ---- [1] names -----------------------------------------------------------
    core_set = set(cache_names(core_cache))
    big = cache_names(big_cache)
    cand = [t for t in big if t not in core_set]
    print(f"\n  [1] NAMES  core {len(core_set)} ({core_cache}) | big cache "
          f"{len(big)} ({big_cache}) | not in core by ticker: {len(cand)}")
    dkey = _hash(["dups", core_cache, sorted(core_set), big_cache, cand,
                  dup_corr, DUP_MIN_OVERLAP])
    dpath = os.path.join(CACHE_DIR, f"dups_{dkey}.pkl")
    if resume and os.path.exists(dpath):
        dups = pd.read_pickle(dpath)
    else:
        print("      checking every candidate against every core name for "
              "share classes / renames ...")
        dups = find_duplicates(core_cache, sorted(core_set), big_cache, cand,
                               thr=dup_corr)
        dups.to_pickle(dpath)
    dups.to_csv(f"{out_dir}/duplicates.csv", index=False)
    drop = set(dups["fresh"])
    fresh_names = [t for t in cand if t not in drop]
    print(f"      near-copies of a core name dropped: {len(drop)}"
          + (f"  e.g. " + ", ".join(f"{r['fresh']}~{r['core match']} "
                                     f"({r['corr']:.3f})"
                                     for _, r in dups.head(6).iterrows())
             if len(drop) else ""))
    print(f"      fresh names: {len(fresh_names)}")

    # ---- [2] data -------------------------------------------------------------
    print(f"\n  [2] DATA")
    core, core_names = build_core(prov, core_cache, verbose=verbose)
    print(f"      core: {len(core):,} candidate entries")
    print(f"      fresh: building {len(fresh_names)} names in chunks of "
          f"{chunk} (cached) ...")
    fresh = build_fresh(big_cache, fresh_names, prov, chunk=chunk,
                        verbose=verbose)
    fresh["liq_floor"] = liquidity_floor(core, fresh["date"], liq_pct)
    fresh["liquid"] = (fresh["log_dollar_vol"].to_numpy(float)
                       >= fresh["liq_floor"].to_numpy(float))
    b, _ = V82.full_breadth(core_cache, core_names)
    core["breadth"] = core["date_n"].map(b).to_numpy(float)
    fresh["breadth"] = fresh["date_n"].map(b).to_numpy(float)
    print(f"      fresh: {len(fresh):,} candidate entries from "
          f"{fresh['ticker'].nunique()} names; liquid rows "
          f"{fresh['liquid'].mean():.0%}")
    hy = lambda fr: fr[np.isin(fr["year"].to_numpy(), HELD_YEARS)]
    dv = lambda s: np.expm1(np.nanmedian(s.to_numpy(float))) / 1e6
    print(f"      median 20-day dollar volume, {HELD_YEARS[0]}-"
          f"{HELD_YEARS[-1]}: core ${dv(hy(core)['log_dollar_vol']):,.1f}M"
          f" | fresh ${dv(hy(fresh)['log_dollar_vol']):,.1f}M | fresh liquid "
          f"${dv(hy(fresh[fresh['liquid']])['log_dollar_vol']):,.1f}M")

    years_all = sorted(core["year"].unique())
    fold_years = [y for y in years_all[int(prov["min_train_years"]):]
                  if first_year is None or y >= first_year]
    all_years = fold_years

    # ---- [3] pre-registration (before any fresh result) --------------------
    print(f"\n  [3] PRE-REGISTRATION")
    settings = {"features": list(m81["features"]), "n_seeds": n_seeds,
                "big_cache": big_cache, "core_cache": core_cache,
                "dup_corr": dup_corr, "dup_overlap": DUP_MIN_OVERLAP,
                "liq_pct": liq_pct, "quantile": V83.QUANTILE,
                "years": HELD_YEARS, "opponent": OPPONENT,
                "n_boot": n_boot}
    prereg_when, prereg_ok = preregister(out_dir, settings, fresh_names,
                                         len(core_set))

    # ---- [4] fold models (core only) -------------------------------------
    print(f"\n  [4] FOLD MODELS - trained on the core only, "
          f"{fold_years[0]}-{fold_years[-1]}, {n_seeds} seeds")
    sig = _hash([len(core), core_names, len(fresh), fresh_names,
                 str(fresh["date"].max())])
    wf = {"V81": walk_forward_oou(V81_CFG, core, fresh, fold_years, n_seeds,
                                  prov["horizon"], "V81", sig, resume=resume,
                                  verbose=verbose)}
    if with_combo:
        wf["combo"] = walk_forward_oou(COMBO_CFG, core, fresh, fold_years,
                                       n_seeds, prov["horizon"], "combo", sig,
                                       resume=resume, verbose=verbose)

    # ---- universes ---------------------------------------------------------------
    def universe(frame, mask, tag):
        pos = np.full(len(frame), -1, int)
        pos[mask] = np.arange(int(mask.sum()))
        U = frame[mask].reset_index(drop=True).copy()
        U["row_id"] = np.arange(len(U))
        w = {k: restrict(*v[tag], pos) for k, v in wf.items()}
        return U, w

    universes = {}
    CORE = f"core {len(core_set)}"
    universes[CORE] = universe(core, np.ones(len(core), bool), "core")
    universes["fresh, liquid (PRIMARY)"] = universe(
        fresh, fresh["liquid"].to_numpy(bool), "fresh")
    universes["fresh, all"] = universe(fresh, np.ones(len(fresh), bool),
                                       "fresh")
    PRIMARY = "fresh, liquid (PRIMARY)"

    # ---- coverage -------------------------------------------------------------
    cov = []
    for name, (U, _) in universes.items():
        for y in all_years:
            m = U["year"].to_numpy() == y
            cov.append({"Universe": name, "Year": y, "Rows": int(m.sum()),
                        "Names": int(U.loc[m, "ticker"].nunique())})
    cov = pd.DataFrame(cov)
    cov.to_csv(f"{out_dir}/coverage.csv", index=False)
    piv = cov.pivot(index="Year", columns="Universe", values="Names")
    print(f"\n      names with a candidate entry, per year")
    print("      " + piv[list(universes)].to_string().replace("\n", "\n      "))

    # ---- [5] arms ----------------------------------------------------------------
    print(f"\n  [5] ARMS on each universe ...")
    results, evs, fires_by = {}, {}, {}
    K = 2 if with_combo else 1
    for name, (U, w) in universes.items():
        f = arms_for(U, w, n_random)
        ev = V85.Evaluator(U, n_boot=n_boot)
        fires_by[name], evs[name] = f, ev
        T = arm_table(U, f, ev, all_years, K)
        T.insert(0, "Universe", name)
        results[name] = T
        print(f"      {name:<26} {len(U):>10,} rows  "
              f"{sum(int(v.sum()) for v in f.values()):>8,} fires across "
              f"{len(f)} arms")
    allT = pd.concat(results.values(), ignore_index=True)
    allT.to_csv(f"{out_dir}/arms_all_universes.csv", index=False)

    # ---- [6] integrity -----------------------------------------------------------
    print(f"\n  [6] INTEGRITY - the core fold models must be the ones V83/V85 "
          f"measured")
    Uc, _ = universes[CORE]
    fc = fires_by[CORE]
    yc = Uc["year"].to_numpy()
    checks = []

    def ref_count(csv, arm, col, rule=None):
        if not os.path.exists(csv):
            return None
        t = pd.read_csv(csv)
        t = t[t["Arm"] == arm]
        if rule is not None and "Rule" in t.columns:
            t = t[t["Rule"] == rule]
        return int(t[col].iloc[0]) if len(t) and pd.notna(t[col].iloc[0]) \
            else None
    checks.append(("V81 2017-2026", int((fc["V81"] & np.isin(yc, HELD_YEARS))
                                        .sum()),
                   ref_count("thesis_tables_v83/arms_held-out.csv", "V81",
                             "Signals", ">")))
    if first_year is None:
        checks.append(("V81 all years", int((fc["V81"]
                                             & np.isin(yc, all_years)).sum()),
                       ref_count("thesis_tables_v83/arms_all.csv", "V81",
                                 "Signals", ">")))
    if with_combo:
        checks.append((COMBO_NAME, int((fc[COMBO_NAME]
                                        & np.isin(yc, HELD_YEARS)).sum()),
                       ref_count("thesis_tables_v85/all_arms.csv", COMBO_NAME,
                                 "held signals")))
    integrity_ok = True
    for lab, now, ref in checks:
        if ref is None:
            verdict = "(no reference table found)"
        elif now == ref:
            verdict = "REPRODUCED"
        else:
            verdict = "*** DIFFERENT ***"
            integrity_ok = False
        print(f"      {lab:<28} fires {now:>6,}   recorded "
              f"{ref if ref is not None else '-':>6}   {verdict}")
    if not integrity_ok:
        print("      the core models differ from the ones measured before - "
              "check RUN_MODEL_IN, the core cache and RUN_N_SEEDS")

    # ---- [7] degeneracy on fresh rows -------------------------------------------
    Up, wp = universes[PRIMARY]
    te_p = wp["V81"][0]
    fr = Up.loc[te_p["row_id"].to_numpy(int), ["year"]].reset_index(drop=True)
    deg = V83.degeneracy(fr, te_p["p"].to_numpy(float))
    deg.to_csv(f"{out_dir}/degeneracy_fresh.csv", index=False)
    bad = deg[deg["Share >= own p99"] > 0.03]
    print(f"\n  [7] PLATEAU CHECK on fresh liquid rows: share >= own p99 is "
          f"{deg['Share >= own p99'].min():.3f}-"
          f"{deg['Share >= own p99'].max():.3f}"
          + (f" - PLATEAU in {', '.join(map(str, bad['Year']))}" if len(bad)
             else " in every year (healthy ~0.010)"))

    # ---- [8] every arm, every universe ----------------------------------------
    fmt = lambda v: f"{v:+.2f}"
    show = ["Arm", "Signals", "Names", "Top-10 share", "Lift pp", "Lift lo",
            "Lift hi", "ExpR", "Cond yrs", "Worst yr", "vs bb", "vs bb lo",
            "vs bb hi", "val vs bb", "test vs bb", "All-yrs lift",
            "all-yrs vs bb"]
    print("\n" + "=" * W)
    print(f"  [8] EVERY ARM, {HELD_YEARS[0]}-{HELD_YEARS[-1]} - lift = win "
          f"rate above the same stock-month pool (pp); 'vs bb' = paired "
          f"difference")
    print("=" * W)
    for name, T in results.items():
        print(f"\n  {name.upper()}")
        t = T[[c for c in show if c in T.columns]].copy()
        if "Top-10 share" in t:
            t["Top-10 share"] = t["Top-10 share"].map(
                lambda v: f"{v:.0%}" if pd.notna(v) else "")
        print(t.to_string(index=False, float_format=fmt))

    # ---- [9] side by side -----------------------------------------------------
    side = []
    for name, T in results.items():
        g = T.set_index("Arm")
        for arm in ["V81", COMBO_NAME] if with_combo else ["V81"]:
            if arm not in g.index:
                continue
            r = g.loc[arm]
            side.append({"Universe": name, "Arm": arm,
                         "Signals": r.get("Signals"),
                         "Arm lift": r.get("Lift pp"),
                         "bb lift": g.loc[OPPONENT].get("Lift pp"),
                         "vs bb": r.get("vs bb"), "lo": r.get("vs bb lo"),
                         "hi": r.get("vs bb hi"),
                         "lo (Bonf k=2)": r.get("vs bb lo (Bonf)")
                         if arm == COMBO_NAME else np.nan,
                         "P(diff<=0)": r.get("P(diff<=0)")})
    side = pd.DataFrame(side)
    side.to_csv(f"{out_dir}/side_by_side.csv", index=False)
    print("\n" + "=" * W)
    print(f"  [9] SIDE BY SIDE - the model against {OPPONENT}, "
          f"{HELD_YEARS[0]}-{HELD_YEARS[-1]}")
    print("=" * W)
    print(side.to_string(index=False, float_format=fmt))

    # ---- [10] normal and bull markets ------------------------------------------
    reg = []
    lo_c, hi_c = V82.BREADTH_CUTS
    for name, (U, _) in universes.items():
        f, ev = fires_by[name], evs[name]
        br = U["breadth"].to_numpy(float)
        for rname, mask in ((f"bull (> {hi_c:.0%})", br > hi_c),
                            (f"not bear (>= {lo_c:.0%})", br >= lo_c)):
            for tag, yrs in (("2017-2026", HELD_YEARS),
                             ("all years", all_years)):
                for arm in ["V81", COMBO_NAME] if with_combo else ["V81"]:
                    p = ev.paired(f[arm] & mask, f[OPPONENT] & mask, yrs)
                    s = ev.summary(f[arm] & mask, yrs)
                    sb = ev.summary(f[OPPONENT] & mask, yrs)
                    if p and s and sb:
                        reg.append({"Universe": name, "Regime": rname,
                                    "Years": tag, "Arm": arm,
                                    "Signals": s["signals"],
                                    "Arm lift": s["lift"],
                                    "bb lift": sb["lift"], "vs bb": p["diff"],
                                    "lo": p["lo"], "hi": p["hi"]})
    reg = pd.DataFrame(reg)
    reg.to_csv(f"{out_dir}/regimes.csv", index=False)
    print(f"\n  [10] NORMAL AND BULL MARKETS - signals on dates whose core "
          f"breadth is in the regime, both arms")
    if len(reg):
        print(reg.to_string(index=False, float_format=fmt))
    else:
        print("      fewer than 30 signals per arm in every regime cell")

    # ---- [11] per year on the primary universe -------------------------------
    py = per_year(Up, fires_by[PRIMARY], evs[PRIMARY], all_years,
                  ["V81", OPPONENT] + ([COMBO_NAME] if with_combo else []))
    py.to_csv(f"{out_dir}/per_year_primary.csv", index=False)
    print(f"\n  [11] PER YEAR - {PRIMARY} (point estimates; lift in pp)")
    print(py.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    held = py[py["Year"].isin(HELD_YEARS)].dropna(subset=["V81 - bb"])
    if len(held):
        print(f"      V81 ahead of the rule in {(held['V81 - bb'] > 0).sum()}"
              f"/{len(held)} held-out years")

    # ---- [12] the verdict --------------------------------------------------------
    g = results[PRIMARY].set_index("Arm")
    r = g.loc["V81"] if "V81" in g.index else None
    print("\n" + "=" * W)
    print(f"  [12] THE PRE-REGISTERED VERDICT - V81 vs {OPPONENT}, "
          f"{PRIMARY}, {HELD_YEARS[0]}-{HELD_YEARS[-1]}")
    print("=" * W)
    lines = []
    if r is None or pd.isna(r.get("vs bb")):
        lines.append("  not enough signals to compare")
    else:
        d_, lo, hi = r["vs bb"], r["vs bb lo"], r["vs bb hi"]
        lines.append(f"  V81 {r['Lift pp']:+.2f} pp over its pools on "
                     f"{int(r['Signals']):,} signals; {OPPONENT} "
                     f"{g.loc[OPPONENT, 'Lift pp']:+.2f} pp on "
                     f"{int(g.loc[OPPONENT, 'Signals']):,}")
        lines.append(f"  paired difference {d_:+.2f} pp, 90% [{lo:+.2f}, "
                     f"{hi:+.2f}], share of bootstrap draws <= 0: "
                     f"{r['P(diff<=0)']:.3f}")
        if lo > 0:
            v = (f"V81 BEATS the simple rule on names it has never seen: "
                 f"{d_:+.2f} pp with the whole 90% interval above zero.")
        elif d_ > 0:
            v = (f"V81 is AHEAD BUT NOT PROVEN: {d_:+.2f} pp, interval "
                 f"includes zero. Report it as matching the rule.")
        elif hi >= 0:
            v = (f"V81 DOES NOT BEAT the rule out of universe ({d_:+.2f} pp,"
                 f" interval includes zero). Report it as matching the rule.")
        else:
            v = (f"V81 is WORSE than the rule out of universe ({d_:+.2f} pp, "
                 f"whole interval below zero). The rule generalises better.")
        lines.append(f"\n  VERDICT: {v}")
        if not prereg_ok:
            lines.append("  (the primary settings changed after an earlier "
                         "result - this verdict is exploratory)")
        if not integrity_ok:
            lines.append("  (integrity check failed - the core models are not "
                         "the measured ones)")
        rc = results[CORE].set_index("Arm")
        if "V81" in rc.index and pd.notna(rc.loc["V81"].get("vs bb")):
            c_ = rc.loc["V81"]
            lines.append(f"\n  for reference, the same comparison on the "
                         f"{CORE}: {c_['vs bb']:+.2f} pp [{c_['vs bb lo']:+.2f}, "
                         f"{c_['vs bb hi']:+.2f}] on {int(c_['Signals']):,} "
                         f"signals")
            lines.append(f"  interval width: core {c_['vs bb hi'] - c_['vs bb lo']:.2f}"
                         f" pp -> fresh {hi - lo:.2f} pp")
        if with_combo and COMBO_NAME in g.index and \
                pd.notna(g.loc[COMBO_NAME].get("vs bb")):
            cb = g.loc[COMBO_NAME]
            lines.append(f"\n  secondary - V85's chosen combo vs "
                         f"{OPPONENT}: {cb['vs bb']:+.2f} pp, 90% "
                         f"[{cb['vs bb lo']:+.2f}, {cb['vs bb hi']:+.2f}], "
                         f"Bonferroni k=2 lower bound "
                         f"{cb['vs bb lo (Bonf)']:+.2f}")
        nb = reg[(reg["Universe"] == PRIMARY) & (reg["Arm"] == "V81")
                 & (reg["Years"] == "2017-2026")
                 & reg["Regime"].str.startswith("not bear")] if len(reg) \
            else reg
        if len(nb):
            q = nb.iloc[0]
            lines.append(f"  secondary - normal and bull markets only (what "
                         f"the strategy is for): V81 vs {OPPONENT} "
                         f"{q['vs bb']:+.2f} pp [{q['lo']:+.2f}, "
                         f"{q['hi']:+.2f}] on {int(q['Signals']):,} signals")
        rnd = [a for a in g.index if a.startswith("random")]
        if rnd:
            lines.append(f"  sanity - random arms' lift on fresh liquid: "
                         + ", ".join(f"{g.loc[a, 'Lift pp']:+.2f}"
                                     for a in rnd)
                         + " pp (should sit near zero)")
    lines.append(f"\n  pre-registered {prereg_when}; tables in {out_dir}/")
    print("\n".join(lines))
    with open(os.path.join(out_dir, "verdict.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  total {(time.time() - t0) / 60:.1f} min")
    return results


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_IN    = "entry_model_v81.joblib"
    RUN_CORE_CACHE  = None              # None = the cache V81 was trained on
    RUN_BIG_CACHE   = "price_cache_v68" # the expanded cache (~2,347 names)
    RUN_N_SEEDS     = 5                 # 5 = V81; needed for the integrity check
    RUN_N_BOOT      = 4000
    RUN_WITH_COMBO  = True              # also test V85's chosen combo (~5 min)
    RUN_FIRST_YEAR  = None              # None = 2008-2026; 2017 = held-out only
    RUN_N_RANDOM    = 3                 # random arms, a sanity floor
    RUN_CHUNK       = 250               # names per build chunk (memory)
    RUN_RESUME      = True              # reuse cached chunks and fold models
    RUN_OUT_DIR     = "thesis_tables_v86"
    # -------------------------------------------------------------------------

    run_v86(model_in=RUN_MODEL_IN, core_cache=RUN_CORE_CACHE,
            big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS, n_boot=RUN_N_BOOT,
            with_combo=RUN_WITH_COMBO, first_year=RUN_FIRST_YEAR,
            n_random=RUN_N_RANDOM, chunk=RUN_CHUNK, resume=RUN_RESUME,
            out_dir=RUN_OUT_DIR)
