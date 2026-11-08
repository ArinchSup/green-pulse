"""
V98 - DOES A FUNDAMENTAL FILTER PICK BETTER STOCKS? (pre-registered)

THE QUESTION
------------
V89 showed V88 picks a better DAY (+10 pp) but fires in stock-months that are
worse than average (-9 pp), which is why its hit rate looks ordinary. Price
trend rules could not fix that (V93, V94). This tests whether financial
statements can: written down before any result is seen.

  1. SELECTION   do stocks the filter keeps go on to hit their targets more
                 often than the stocks it drops, in the same month?
  2. NOT JUST    is that still true among stocks of similar volatility?
     VOLATILITY  (Pillar 2 already uses volatility; weak companies are often
                 volatile)
  3. FRIEND      does it beat your friend's rules - at the same strictness,
                 and as the pipeline uses them (health score <= 4 vetoes)?
  4. WITH V88    applied to V88's ENTER signals, does the hit rate rise?

THE FILTER - fixed from published research, not from our data
-------------------------------------------------------------
Six published signals, each turned into a percentile within the month among
non-financial stocks, in the direction the papers report:

  gross profitability  higher is better   Novy-Marx (2013)
  accruals             lower is better    Sloan (1996)
  asset growth         lower is better    Cooper, Gulen & Schill (2008)
  share growth         lower is better    Pontiff & Woodgate (2008)
  F-score (partial)    higher is better   Piotroski (2000)
  Altman Z''           higher is better   Altman (1995); distressed firms
                                          earn less - Dichev (1998)

  composite = average of the available percentiles (at least 4 of 6)
  FILTER    = drop the month's bottom 30% by composite, keep the other 70%

The signal for month M+1 uses only filings public by the end of month M
(V97's point-in-time panel).

THE OUTCOME
-----------
The same 60-day target / stop / expiry label as V88, on every candidate day
(every 5th trading day) in the month after the signal date. So "kept minus
dropped" is directly the selection effect V89 measured.

DATA
----
  PRIMARY    unseen liquid non-financial stocks, 2012-2026 (V86's test set;
             fundamentals have never been tested on any stock)
  SECONDARY  each half (2012-18, 2019-26); the core stocks; each signal on
             its own; 6-month returns - all descriptive

PASS RULES (primary data; 90% intervals from a calendar-month bootstrap)
  1. kept minus dropped, target hit rate: lower bound > 0
  2. kept minus dropped, average R: lower bound > 0
  3. rule 1 inside volatility thirds (average of the three): estimate > 0
  CONFIRMED only if 1-3 all hold.
  Reported, not required:
  4. friend, same strictness: ours minus his (his average score, bottom 30%
     dropped), paired by month
  5. friend as used: his veto (health <= 4) vs ours dropping the same share
  6. V88's ENTER signals: kept vs skipped, and the table to present
"""

import hashlib
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import entry_filters_v93 as V93
import entry_filters_v94 as V94
import fundamentals_audit_v96 as F96
import fundamental_signals_v97 as F97

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = "thesis_tables_v98"
YEARS = list(range(2012, 2027))
HALVES = {"2012-2018": list(range(2012, 2019)),
          "2019-2026": list(range(2019, 2027))}
DROP = 0.30
MIN_PARTS = 4
VETO = 4.0                       # pipeline's FUND_HEALTH_MIN: score <= 4 vetoes
SEED = 98
COMPONENTS = [("gp_assets", +1), ("accruals", -1), ("asset_growth", -1),
              ("share_growth", -1), ("fscore_partial", +1), ("altman_z", +1)]
ALONE = [("gp_assets", +1), ("roa", +1), ("op_assets", +1), ("accruals", -1),
         ("asset_growth", -1), ("share_growth", -1), ("net_issuance", -1),
         ("fscore", +1), ("fscore_partial", +1), ("altman_z", +1),
         ("runway_q", +1), ("int_cover", +1), ("fr_profit", +1),
         ("fr_growth", +1), ("fr_health", +1), ("fr_avg", +1)]
NAME_F = "F fundamental filter (drop bottom 30%)"
NAME_H = "H friend's average (drop bottom 30%)"
NAME_V = "V friend as used (health <= 4 vetoed)"

PREREG = """PRE-REGISTRATION - V98 fundamental filter
written {now}, before any V98 result was computed{replaces}.

FILTER    composite = mean of within-month percentiles (non-financial stocks,
          same universe) of: gross profitability (+), accruals (-), asset
          growth (-), share growth (-), F-score partial (+), Altman Z'' (+);
          at least {mp} of 6 present. Drop the bottom {drop:.0%} each month.
          Signals for month M+1 from V97 {ver} as of the end of month M.
OUTCOME   V88's 60-day target/stop label on every candidate day.
PRIMARY   unseen liquid non-financial stocks, {y0}-{y1}.
RULES     1. kept - dropped, target hit rate: 90% lower bound > 0
          2. kept - dropped, average R: 90% lower bound > 0
          3. rule 1 within volatility thirds (60-day volatility at the
             signal date), averaged: estimate > 0
          CONFIRMED only if 1-3 hold.
REPORTED  4. ours - friend's (his average score, bottom {drop:.0%} dropped)
          5. friend's veto (health <= {veto:g}) vs ours dropping the same share
          6. V88 ENTER signals kept vs skipped
METHOD    calendar-month bootstrap, {nb} draws, comparisons paired by month.
"""


# =============================================================================
# 1. SCORES ON V97'S PANEL
# =============================================================================
def add_scores(P):
    """Within-month percentiles per universe group (non-financial only)."""
    P = P.copy()
    nf = ~P["financial"].astype(bool)
    keys = [P["group"], P["month"]]
    for c, s in ALONE:
        v = P[c].where(nf)
        P[f"pct_{c}"] = v.groupby(keys).rank(pct=True, ascending=s > 0)
    parts = np.vstack([P[f"pct_{c}"].to_numpy(float) for c, _ in COMPONENTS])
    n = np.isfinite(parts).sum(axis=0)
    P["composite"] = np.where(n >= MIN_PARTS, np.nanmean(parts, axis=0),
                              np.nan)
    P["comp_pct"] = P["composite"].where(nf).groupby(keys).rank(pct=True)
    P["fr_pct"] = P["pct_fr_avg"]
    return P


def price_monthly(names, cache):
    """60-day volatility and the next 6-month return, at each month-end."""
    rows = []
    for t in names:
        df = E64.P2.load_prices(t, cache)
        if df is None or len(df) < 70:
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")].sort_index()
        c = c[c > 0]
        vol = np.log(c).diff().rolling(60, min_periods=50).std() * np.sqrt(252)
        try:
            me = c.resample("ME").last()
            mv = vol.resample("ME").last()
        except (ValueError, TypeError):
            me = c.resample("M").last()
            mv = vol.resample("M").last()
        rows.append(pd.DataFrame({"ticker": t, "month": me.index,
                                  "vol60": mv.reindex(me.index).to_numpy(),
                                  "fwd6": (me.shift(-6) / me - 1).to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=["ticker", "month", "vol60", "fwd6"])


def attach(U, P, group):
    """For each candidate day: the scores as of the previous month-end."""
    prev = (pd.DatetimeIndex(U["date"]).to_period("M") - 1) \
        .to_timestamp(how="end").normalize()
    cols = ["ticker", "month", "financial", "comp_pct", "fr_pct", "fr_health",
            "vt"] + [f"pct_{c}" for c, _ in ALONE]
    Pg = P[P["group"] == group][cols]
    key = pd.DataFrame({"ticker": U["ticker"].to_numpy(), "month": prev})
    J = key.merge(Pg, on=["ticker", "month"], how="left")
    J["financial"] = J["financial"].fillna(False).astype(bool)
    return J


# =============================================================================
# 2. MONTH BOOTSTRAP WITH ACCESS TO THE DRAWS
# =============================================================================
class Boot:
    def __init__(self, dates, n_boot, seed=SEED):
        per = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
        self.mcode, self.months = pd.factorize(per)
        self.myear = np.array([m.year for m in self.months])
        self.year = pd.DatetimeIndex(dates).year.to_numpy()
        self.n_boot, self.rng, self._W = n_boot, np.random.default_rng(seed), {}

    def _w(self, years):
        k = tuple(years)
        if k not in self._W:
            idx = np.flatnonzero(np.isin(self.myear, years))
            W = self.rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)),
                                     size=self.n_boot).astype(float)
            self._W[k] = (idx, W)
        return self._W[k]

    def mean(self, x, m, years):
        idx, W = self._w(years)
        mm = m & np.isin(self.year, years) & np.isfinite(x)
        K = len(self.months)
        s = np.bincount(self.mcode[mm], x[mm], K)[idx]
        c = np.bincount(self.mcode[mm], minlength=K)[idx].astype(float)
        if c.sum() < 1:
            return np.nan, np.full(self.n_boot, np.nan), 0
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (W @ s) / (W @ c)
        return s.sum() / c.sum(), draws, int(c.sum())

    def diff(self, x, ma, mb, years):
        pa, da, na = self.mean(x, ma, years)
        pb, db, nb = self.mean(x, mb, years)
        return pa - pb, da - db, na, nb


def ci(draws):
    d = draws[np.isfinite(draws)]
    return (np.percentile(d, 5), np.percentile(d, 95)) if d.size else \
        (np.nan, np.nan)


def fmt(p, draws, scale=1.0, dec=1):
    lo, hi = ci(draws * scale)
    if not np.isfinite(p):
        return "-"
    return f"{p * scale:+.{dec}f} [{lo:+.{dec}f}, {hi:+.{dec}f}]"


def vol_neutral(B, x, kept, dropped, vt, years):
    pts, dws = [], []
    for t in range(3):
        p, d, na, nb = B.diff(x, kept & (vt == t), dropped & (vt == t), years)
        if na and nb:
            pts.append(p)
            dws.append(d)
    if not pts:
        return np.nan, np.full(B.n_boot, np.nan)
    return float(np.mean(pts)), np.nanmean(np.vstack(dws), axis=0)


# =============================================================================
# 3. PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, n_boot):
    settings = {"components": COMPONENTS, "drop": DROP, "min": MIN_PARTS,
                "years": YEARS, "veto": VETO, "n_boot": n_boot,
                "panel": F97.VERSION}
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
    txt = PREREG.format(now=now, replaces="" if status == "fresh"
                        else " (REPLACES an earlier pre-registration)",
                        mp=MIN_PARTS, drop=DROP, ver=F97.VERSION, y0=YEARS[0],
                        y1=YEARS[-1], veto=VETO, nb=n_boot)
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
# 4. RUNNER
# =============================================================================
def banner(t, W):
    print("\n" + "-" * W + f"\n  {t}\n" + "-" * W)


def outcome_row(label, ind, m, years, yr):
    mm = m & np.isin(yr, years)
    n = int(mm.sum())
    if not n:
        return {"": label, "candidate days": 0}
    return {"": label, "candidate days": f"{n:,}",
            "Target %": f"{ind['tgt'][mm].mean() * 100:.1f}",
            "Stop %": f"{ind['stp'][mm].mean() * 100:.1f}",
            "Expired %": f"{ind['exp'][mm].mean() * 100:.1f}",
            "Avg R": f"{ind['R'][mm].mean():+.3f}"}


def run_v98(model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
            out_dir=OUT_DIR, verbose=False):
    t0 = time.time()
    W = 118
    print("=" * W)
    print("V98 - DOES A FUNDAMENTAL FILTER PICK BETTER STOCKS? (pre-registered; "
          "unseen stocks 2012-2026)")
    print("=" * W)
    panel_p = os.path.join(F96.FUND_CACHE, f"signals_{F97.VERSION}.pkl")
    if not os.path.exists(panel_p):
        raise SystemExit(f"{panel_p} not found - run V97 ({F97.VERSION}) first.")
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]

    print("  reading V86's data and walk-forward models (cached) ...")
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    print(f"  reading V97's signal panel ({F97.VERSION}) and scoring it ...")
    P = pd.read_pickle(panel_p)
    P["month"] = pd.DatetimeIndex(P["month"]).normalize()
    P["financial"] = P["financial"].fillna(False).astype(bool)
    P = add_scores(P)
    print("  60-day volatility and 6-month returns from the prices ...")
    PM = pd.concat([price_monthly(sorted(P.loc[P["group"] == g, "ticker"]
                                         .unique()), c)
                    for g, c in (("core", core_cache), ("unseen", big_cache))],
                   ignore_index=True)
    PM["month"] = pd.DatetimeIndex(PM["month"]).normalize()
    P = P.merge(PM, on=["ticker", "month"], how="left")
    ok = ~P["financial"] & P["comp_pct"].notna() & P["vol60"].notna()
    P["vt"] = np.nan
    P.loc[ok, "vt"] = P[ok].groupby(["group", "month"])["vol60"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 3, labels=False)
        if len(v) >= 3 else np.nan)

    unis = {}
    for name, frame, mask, tag, grp in (
            ("unseen, liquid", fresh, fresh["liquid"].to_numpy(bool), "fresh",
             "unseen"),
            (f"core {n_core}", core, np.ones(len(core), bool), "core",
             "core")):
        U, w = V87.universe(frame, mask, wf, tag)
        fires = V85.model_fire(U, *w["combo"])
        J = attach(U, P, grp)
        unis[name] = (U, fires, J)
    UNSEEN, CORE = "unseen, liquid", f"core {n_core}"

    print("\n  PRE-REGISTRATION")
    when, prereg_ok = preregister(out_dir, n_boot)

    # ---- [1] coverage ----------------------------------------------------------
    U, fires, J = unis[UNSEEN]
    yr = U["year"].to_numpy()
    inyrs = np.isin(yr, YEARS)
    nonfin = ~J["financial"].to_numpy()
    has = np.isfinite(J["comp_pct"].to_numpy())
    banner("[1] WHO THE FILTER CAN JUDGE - unseen liquid candidate days, "
           f"{YEARS[0]}-{YEARS[-1]}", W)
    n_all = int(inyrs.sum())
    print(f"  {n_all:,} candidate days; banks/insurers/REITs "
          f"{(inyrs & ~nonfin).sum() / n_all:.0%}; non-financial with a "
          f"composite {(inyrs & nonfin & has).sum() / n_all:.0%} (the test); "
          f"no data {(inyrs & nonfin & ~has).sum() / n_all:.0%}")

    # ---- [2] primary --------------------------------------------------------------
    ind = V93.outcome_indicators(U)
    B = Boot(U["date"], n_boot)
    cp = J["comp_pct"].to_numpy(float)
    base = nonfin & has
    kept, dropped = base & (cp > DROP), base & (cp <= DROP)
    vt = J["vt"].to_numpy(float)
    pt, dt, nk, nd = B.diff(ind["tgt"], kept, dropped, YEARS)
    pr, dr, _, _ = B.diff(ind["R"], kept, dropped, YEARS)
    pv, dv = vol_neutral(B, ind["tgt"], kept, dropped, vt, YEARS)
    banner(f"[2] PRIMARY - stocks the filter keeps vs drops, same months "
           f"(unseen liquid, {YEARS[0]}-{YEARS[-1]})", W)
    T = pd.DataFrame([outcome_row("kept (top 70%)", ind, kept, YEARS, yr),
                      outcome_row("dropped (bottom 30%)", ind, dropped, YEARS,
                                  yr)])
    print(T.to_string(index=False))
    print(f"\n  kept minus dropped, target hit rate : {fmt(pt, dt, 100)} pp")
    print(f"  kept minus dropped, average R       : {fmt(pr, dr, 1, 3)}")
    print(f"  same, inside volatility thirds      : {fmt(pv, dv, 100)} pp")
    rows = []
    for q in range(1, 6):
        mq = base & (np.ceil(cp * 5) == q)
        mm = mq & inyrs
        rows.append({"composite fifth": f"Q{q}" + (" (worst)" if q == 1 else
                                                   " (best)" if q == 5 else ""),
                     "candidate days": f"{int(mm.sum()):,}",
                     "Target %": f"{ind['tgt'][mm].mean() * 100:.1f}",
                     "Avg R": f"{ind['R'][mm].mean():+.3f}"})
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "primary_fifths.csv"),
                              index=False)

    # ---- [3] each half, each signal ------------------------------------------------
    banner("[3] SECONDARY - each half, and each signal on its own (drop its "
           "bottom 30%) - descriptive", W)
    for lab, yrs in HALVES.items():
        p1, d1, _, _ = B.diff(ind["tgt"], kept, dropped, yrs)
        p2, d2, _, _ = B.diff(ind["R"], kept, dropped, yrs)
        print(f"  {lab}: target {fmt(p1, d1, 100)} pp | R {fmt(p2, d2, 1, 3)}")
    rows = []
    for c, s in ALONE:
        pc = J[f"pct_{c}"].to_numpy(float)
        b2 = nonfin & np.isfinite(pc)
        p1, d1, n1, n2 = B.diff(ind["tgt"], b2 & (pc > DROP), b2 & (pc <= DROP),
                                YEARS)
        p2, d2, _, _ = B.diff(ind["R"], b2 & (pc > DROP), b2 & (pc <= DROP),
                              YEARS)
        rows.append({"signal": c, "direction": "higher" if s > 0 else "lower",
                     "days judged": f"{n1 + n2:,}",
                     "kept - dropped, target pp": fmt(p1, d1, 100),
                     "kept - dropped, R": fmt(p2, d2, 1, 3)})
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(out_dir, "each_signal.csv"), index=False)
    print()
    print(A.to_string(index=False))

    # ---- [4] friend ------------------------------------------------------------------
    banner("[4] YOUR FRIEND'S RULES - same strictness, and as the pipeline "
           "uses them", W)
    fp = J["fr_pct"].to_numpy(float)
    both = base & np.isfinite(fp)
    pk, dk, _, _ = B.diff(ind["tgt"], both & (cp > DROP), both & (cp <= DROP),
                          YEARS)
    ph, dh, _, _ = B.diff(ind["tgt"], both & (fp > DROP), both & (fp <= DROP),
                          YEARS)
    print(f"  stocks both can judge: ours {fmt(pk, dk, 100)} pp, his "
          f"(average score) {fmt(ph, dh, 100)} pp")
    print(f"  ours minus his, paired by month: {fmt(pk - ph, dk - dh, 100)} pp")
    fh = J["fr_health"].to_numpy(float)
    bv = base & np.isfinite(fh)
    veto = bv & (fh <= VETO)
    share = (veto & inyrs).sum() / max((bv & inyrs).sum(), 1)
    # ours at the same strictness, month by month
    mcode = B.mcode
    v_share = pd.Series(veto.astype(float)).groupby(mcode).sum() / \
        pd.Series(bv.astype(float)).groupby(mcode).sum()
    cut = v_share.reindex(mcode).to_numpy()
    o_drop = bv & (cp <= cut)
    pvh, dvh, _, _ = B.diff(ind["tgt"], bv & ~veto, veto, YEARS)
    pvo, dvo, _, _ = B.diff(ind["tgt"], bv & ~o_drop, o_drop, YEARS)
    print(f"\n  his veto (health <= {VETO:g}) drops {share:.1%} of candidate "
          f"days")
    print(f"  his veto, kept minus vetoed      : {fmt(pvh, dvh, 100)} pp")
    print(f"  ours dropping the same share     : {fmt(pvo, dvo, 100)} pp")
    print(f"  ours minus his veto, paired      : {fmt(pvo - pvh, dvo - dvh, 100)}"
          f" pp")

    # ---- [5] with V88 -------------------------------------------------------------------
    st = V93.Stats(U, n_boot)
    valid = fires & base
    M = {V94.F0: valid, NAME_F: valid & (cp > DROP),
         NAME_H: valid & ~(np.isfinite(fp) & (fp <= DROP)),
         NAME_V: valid & ~(np.isfinite(fh) & (fh <= VETO))}
    Tv = V94.table(st, valid, M, YEARS)
    V94.show(Tv, f"[5] WITH V88 - ENTER signals on stocks the filters can "
                 f"judge (unseen liquid, {YEARS[0]}-{YEARS[-1]})")
    Tv.to_csv(os.path.join(out_dir, "with_v88.csv"), index=False)
    Pp = V94.presentable(st, M, valid, NAME_F, YEARS)
    V94.show_outcomes(Pp, f"[6] THE TABLE TO PRESENT - unseen liquid, "
                          f"{YEARS[0]}-{YEARS[-1]}. Each 'vs' row uses the same "
                          f"stock-months / months as the ENTER row above it.")
    Pp.to_csv(os.path.join(out_dir, "presentable.csv"), index=False)

    # ---- [7] core, 6-month returns ---------------------------------------------------
    banner("[7] SECONDARY - core stocks, and 6-month returns (descriptive)", W)
    Uc, fc, Jc = unis[CORE]
    indc = V93.outcome_indicators(Uc)
    Bc = Boot(Uc["date"], n_boot)
    cpc = Jc["comp_pct"].to_numpy(float)
    bc = ~Jc["financial"].to_numpy() & np.isfinite(cpc)
    p1, d1, _, _ = Bc.diff(indc["tgt"], bc & (cpc > DROP), bc & (cpc <= DROP),
                           YEARS)
    p2, d2, _, _ = Bc.diff(indc["R"], bc & (cpc > DROP), bc & (cpc <= DROP),
                           YEARS)
    print(f"  {CORE}: kept minus dropped, target {fmt(p1, d1, 100)} pp | R "
          f"{fmt(p2, d2, 1, 3)}")
    S6 = P[(P["group"] == "unseen") & ~P["financial"] & P["comp_pct"].notna()
           & P["fwd6"].notna() & P["month"].dt.year.isin(YEARS)].copy()
    S6["ex6"] = S6["fwd6"] - S6.groupby("month")["fwd6"].transform("mean")
    B6 = Boot(S6["month"], n_boot)
    x6 = S6["ex6"].to_numpy(float)
    k6 = (S6["comp_pct"] > DROP).to_numpy()
    p6, d6, _, _ = B6.diff(x6, k6, ~k6, YEARS)
    print(f"  unseen, all non-financial stocks: 6-month return, kept minus "
          f"dropped {fmt(p6, d6, 100)} percentage points (overlapping "
          f"windows - read loosely)")

    # ---- verdict ---------------------------------------------------------------------------
    c1 = ci(dt * 100)[0] > 0
    c2 = ci(dr)[0] > 0
    c3 = np.isfinite(pv) and pv > 0
    pf = lambda c: "PASS" if c else "FAIL"
    lines = [
        f"  1. kept minus dropped, target hit rate: {fmt(pt, dt, 100)} pp  -> "
        f"{pf(c1)}",
        f"  2. kept minus dropped, average R: {fmt(pr, dr, 1, 3)}  -> {pf(c2)}",
        f"  3. inside volatility thirds: {pv * 100:+.1f} pp  -> {pf(c3)}",
        "\n  VERDICT: " + ("CONFIRMED - the filter keeps stocks that do "
                           "better in the same month. Adopt it as the "
                           "stock-level layer in front of V88."
                           if c1 and c2 and c3 else
                           "NOT CONFIRMED - do not adopt the filter."),
        f"  (friend, same strictness: ours minus his {fmt(pk - ph, dk - dh, 100)}"
        f" pp; vs his veto: {fmt(pvo - pvh, dvo - dvh, 100)} pp - reported, not "
        f"required)"]
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
    RUN_OUT_DIR   = "thesis_tables_v98"
    # -------------------------------------------------------------------------

    run_v98(big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK, n_boot=RUN_N_BOOT,
            out_dir=RUN_OUT_DIR)
