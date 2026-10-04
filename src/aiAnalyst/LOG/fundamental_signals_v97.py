"""
V97 - FUNDAMENTAL SIGNALS, POINT-IN-TIME, AT EVERY MONTH-END

WHAT V96 SHOWED, AND WHAT V97 CHANGES
-------------------------------------
V96: 2,360 of our 2,362 stocks are on EDGAR. Assets, net income and
operating cash flow are public for about 80% of them from 2013 (rising to
~88%), with a median filing lag of 37 days. It also found four problems,
which V97 fixes:

  1. Share counts jump at stock splits (385 companies), so a split reads as
     massive dilution. V97 measures share issuance from the weighted-average
     share counts INSIDE ONE FILING instead: every 10-Q compares this quarter
     with the same quarter last year, and every 10-K this year with last
     year, both on the same split basis. No split table is needed.
  2. Interest-expense coverage fell from ~59% to ~26% in 2025-26 - the
     pattern of a tag change. V97 adds the newer tags.
  3. Your friend's rules need short-term debt and cash + short-term
     investments. V97 adds both.
  4. The F-score needs all 9 parts, and only 40-55% of stocks have them all.
     V97 keeps the full score and adds a partial one (7+ parts, rescaled).

WHAT IT DOES - AND WHAT IT DOES NOT
-----------------------------------
For every stock and every month-end from January 2010, it computes each
signal from values that were already public on that day. It reads only the
files V96 downloaded (no network). It does NOT look at prices or returns:
the signal definitions are fixed here, before V98 tests them.

THE SIGNALS (point-in-time; TTM = trailing 12 months)
-----------------------------------------------------
  gp_assets       gross profit / assets                   higher = better
  roa             net income / assets                     higher = better
  op_assets       operating income / assets               higher = better
  accruals        (net income - operating cash flow)      lower = better
                  / average assets
  asset_growth    assets vs one year earlier              lower = better
  share_growth    weighted-average shares vs one year     lower = better
                  earlier (same filing, split-proof)
  net_issuance    (equity issued - buybacks) / assets     lower = better
  fscore          Piotroski F-score, 0-9, all 9 parts     higher = better
  fscore_partial  F-score from 7+ available parts,        higher = better
                  rescaled to 0-9
  altman_z        Altman Z'' (book version): above 2.6    higher = better
                  "safe", below 1.1 "distress"
  runway_q        quarters of cash + short-term           lower = worse
                  investments at the current burn
                  (40 = not burning cash)
  int_cover       operating income / interest expense     higher = better
                  (100 = no debt, no interest)

  Your friend's three rules, rebuilt from the same EDGAR data:
  fr_profit, fr_growth, fr_health, fr_avg     as written: latest annual
                                              report, his formulas
  fr_*_ttm                                    same formulas on the latest
                                              quarterly (TTM) figures

Banks, insurers and REITs are flagged (financial = True); most of these
signals are not meaningful for them and V98 leaves them out.

v97.2 - FIXES FROM THE FIRST RUN'S COMPARISON WITH HIS LIVE CODE
----------------------------------------------------------------
Profit and growth matched within 1 point for 95% of names, health for 79%.
The gaps had three causes, now fixed for his rules (our signals unchanged,
except runway, which uses the same cash figure):
  - his "total debt" comes from Yahoo, which counts lease liabilities (store
    leases make a debt-free retailer look indebted): leases added to HIS debt
  - his cash includes short-term investments; more tags for them added
  - his revenue growth compares years as shown in the LATEST annual report
    (restated after a divestiture); the first run compared each year's own
    first report, so a sale of a business looked like a 20%+ revenue drop.
    His growth now compares years inside the same annual report.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import fundamentals_audit_v96 as F

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = "thesis_tables_v97"
VERSION = "v97.2"
DAY = np.int64(86_400 * 10**9)
MAX_AGE = F.MAX_AGE_DAYS              # 200 days, as in V96
ANNUAL_AGE = 456                      # an annual report stays current ~15 months
RUNWAY_CAP = 40.0
COVER_CAP = 100.0

# ---- items: V96's, plus what V97 needs --------------------------------------
CONCEPTS = dict(F.CONCEPTS)
CONCEPTS["interest_expense"] = ("flow", "USD", F.CONCEPTS["interest_expense"][2]
                                + ["us-gaap:InterestExpenseNonoperating",
                                   "us-gaap:InterestAndDebtExpense",
                                   "us-gaap:InterestExpenseOperating"])
CONCEPTS.update({
    "debt_current": ("instant", "USD", ["us-gaap:DebtCurrent"]),
    "ltd_current": ("instant", "USD", [
        "us-gaap:LongTermDebtCurrent",
        "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent"]),
    "st_borrowings": ("instant", "USD", ["us-gaap:ShortTermBorrowings"]),
    "cash_sti": ("instant", "USD", [
        "us-gaap:CashCashEquivalentsAndShortTermInvestments"]),
    "st_investments": ("instant", "USD", [
        "us-gaap:ShortTermInvestments", "us-gaap:MarketableSecuritiesCurrent",
        "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "us-gaap:AvailableForSaleSecuritiesCurrent",
        "us-gaap:HeldToMaturitySecuritiesCurrent",
        "us-gaap:OtherShortTermInvestments"]),
    # leases - only for your friend's "total debt" (Yahoo counts them)
    "op_lease": ("instant", "USD", ["us-gaap:OperatingLeaseLiability"]),
    "op_lease_cur": ("instant", "USD",
                     ["us-gaap:OperatingLeaseLiabilityCurrent"]),
    "op_lease_noncur": ("instant", "USD",
                        ["us-gaap:OperatingLeaseLiabilityNoncurrent"]),
    "fin_lease": ("instant", "USD", ["us-gaap:FinanceLeaseLiability"]),
    "fin_lease_cur": ("instant", "USD",
                      ["us-gaap:FinanceLeaseLiabilityCurrent"]),
    "fin_lease_noncur": ("instant", "USD",
                         ["us-gaap:FinanceLeaseLiabilityNoncurrent"]),
})
WASO_TAGS = ["us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
             "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"]

MAIN = ["gp_assets", "roa", "op_assets", "accruals", "asset_growth",
        "share_growth", "net_issuance", "fscore", "fscore_partial",
        "altman_z", "runway_q", "int_cover"]
FRIEND = ["fr_profit", "fr_growth", "fr_health", "fr_avg",
          "fr_profit_ttm", "fr_growth_ttm", "fr_health_ttm", "fr_avg_ttm"]
FPARTS = [f"f{i}" for i in range(1, 10)]


def _tags():
    out = {}
    for _, (_, unit, aliases) in CONCEPTS.items():
        for a in aliases:
            out.setdefault(a, set()).add(unit)
    for a in WASO_TAGS:
        out.setdefault(a, set()).add("shares")
    return out


TAGS = _tags()


# =============================================================================
# 1. RAW ROWS (cached per company, so later versions never re-read the JSON)
# =============================================================================
def raw_from_json(facts):
    cols = {k: [] for k in ("tag", "unit", "start", "end", "val", "filed",
                            "accn")}
    block = (facts or {}).get("facts") or {}
    for qt, units in TAGS.items():
        space, tag = qt.split(":")
        node = (block.get(space) or {}).get(tag)
        if not node:
            continue
        for unit in units:
            for e in (node.get("units") or {}).get(unit, []) or []:
                v, en, fi = e.get("val"), e.get("end"), e.get("filed")
                if v is None or not en or not fi:
                    continue
                cols["tag"].append(qt)
                cols["unit"].append(unit)
                cols["start"].append(e.get("start"))
                cols["end"].append(en)
                cols["val"].append(float(v))
                cols["filed"].append(fi)
                cols["accn"].append(e.get("accn") or "")
    df = pd.DataFrame({"tag": cols["tag"], "unit": cols["unit"],
                       "start": pd.to_datetime(F._dates(cols["start"])),
                       "end": pd.to_datetime(F._dates(cols["end"])),
                       "val": np.array(cols["val"], dtype=float),
                       "filed": pd.to_datetime(F._dates(cols["filed"])),
                       "accn": cols["accn"]})
    return df.dropna(subset=["end", "filed"]).reset_index(drop=True)


def load_raw(sec, cik):
    """Raw rows for one company: from the V97 cache, else from V96's JSON."""
    path = os.path.join(F.FUND_CACHE, f"raw_{VERSION}_{cik}.pkl")
    src = sec.local(f"facts_{cik}.json")
    if os.path.exists(path) and (src is None or
                                 os.path.getmtime(path) >= os.path.getmtime(src)):
        return pd.read_pickle(path)
    if src is None:
        return None
    raw = raw_from_json(sec.read(src))
    raw.to_pickle(path)
    return raw


def concept_rows(raw, aliases, unit):
    sub = raw[(raw["unit"] == unit) & raw["tag"].isin(aliases)]
    pr = {a: i for i, a in enumerate(aliases)}
    return pd.DataFrame({"prio": sub["tag"].map(pr).to_numpy(int),
                         "tag": sub["tag"].to_numpy(),
                         "start": sub["start"].to_numpy(),
                         "end": sub["end"].to_numpy(),
                         "val": sub["val"].to_numpy(float),
                         "filed": sub["filed"].to_numpy()})


def share_growth_series(raw):
    """
    Split-proof share issuance. In each filing, the weighted-average share
    count of its latest period vs the same period one year earlier - both
    figures come from the same filing, so a split is applied to both.
    Basic shares first, diluted as fallback; quarter or year preferred over
    year-to-date. Each period is dated by the first filing that reported it.
    """
    m = ((raw["unit"].to_numpy() == "shares")
         & raw["tag"].isin(WASO_TAGS).to_numpy()
         & raw["start"].notna().to_numpy()
         & (raw["accn"].to_numpy() != ""))
    if not m.any():
        return None
    w = raw[m]
    st = w["start"].to_numpy("datetime64[D]").astype(np.int64)
    en = w["end"].to_numpy("datetime64[D]").astype(np.int64)
    fi = w["filed"].to_numpy("datetime64[D]").astype(np.int64)
    va = w["val"].to_numpy(float)
    pr = w["tag"].map({t: i for i, t in enumerate(WASO_TAGS)}).to_numpy(int)
    days = en - st
    cls = np.zeros(len(w), int)
    for k, (_, lo, hi) in enumerate(F.DUR_CLASSES, 1):
        cls[(days >= lo) & (days <= hi)] = k
    keep = np.flatnonzero(cls > 0)
    if not keep.size:
        return None
    codes, _ = pd.factorize(w["accn"].to_numpy()[keep])
    order = keep[np.argsort(codes, kind="mergesort")]
    sc = np.sort(codes, kind="mergesort")
    bounds = np.r_[0, np.flatnonzero(np.diff(sc)) + 1, len(sc)]
    rank = {1: 0, 4: 1, 2: 2, 3: 3}                 # quarter, year, then YTD
    out = {}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]
        E = en[idx].max()
        cur = sorted((i for i in idx if en[i] == E),
                     key=lambda i: (pr[i], rank[cls[i]]))
        for i in cur:
            if va[i] <= 0:
                continue
            tgt = en[i] - 365
            cands = [c for c in idx if pr[c] == pr[i] and cls[c] == cls[i]
                     and abs(en[c] - tgt) <= 20 and va[c] > 0]
            if cands:
                c = min(cands, key=lambda c: abs(en[c] - tgt))
                if E not in out or fi[i] < out[E][0]:
                    out[E] = (fi[i], va[i] / va[c] - 1)
                break
    if not out:
        return None
    ends = np.array(sorted(out), dtype=np.int64)
    return pd.DataFrame({
        "end": pd.to_datetime(ends.astype("datetime64[D]")),
        "filed": pd.to_datetime(np.array([out[e][0] for e in ends],
                                         dtype=np.int64).astype("datetime64[D]")),
        "val": np.array([out[e][1] for e in ends], dtype=float)})


def filing_revenue(raw):
    """
    Annual revenue as shown inside each annual report: the latest year and
    the two before it, all from the same filing (restated on the same basis,
    as Yahoo shows them). Returns three series - r0 (latest year), r1, r2 -
    keyed by the report's latest year end, dated by its first filing.
    """
    aliases = CONCEPTS["revenue"][2]
    m = ((raw["unit"].to_numpy() == "USD") & raw["tag"].isin(aliases).to_numpy()
         & raw["start"].notna().to_numpy() & (raw["accn"].to_numpy() != ""))
    if not m.any():
        return None
    r = raw[m]
    days = (r["end"] - r["start"]).dt.days.to_numpy()
    r = r[(days >= 350) & (days <= 380)]
    if r.empty:
        return None
    pr = r["tag"].map({a: i for i, a in enumerate(aliases)}).to_numpy(int)
    en = r["end"].to_numpy("datetime64[D]").astype(np.int64)
    fi = r["filed"].to_numpy("datetime64[D]").astype(np.int64)
    va = r["val"].to_numpy(float)
    acc = r["accn"].to_numpy()
    out = {}
    for a in pd.unique(acc):
        idx = np.flatnonzero(acc == a)
        E = en[idx].max()
        top = idx[en[idx] == E]
        p = pr[top].min()
        rows = idx[pr[idx] == p]
        vals = []
        for k in range(3):
            tgt = E - 365 * k
            c = rows[np.abs(en[rows] - tgt) <= 20]
            vals.append(va[c[np.argmin(np.abs(en[c] - tgt))]] if c.size
                        else np.nan)
        f0 = fi[top].min()
        if E not in out or f0 < out[E][0]:
            out[E] = (f0, *vals)
    ends = np.array(sorted(out), dtype=np.int64)
    base = {"end": pd.to_datetime(ends.astype("datetime64[D]")),
            "filed": pd.to_datetime(np.array([out[e][0] for e in ends],
                                             dtype=np.int64)
                                    .astype("datetime64[D]"))}
    return {f"rev_f{k}": pd.DataFrame({**base, "val": np.array(
        [out[e][1 + k] for e in ends], dtype=float)}) for k in range(3)}


def company_series(raw):
    """Every concept as a point-in-time series (instant, or TTM for flows)."""
    S = {}
    for name, (kind, unit, aliases) in CONCEPTS.items():
        rows = concept_rows(raw, aliases, unit)
        if rows.empty:
            continue
        if kind == "instant":
            s = F.instant_series(rows)
            if len(s):
                S[name] = s[["end", "filed", "val"]].reset_index(drop=True)
        else:
            _, T, _ = F.flow_series(rows)
            if len(T):
                S[name] = T[["end", "filed", "val", "src"]]
    sg = share_growth_series(raw)
    if sg is not None:
        S["share_growth"] = sg
    fr = filing_revenue(raw)
    if fr is not None:
        S.update(fr)
    return S


# =============================================================================
# 2. FAST AS-OF LOOKUP (same rules as V96.asof, vectorised)
# =============================================================================
def asof_fast(series, d, max_age=MAX_AGE, lags=(0,)):
    """
    d: int64 nanoseconds. For each date, the latest period already filed (if
    it ended at most max_age days earlier); for lag L, the value for the
    period L years before that one (+/- V96's tolerance), also already filed.
    """
    out = {L: np.full(len(d), np.nan) for L in lags}
    if series is None or not len(series):
        return out
    e = series["end"].to_numpy("datetime64[ns]").astype(np.int64)
    f = series["filed"].to_numpy("datetime64[ns]").astype(np.int64)
    v = series["val"].to_numpy(float)
    o = np.lexsort((e, f))
    ef, ff, vf = e[o], f[o], v[o]
    cm = np.maximum.accumulate(ef)
    best = np.maximum.accumulate(np.where(ef >= cm, np.arange(len(o)), -1))
    k = np.searchsorted(ff, d, side="right") - 1
    j = np.flatnonzero(k >= 0)
    if not j.size:
        return out
    r = best[k[j]]
    E = ef[r]
    ok = (d[j] - E) <= max_age * DAY
    j, r, E = j[ok], r[ok], E[ok]
    if 0 in out:
        out[0][j] = vf[r]
    if any(L > 0 for L in lags):
        oe = np.argsort(e, kind="mergesort")
        es, fs, vs = e[oe], f[oe], v[oe]
        n, tol, dj = len(es), F.YEAR_AGO_TOL * DAY, d[j]
        for L in lags:
            if L == 0:
                continue
            tgt = E - np.int64(365 * L) * DAY
            pos = np.searchsorted(es, tgt)
            lo, hi = np.clip(pos - 1, 0, n - 1), np.clip(pos, 0, n - 1)
            g_lo, g_hi = np.abs(es[lo] - tgt), np.abs(es[hi] - tgt)
            ok_lo = (g_lo <= tol) & (fs[lo] <= dj)
            ok_hi = (g_hi <= tol) & (fs[hi] <= dj)
            pick_lo = ok_lo & (~ok_hi | (g_lo <= g_hi))
            pick_hi = ok_hi & ~pick_lo
            val = np.full(len(j), np.nan)
            val[pick_lo] = vs[lo[pick_lo]]
            val[pick_hi] = vs[hi[pick_hi]]
            out[L][j] = val
    return out


def fiscal_year_ends(S):
    ends = []
    for name in ("revenue", "net_income", "cfo", "operating_income"):
        s = S.get(name)
        if s is not None and "src" in s:
            ends.append(s.loc[s["src"] == "annual", "end"]
                        .to_numpy("datetime64[D]").astype(np.int64))
    return np.unique(np.concatenate(ends)) if ends else np.array([], np.int64)


def annual_only(s, fy):
    """Annual figures: flows from the annual report; balance-sheet values at a
    fiscal year end (+/- 7 days)."""
    if s is None or not len(s):
        return None
    if "src" in s:
        return s[s["src"] == "annual"]
    if not len(fy):
        return None
    e = s["end"].to_numpy("datetime64[D]").astype(np.int64)
    pos = np.searchsorted(fy, e)
    near = np.minimum(np.abs(e - fy[np.clip(pos - 1, 0, len(fy) - 1)]),
                      np.abs(e - fy[np.clip(pos, 0, len(fy) - 1)])) <= 7
    return s[near]


# =============================================================================
# 3. THE SIGNALS
# =============================================================================
def div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(a) & np.isfinite(b) & (b != 0), a / b,
                       np.nan)
    return out


def first_of(*arrs):
    out = np.full(len(arrs[0]), np.nan)
    for a in arrs:
        out = np.where(np.isfinite(out), out, a)
    return out


def zero_if(a, have):
    """Missing -> 0 where the company otherwise reported (have = finite)."""
    return np.where(np.isfinite(a), a, np.where(np.isfinite(have), 0.0, np.nan))


def friend_scores(rev, oi, cash, ca, cl, debt, cfo):
    """
    His three rules, vectorised. rev: list of revenue arrays, newest first
    (index 0 = latest period, 1 = one year earlier, ...).
    """
    r0 = rev[0]
    profit = np.where(r0 > 0, np.clip(5.0 + (5.0 / 0.35) * div(oi, r0), 0, 10),
                      np.nan)
    profit = np.where(np.isfinite(oi) & np.isfinite(r0), profit, np.nan)

    n = len(r0)
    growth = np.full(n, np.nan)
    revs = np.vstack(rev)                                   # (k, n)
    avail = np.isfinite(revs)
    # the run of consecutive available years starting from the latest
    run = np.cumprod(avail, axis=0).astype(bool)
    k_avail = run.sum(axis=0)
    nonpos = ((revs <= 0) & run).any(axis=0)
    ok = (k_avail >= 2) & ~nonpos
    g = np.full((revs.shape[0] - 1, n), np.nan)
    for i in range(revs.shape[0] - 1):
        g[i] = np.where(run[i + 1], div(revs[i], revs[i + 1]) - 1, np.nan)
    score = 4.0 + 20.0 * g[0]
    earlier = g[1:]
    has_e = np.isfinite(earlier).any(axis=0)
    all_pos = np.where(np.isfinite(earlier), earlier > 0, True).all(axis=0)
    all_neg = np.where(np.isfinite(earlier), earlier < 0, True).all(axis=0)
    score = score + np.where(has_e & all_pos, 0.5, 0) \
        - np.where(has_e & all_neg & ~all_pos, 0.5, 0)
    growth[ok] = np.clip(score[ok], 0, 10)

    need = np.isfinite(cash) & np.isfinite(ca) & np.isfinite(cl) & \
        np.isfinite(cfo)
    debt = np.where(np.isfinite(debt), debt, 0.0)
    neg = (cash < 0) | (ca < 0) | (cl < 0) | (debt < 0)
    runway_years = np.where(cfo < 0, div(cash, -cfo), 3.0)
    runway_score = 5.0 * np.minimum(runway_years / 3.0, 1.0)
    cr = np.where(cl > 0, div(ca, cl), 2.0)
    cover_score = 3.0 * np.minimum(cr / 2.0, 1.0)
    debt_score = np.where(cash + debt > 0, 2.0 * div(cash, cash + debt), 2.0)
    health = np.where(need & ~neg, runway_score + cover_score + debt_score,
                      np.nan)
    trio = np.vstack([profit, growth, health])
    cnt = np.isfinite(trio).sum(axis=0)
    with np.errstate(invalid="ignore"):
        avg = np.where(cnt >= 2, np.nanmean(trio, axis=0), np.nan)
    return (np.round(profit, 1), np.round(growth, 1), np.round(health, 1),
            avg)


def leases(get, have):
    """Operating + finance lease liabilities (total, else current +
    non-current); 0 where the company reported a balance sheet but no
    leases - always the case before the 2019 lease standard."""
    tot = 0.0
    for kind in ("op_lease", "fin_lease"):
        parts = zero_if(get(f"{kind}_cur")[0], have) + \
            zero_if(get(f"{kind}_noncur")[0], have)
        tot = tot + zero_if(first_of(get(kind)[0], parts), have)
    return tot


def company_signals(S, d):
    """All signals for one company on the date grid d (int64 ns)."""
    q = lambda name, lags=(0,), age=MAX_AGE, s=None: asof_fast(
        S.get(name) if s is None else s, d, age, lags)
    A = q("assets", (0, 1))
    CA, CL = q("assets_current", (0, 1)), q("liabilities_current", (0, 1))
    LIAB = first_of(q("liabilities")[0],
                    q("liab_and_equity")[0] - q("equity")[0])
    EQ, RE, CASH = q("equity")[0], q("retained_earnings")[0], q("cash")[0]
    CASHLIKE = first_of(q("cash_sti")[0],
                        CASH + zero_if(q("st_investments")[0], CASH))
    LTD = q("lt_debt", (0, 1))
    LTD0, LTD1 = zero_if(LTD[0], A[0]), zero_if(LTD[1], A[1])
    DC = first_of(q("debt_current")[0],
                  zero_if(q("ltd_current")[0], A[0])
                  + zero_if(q("st_borrowings")[0], A[0]))
    DEBT = LTD0 + zero_if(DC, A[0])
    REV = q("revenue", (0, 1, 2, 3))
    COGS = q("cogs", (0, 1))
    GPT = q("gross_profit", (0, 1))
    GP0 = first_of(GPT[0], REV[0] - COGS[0])
    GP1 = first_of(GPT[1], REV[1] - COGS[1])
    OI = q("operating_income")[0]
    NI = q("net_income", (0, 1))
    CFO = q("cfo")[0]
    INT = q("interest_expense")[0]
    ISS = zero_if(q("equity_issued")[0], CFO)
    BUY = zero_if(q("buybacks")[0], CFO)
    SG = q("share_growth")[0]

    o = {}
    o["gp_assets"] = div(GP0, A[0])
    o["roa"] = div(NI[0], A[0])
    o["op_assets"] = div(OI, A[0])
    avgA = np.where(np.isfinite(A[1]), (A[0] + A[1]) / 2, A[0])
    o["accruals"] = div(NI[0] - CFO, avgA)
    o["asset_growth"] = div(A[0], A[1]) - 1
    o["share_growth"] = SG
    o["net_issuance"] = div(ISS - BUY, A[0])

    roa0, roa1 = div(NI[0], A[0]), div(NI[1], A[1])
    lev0, lev1 = div(LTD0, A[0]), div(LTD1, A[1])
    cr0, cr1 = div(CA[0], CL[0]), div(CA[1], CL[1])
    gm0, gm1 = div(GP0, REV[0]), div(GP1, REV[1])
    at0, at1 = div(REV[0], A[0]), div(REV[1], A[1])

    def test(cond, *inputs):
        ok = np.logical_and.reduce([np.isfinite(x) for x in inputs])
        return np.where(ok, cond.astype(float), np.nan)

    parts = [test(roa0 > 0, roa0), test(CFO > 0, CFO),
             test(roa0 > roa1, roa0, roa1), test(CFO > NI[0], CFO, NI[0]),
             test(lev0 < lev1, lev0, lev1), test(cr0 > cr1, cr0, cr1),
             test(ISS <= 0, ISS), test(gm0 > gm1, gm0, gm1),
             test(at0 > at1, at0, at1)]
    P = np.vstack(parts)
    have = np.isfinite(P).sum(axis=0)
    total = np.nansum(P, axis=0)
    o["fscore"] = np.where(have == 9, total, np.nan)
    o["fscore_partial"] = np.where(have >= 7, total / np.maximum(have, 1) * 9,
                                   np.nan)
    o["fscore_parts"] = have.astype(float)
    for i, p in enumerate(parts, 1):
        o[f"f{i}"] = p

    o["altman_z"] = (6.56 * div(CA[0] - CL[0], A[0]) + 3.26 * div(RE, A[0])
                     + 6.72 * div(OI, A[0]) + 1.05 * div(EQ, LIAB))
    burn = np.where(CFO < 0, -CFO / 4.0, np.nan)
    o["runway_q"] = np.where(np.isfinite(CFO) & np.isfinite(CASHLIKE),
                             np.where(CFO < 0,
                                      np.minimum(div(CASHLIKE, burn),
                                                 RUNWAY_CAP), RUNWAY_CAP),
                             np.nan)
    no_int = ~(np.isfinite(INT) & (INT > 0))
    o["int_cover"] = np.where(~no_int, np.clip(div(OI, INT), -COVER_CAP,
                                               COVER_CAP),
                              np.where(np.isfinite(DEBT) & (DEBT <= 0)
                                       & np.isfinite(OI), COVER_CAP, np.nan))
    o["log_assets"] = np.log(np.where(A[0] > 0, A[0], np.nan))

    # ---- friend's rules: as written (annual report) and on TTM -------------
    fy = fiscal_year_ends(S)
    an = lambda name: annual_only(S.get(name), fy)
    qa = lambda name, lags=(0,): asof_fast(an(name), d, ANNUAL_AGE, lags)
    RA = qa("revenue", (0, 1, 2, 3))
    # as shown inside the latest annual report (falls back to first reports)
    # (three years from one report; a fourth year would come from an older
    # report on a different basis, so it is left out when the report exists)
    RF = [asof_fast(S.get(f"rev_f{k}"), d, ANNUAL_AGE)[0] for k in range(3)]
    RA = {0: first_of(RF[0], RA[0]), 1: first_of(RF[1], RA[1]),
          2: first_of(RF[2], RA[2]),
          3: np.where(np.isfinite(RF[0]), np.nan, RA[3])}
    Aa = qa("assets")[0]
    CASHa = first_of(qa("cash_sti")[0],
                     qa("cash")[0] + zero_if(qa("st_investments")[0],
                                             qa("cash")[0]))
    DCa = first_of(qa("debt_current")[0],
                   zero_if(qa("ltd_current")[0], Aa)
                   + zero_if(qa("st_borrowings")[0], Aa))
    DEBTa = zero_if(qa("lt_debt")[0], Aa) + zero_if(DCa, Aa) \
        + leases(qa, Aa)
    fa = friend_scores([RA[0], RA[1], RA[2], RA[3]],
                       qa("operating_income")[0], CASHa,
                       qa("assets_current")[0], qa("liabilities_current")[0],
                       DEBTa, qa("cfo")[0])
    ft = friend_scores([REV[0], REV[1], REV[2], REV[3]], OI, CASHLIKE, CA[0],
                       CL[0], DEBT + leases(q, A[0]), CFO)
    for name, v in zip(FRIEND[:4], fa):
        o[name] = v
    for name, v in zip(FRIEND[4:], ft):
        o[name] = v
    return o


# =============================================================================
# 4. RUNNER AND REPORTS
# =============================================================================
def month_ends(start, end):
    try:
        return pd.date_range(start, end, freq="ME")
    except (ValueError, TypeError):
        return pd.date_range(start, end, freq="M")


def banner(t, W):
    print("\n" + "-" * W + f"\n  {t}\n" + "-" * W)


def coverage_by_year(D, cols):
    nf = D[~D["financial"]]
    rows = []
    for y, g in nf.groupby(nf["month"].dt.year):
        rec = {"year": y}
        for c in cols:
            rec[c] = g[c].notna().mean() * 100
        rows.append(rec)
    return pd.DataFrame(rows).set_index("year")


def avg_spearman(D, cols, months):
    mats = []
    for m in months:
        g = D[(D["month"] == m) & ~D["financial"]][cols]
        if len(g) < 50:
            continue
        mats.append(g.rank().corr(min_periods=30).to_numpy())
    return pd.DataFrame(np.nanmean(mats, axis=0), index=cols, columns=cols) \
        if mats else None


PARITY_ALWAYS = ["TFX", "XNCR", "VRA", "ZUMZ", "SM", "TRN", "VC"]


def friend_parity(D, n, seed=7):
    """His live yfinance code vs our EDGAR rebuild, latest month. The sample
    is fixed by ticker (a stable hash), plus the names that differed in the
    first run, so runs can be compared."""
    try:
        import class_ai_pillar1_funda_helper as FR
    except Exception as e:
        print(f"  skipped - could not import his file ({type(e).__name__}: "
              f"{e})")
        return None
    last = D["month"].max()
    pool = D[(D["month"] == last) & ~D["financial"] & D["fr_avg"].notna()]
    if not len(pool):
        print("  skipped - no recent rows")
        return None
    import hashlib
    key = pool["ticker"].map(
        lambda t: hashlib.md5(f"{seed}{t}".encode()).hexdigest())
    pick = pool.assign(_k=key).sort_values("_k").head(n)
    extra = D[(D["month"] == last) & D["ticker"].isin(PARITY_ALWAYS)
              & ~D["ticker"].isin(pick["ticker"])]
    pick = pd.concat([pick.drop(columns="_k"), extra])
    rows = []
    for r in pick.itertuples(index=False):
        try:
            f = FR.give_me_foundation(r.ticker)
        except Exception:
            continue
        rows.append({"ticker": r.ticker,
                     "his profit": f.get("profitability"),
                     "ours profit": r.fr_profit,
                     "his growth": f.get("growth"), "ours growth": r.fr_growth,
                     "his health": f.get("financial_health"),
                     "ours health": r.fr_health})
    return pd.DataFrame(rows)


def run_v97(core_cache=None, big_cache="price_cache_v68",
            model_in81="entry_model_v81.joblib", edgar_cache="edgar_cache",
            start="2010-01-31", parity_n=30, out_dir=OUT_DIR):
    t0 = time.time()
    W = 118
    os.makedirs(out_dir, exist_ok=True)
    print("=" * W)
    print("V97 - FUNDAMENTAL SIGNALS, POINT-IN-TIME, EVERY MONTH-END (no prices "
          "or returns used)")
    print("=" * W)
    meta_p = os.path.join(F.FUND_CACHE, f"meta_{F.EXTRACT_VERSION}.pkl")
    if not os.path.exists(meta_p):
        raise SystemExit(f"{meta_p} not found - run V96 first.")
    meta = pd.read_pickle(meta_p)
    if core_cache is None:
        import class_ai_entry_model_v81 as M81
        core_cache = M81.load_model(model_in81)["provenance"]["price_cache"]
    print("  reading the price caches (dates only) ...")
    U = F.load_universe(core_cache, big_cache)
    U = U.merge(meta[["ticker", "cik", "financial", "sic"]], on="ticker",
                how="inner")
    U["financial"] = U["financial"].fillna(False).astype(bool)
    grid = month_ends(start, U["last_price"].max())
    d = grid.values.astype("datetime64[ns]").astype(np.int64)
    sec = F.Sec("", edgar_cache, offline=True)

    ciks = sorted(U["cik"].unique())
    print(f"  {len(U):,} stocks ({len(ciks):,} companies), {len(grid)} "
          f"month-ends {grid[0].date()} to {grid[-1].date()}")
    print("  building signals from the cached EDGAR files ...")
    per_cik, t1 = {}, time.time()
    for i, cik in enumerate(ciks, 1):
        raw = load_raw(sec, cik)
        if raw is not None and len(raw):
            per_cik[cik] = company_signals(company_series(raw), d)
        if i % 250 == 0 or i == len(ciks):
            el = time.time() - t1
            print(f"      {i:,}/{len(ciks):,} companies  {el / 60:4.1f} min"
                  + (f"  ~{el / i * (len(ciks) - i) / 60:.0f} min left"
                     if i < len(ciks) else ""))

    parts = []
    for r in U.itertuples(index=False):
        sig = per_cik.get(r.cik)
        alive = (grid >= r.first_price) & (grid <= r.last_price)
        if not alive.any():
            continue
        frame = pd.DataFrame({"ticker": r.ticker, "cik": r.cik,
                              "group": r.group, "financial": r.financial,
                              "sic": r.sic, "month": grid[alive]})
        if sig is not None:
            for k, v in sig.items():
                frame[k] = np.asarray(v, dtype=np.float32)[alive]
        parts.append(frame)
    D = pd.concat(parts, ignore_index=True)
    for c in MAIN + FRIEND + FPARTS + ["fscore_parts", "log_assets"]:
        if c not in D:
            D[c] = np.nan
    D.to_pickle(os.path.join(F.FUND_CACHE, f"signals_{VERSION}.pkl"))

    # ---- [1] coverage ---------------------------------------------------------
    banner("[1] COVERAGE - % of non-financial stock-months with a value", W)
    cov = coverage_by_year(D, MAIN + FRIEND)
    cov.to_csv(os.path.join(out_dir, "coverage_by_year.csv"))
    yrs = [y for y in F.REPORT_YEARS if y in cov.index]
    show = cov.loc[yrs].T.apply(lambda c: c.map(lambda v: f"{v:.0f}"))
    show.columns = [str(c) for c in show.columns]
    print(show.to_string())

    # ---- [2] distributions ----------------------------------------------------
    banner("[2] WHAT THE SIGNALS LOOK LIKE - non-financial stock-months, "
           "2012 onward", W)
    nf = D[~D["financial"] & (D["month"].dt.year >= 2012)]
    rows = []
    for c in MAIN + FRIEND:
        v = nf[c].dropna()
        if not len(v):
            continue
        rows.append({"signal": c, "values": f"{len(v):,}",
                     "p10": v.quantile(0.1), "median": v.median(),
                     "p90": v.quantile(0.9)})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out_dir, "distributions.csv"), index=False)
    for c in ("p10", "median", "p90"):
        T[c] = T[c].map(lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}")
    print(T.to_string(index=False))
    flags = {
        "Altman Z'' below 1.1 (distress zone)": (nf["altman_z"] < 1.1,
                                                 nf["altman_z"]),
        "F-score 0-3 (weak)": (nf["fscore"] <= 3, nf["fscore"]),
        "F-score 8-9 (strong)": (nf["fscore"] >= 8, nf["fscore"]),
        "runway under 4 quarters": (nf["runway_q"] < 4, nf["runway_q"]),
        "shares up more than 10% in a year": (nf["share_growth"] > 0.10,
                                              nf["share_growth"]),
        "your friend's average below 4": (nf["fr_avg"] < 4, nf["fr_avg"])}
    print()
    for lab, (m, base) in flags.items():
        print(f"  {lab:<38} {m.sum() / max(base.notna().sum(), 1):6.1%} of "
              f"stock-months with a value")

    # ---- [3] how they relate ----------------------------------------------------
    banner("[3] HOW THE SIGNALS RELATE - average month-by-month rank "
           "correlation (non-financial; each December)", W)
    keyc = ["gp_assets", "roa", "accruals", "asset_growth", "share_growth",
            "fscore_partial", "altman_z", "runway_q", "fr_avg"]
    decs = [m for m in grid if m.month == 12 and m.year >= 2012]
    Cm = avg_spearman(D, keyc, decs)
    if Cm is not None:
        Cm.to_csv(os.path.join(out_dir, "rank_correlations.csv"))
        print(Cm.apply(lambda c: c.map(lambda v: f"{v:+.2f}")).to_string())

    # ---- [4] friend's rules: our rebuild vs his live code -----------------------
    banner("[4] YOUR FRIEND'S RULES - our EDGAR rebuild vs his live yfinance "
           "code, latest month", W)
    if parity_n:
        Pf = friend_parity(D, parity_n)
        if Pf is not None and len(Pf):
            Pf.to_csv(os.path.join(out_dir, "friend_parity.csv"), index=False)
            print(Pf.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
            for s in ("profit", "growth", "health"):
                a = pd.to_numeric(Pf[f"his {s}"], errors="coerce")
                b = pd.to_numeric(Pf[f"ours {s}"], errors="coerce")
                m = a.notna() & b.notna()
                if m.any():
                    diff = (a[m] - b[m]).abs()
                    print(f"  {s:<7} {int(m.sum())} pairs: median gap "
                          f"{diff.median():.1f} points, within 1 point "
                          f"{(diff <= 1).mean():.0%}")
            print("  (small gaps are expected: Yahoo and EDGAR define some "
                  "lines differently, e.g. total debt and operating income)")
    else:
        print("  skipped (RUN_FRIEND_PARITY_N = 0)")

    # ---- [5] spot check -----------------------------------------------------------
    banner("[5] SPOT CHECK - latest month, a few well-known names", W)
    last = D["month"].max()
    names = ["AAPL", "MSFT", "NVDA", "AMZN", "KO", "XOM", "F", "TSLA", "INTC",
             "PLUG"]
    sp = D[(D["month"] == last) & D["ticker"].isin(names)]
    if len(sp):
        cols = ["ticker", "gp_assets", "accruals", "asset_growth",
                "share_growth", "fscore", "altman_z", "runway_q", "fr_profit",
                "fr_growth", "fr_health"]
        print(sp[cols].to_string(index=False,
                                 float_format=lambda v: f"{v:.2f}"))
    print(f"\n  signal panel for V98: {F.FUND_CACHE}/signals_{VERSION}.pkl "
          f"({len(D):,} stock-months) | tables in {out_dir}/ | "
          f"{(time.time() - t0) / 60:.1f} min")
    return D


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_CORE_CACHE      = None          # None = the core cache V86-V96 used
    RUN_BIG_CACHE       = "price_cache_v68"
    RUN_EDGAR_CACHE     = "edgar_cache" # V96's downloads (read only)
    RUN_START           = "2010-01-31"
    RUN_FRIEND_PARITY_N = 30            # tickers to compare with his live
                                        # code (needs internet); 0 = skip
    RUN_OUT_DIR         = "thesis_tables_v97"
    # -------------------------------------------------------------------------

    run_v97(core_cache=RUN_CORE_CACHE, big_cache=RUN_BIG_CACHE,
            edgar_cache=RUN_EDGAR_CACHE, start=RUN_START,
            parity_n=RUN_FRIEND_PARITY_N, out_dir=RUN_OUT_DIR)
