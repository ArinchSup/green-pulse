"""
V96 - FUNDAMENTALS FROM SEC EDGAR: CAN WE BUILD A POINT-IN-TIME FILTER?

WHY THIS COMES FIRST
--------------------
V89 split V88's edge into two parts: timing (+10 pp, the day it picks) and
selection (-9 pp, the stocks it fires on are having bad months). V93-V94
showed that price-trend rules cannot fix the selection part. Financial
statements are the standard way to tell healthy companies from weak ones, and
SEC EDGAR gives them for free. Before any modelling, this script answers:
which accounting figures does EDGAR give us, for how many of our ~2,400
stocks, from which year - and can every number be dated by the day it became
public?

WHAT IT DOES (no modelling, no returns, no outcomes)
----------------------------------------------------
  1. maps every ticker in the two price caches to its SEC company number (CIK)
  2. downloads each company's XBRL facts and filing profile. Everything is
     cached; a second run makes no requests (files older than
     RUN_REFRESH_DAYS are fetched again so recent filings are included).
  3. extracts about 20 accounting items, point-in-time:
       - every number is dated by the day it was FIRST filed, never by the
         period it covers; later restatements are ignored
       - tag aliases are merged: companies switch tags (revenue moved to
         RevenueFromContractWithCustomer... around 2018), and reading only one
         tag silently loses years
       - income and cash-flow items are rebuilt into quarters and trailing-12-
         month (TTM) totals. 10-Q cash-flow statements are year-to-date, so
         Q2 and Q3 are differenced out of the 6- and 9-month figures, and Q4
         out of the annual one
  4. reports coverage by year: for each item and each planned signal, the
     share of our stocks that had a usable, already-public value
  5. saves the point-in-time table for the next step (V97: the signals)

THE SIGNALS IT CHECKS (taken from published research, not from our data)
------------------------------------------------------------------------
  gross profitability   gross profit / assets             Novy-Marx (2013)
  accruals              (net income - cash flow) / assets Sloan (1996)
  asset growth          1-year growth in total assets     Cooper, Gulen &
                                                          Schill (2008)
  share issuance        1-year growth in shares           Pontiff & Woodgate
                                                          (2008); also our
                                                          V46 dilution result
  F-score               9 pass/fail checks on profit,     Piotroski (2000)
                        leverage, liquidity, efficiency
  Altman Z''            distress score from working       Altman (1995 book
                        capital, retained earnings, EBIT, version)
                        book equity
  cash runway           quarters of cash at the current   Pillar 2's flag
                        burn
  interest coverage     EBIT / interest expense           standard credit ratio

Banks, insurers and REITs (SIC 6000-6799) are counted separately: most of
these signals are not defined for them.

BEFORE THE FIRST RUN
--------------------
  SEC asks every program to identify itself. Put your name and email in
  RUN_SEC_CONTACT below (it is sent only to sec.gov, in the User-Agent).
  About 2 requests per company at under 8 a second: roughly 30-60 minutes the
  first time, then minutes. New downloads are stored gzipped.
"""

import gzip
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

OUT_DIR = "thesis_tables_v96"
FUND_CACHE = "fund_cache_v96"
EXTRACT_VERSION = "v96.1"

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# ---- the accounting items ----------------------------------------------------
# name: (kind, unit, aliases in priority order). "instant" = balance-sheet
# figure at a date; "flow" = amount over a period (rebuilt into quarters/TTM).
CONCEPTS = {
    "assets": ("instant", "USD", ["us-gaap:Assets"]),
    "assets_current": ("instant", "USD", ["us-gaap:AssetsCurrent"]),
    "liabilities": ("instant", "USD", ["us-gaap:Liabilities"]),
    "liabilities_current": ("instant", "USD", ["us-gaap:LiabilitiesCurrent"]),
    "liab_and_equity": ("instant", "USD",
                        ["us-gaap:LiabilitiesAndStockholdersEquity"]),
    "equity": ("instant", "USD", [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "retained_earnings": ("instant", "USD",
                          ["us-gaap:RetainedEarningsAccumulatedDeficit"]),
    "cash": ("instant", "USD", [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "us-gaap:Cash"]),
    "lt_debt": ("instant", "USD", [
        "us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt",
        "us-gaap:LongTermDebtAndCapitalLeaseObligations"]),
    "shares": ("instant", "shares", [
        "us-gaap:CommonStockSharesOutstanding",
        "dei:EntityCommonStockSharesOutstanding"]),
    "revenue": ("flow", "USD", [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet",
        "us-gaap:SalesRevenueServicesNet"]),
    "cogs": ("flow", "USD", [
        "us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold", "us-gaap:CostOfServices",
        "us-gaap:CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"]),
    "gross_profit": ("flow", "USD", ["us-gaap:GrossProfit"]),
    "operating_income": ("flow", "USD", ["us-gaap:OperatingIncomeLoss"]),
    "net_income": ("flow", "USD", [
        "us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"]),
    "cfo": ("flow", "USD", [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
    "interest_expense": ("flow", "USD", ["us-gaap:InterestExpense",
                                         "us-gaap:InterestExpenseDebt"]),
    "equity_issued": ("flow", "USD", [
        "us-gaap:ProceedsFromIssuanceOfCommonStock",
        "us-gaap:ProceedsFromIssuanceOrSaleOfEquity"]),
    "buybacks": ("flow", "USD", ["us-gaap:PaymentsForRepurchaseOfCommonStock"]),
}

# Duration classes for flow facts, in days. Wide enough for 52/53-week years
# and for retailers whose first quarter is 16 weeks.
DUR_CLASSES = (("Q", 80, 115), ("H", 170, 200), ("N", 255, 290),
               ("FY", 350, 380))
CHAIN = {"Q": 1, "H": 2, "N": 3, "FY": 4}

MAX_AGE_DAYS = 200        # a value whose period ended longer ago is stale
YEAR_AGO_TOL = 45         # "one year earlier" = 365 days +/- this
FIN_SIC = (6000, 6799)    # banks, insurers, REITs, other financials

# A signal is computable when every input is there, now (0) and/or one year
# earlier (1). Each requirement is a list of alternatives; any one will do.
GP = [[("gross_profit", 0)], [("revenue", 0), ("cogs", 0)]]
GP1 = [[("gross_profit", 1)], [("revenue", 1), ("cogs", 1)]]
LIAB = [[("liabilities", 0)], [("liab_and_equity", 0), ("equity", 0)]]
ISSUE = [[("equity_issued", 0)], [("shares", 0), ("shares", 1)]]
FCORE = [[("net_income", 0), ("net_income", 1), ("cfo", 0), ("assets", 0),
          ("assets", 1), ("assets_current", 0), ("assets_current", 1),
          ("liabilities_current", 0), ("liabilities_current", 1),
          ("revenue", 0), ("revenue", 1)]]
SIGNALS = {
    "gross profitability": [GP, [[("assets", 0)]]],
    "accruals": [[[("net_income", 0), ("cfo", 0), ("assets", 0)]]],
    "asset growth": [[[("assets", 0), ("assets", 1)]]],
    "share issuance": [[[("shares", 0), ("shares", 1)]]],
    "F-score (all 9 parts)": [FCORE, GP, GP1, ISSUE,
                              [[("lt_debt", 0), ("lt_debt", 1)]]],
    "F-score (debt missing = 0)": [FCORE, GP, GP1, ISSUE],
    "Altman Z'' (book)": [
        [[("assets_current", 0), ("liabilities_current", 0),
          ("retained_earnings", 0), ("operating_income", 0), ("equity", 0),
          ("assets", 0)]], LIAB],
    "cash runway": [[[("cash", 0), ("cfo", 0)]]],
    "interest coverage": [[[("operating_income", 0),
                            ("interest_expense", 0)]]],
}
REPORT_YEARS = [2009, 2010, 2011, 2012, 2013, 2015, 2017, 2019, 2021, 2023,
                2025, 2026]


# =============================================================================
# 1. DOWNLOADS (cache-first, polite)
# =============================================================================
class Sec:
    def __init__(self, contact, cache_dir, offline=False, refresh_days=45,
                 pause=0.13):
        self.contact, self.dir = (contact or "").strip(), cache_dir
        self.offline, self.refresh_days, self.pause = offline, refresh_days, \
            pause
        self.n_fetched, self.errors = 0, Counter()
        os.makedirs(cache_dir, exist_ok=True)

    def local(self, name):
        """The newest cached copy of `name` (.json or .json.gz), or None."""
        p = os.path.join(self.dir, name)
        have = [x for x in (p, p + ".gz") if os.path.exists(x)]
        return max(have, key=os.path.getmtime) if have else None

    def fresh(self, path):
        if path is None:
            return False
        if self.offline or not self.refresh_days:
            return True
        return time.time() - os.path.getmtime(path) < self.refresh_days * 86400

    @staticmethod
    def read(path):
        try:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get(self, url, name):
        p = self.local(name)
        if p and self.fresh(p):
            return self.read(p)
        if self.offline:
            return self.read(p) if p else None
        if "@" not in self.contact:
            raise SystemExit("Put your name and email in RUN_SEC_CONTACT - SEC "
                             "requires a contact in the User-Agent.")
        import requests
        headers = {"User-Agent": self.contact,
                   "Accept-Encoding": "gzip, deflate"}
        for attempt in range(4):
            try:
                time.sleep(self.pause)
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code == 404:
                    self.errors["not on EDGAR (404)"] += 1
                    return self.read(p) if p else None
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(3 + 5 * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                out = os.path.join(self.dir, name + ".gz")
                with gzip.open(out, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
                self.n_fetched += 1
                return data
            except Exception as e:
                if attempt == 3:
                    self.errors[type(e).__name__] += 1
                    return self.read(p) if p else None
                time.sleep(2 + 2 * attempt)
        return self.read(p) if p else None


# =============================================================================
# 2. UNIVERSE AND CIKS
# =============================================================================
def cache_names(path):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(path)
                  if f.endswith(".pkl"))


def price_spans(names, cache):
    """First and last price date per ticker (normalised, tz-naive)."""
    out = {}
    for t in names:
        try:
            df = pd.read_pickle(os.path.join(cache, f"{t}.pkl"))
        except Exception:
            continue
        if df is None or not len(df):
            continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        out[t] = (idx.min().normalize(), idx.max().normalize())
    return out


def load_universe(core_cache, big_cache):
    core = cache_names(core_cache)
    cs = set(core)
    rows = [(t, "core", core_cache) for t in core] + \
           [(t, "unseen", big_cache) for t in cache_names(big_cache)
            if t not in cs]
    U = pd.DataFrame(rows, columns=["ticker", "group", "cache"])
    spans = {}
    for cache, g in U.groupby("cache"):
        spans.update(price_spans(g["ticker"].tolist(), cache))
    U["first_price"] = U["ticker"].map(lambda t: spans.get(t, (pd.NaT,))[0])
    U["last_price"] = U["ticker"].map(
        lambda t: spans.get(t, (pd.NaT, pd.NaT))[1])
    return U.dropna(subset=["first_price"]).reset_index(drop=True)


def map_ciks(sec, tickers):
    data = sec.get(SEC_TICKERS, "company_tickers.json")
    if not data:
        raise SystemExit("Could not load SEC's ticker -> CIK map.")
    look = {}
    for row in (data.values() if isinstance(data, dict) else data):
        t = str(row.get("ticker", "")).upper()
        if t:
            look[t] = (f"{int(row['cik_str']):010d}", row.get("title", ""))
    out = {}
    for t in tickers:
        T = t.upper()
        for v in (T, T.replace("-", "."), T.replace(".", "-"),
                  T.replace("-", "")):
            if v in look:
                out[t] = look[v]
                break
    return out


# =============================================================================
# 3. POINT-IN-TIME EXTRACTION
# =============================================================================
PIT_COLS = ["concept", "series", "start", "end", "val", "filed", "src"]


def _dates(values):
    """ISO date strings (None allowed) -> datetime64[D], fast; NaT if bad."""
    try:
        return np.array([v if v else "NaT" for v in values],
                        dtype="datetime64[D]")
    except ValueError:
        return pd.to_datetime(pd.Series(values), errors="coerce") \
            .to_numpy("datetime64[D]")


def raw_rows(facts, aliases, unit):
    """Every reported value of every alias, with its filing date."""
    prio, tags, starts, ends, vals, filed = [], [], [], [], [], []
    block = (facts or {}).get("facts") or {}
    for p, qt in enumerate(aliases):
        space, tag = qt.split(":")
        node = (block.get(space) or {}).get(tag)
        if not node:
            continue
        for e in (node.get("units") or {}).get(unit, []) or []:
            v, en, fi = e.get("val"), e.get("end"), e.get("filed")
            if v is None or not en or not fi:
                continue
            prio.append(p)
            tags.append(tag)
            starts.append(e.get("start"))
            ends.append(en)
            vals.append(float(v))
            filed.append(fi)
    df = pd.DataFrame({"prio": np.array(prio, dtype=int), "tag": tags,
                       "start": pd.to_datetime(_dates(starts)),
                       "end": pd.to_datetime(_dates(ends)),
                       "val": np.array(vals, dtype=float),
                       "filed": pd.to_datetime(_dates(filed))})
    return df.dropna(subset=["end", "filed"]).reset_index(drop=True)


def dur_class(days):
    for c, lo, hi in DUR_CLASSES:
        if lo <= days <= hi:
            return c
    return None


def instant_series(df):
    """One value per period end: the first one filed. Alias priority only
    breaks ties inside the same filing."""
    df = df[df["start"].isna()]
    if df.empty:
        return df
    return (df.sort_values(["end", "filed", "prio"])
              .drop_duplicates("end", keep="first")
              .reset_index(drop=True))


def _frame(rows):
    """(start, end, val, filed, src) with dates as day numbers -> DataFrame."""
    rows = sorted(rows, key=lambda x: x[1])
    day = lambda i: pd.to_datetime(np.array([r[i] for r in rows],
                                            dtype="int64").astype("datetime64[D]"))
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "src"])
    return pd.DataFrame({"start": day(0), "end": day(1),
                         "val": np.array([r[2] for r in rows], dtype=float),
                         "filed": day(3), "src": [r[4] for r in rows]})


def flow_series(df):
    """
    Quarters and trailing-12-month totals, each dated by the filing that
    first made it computable.

      reported quarter   a 3-month fact
      derived quarter    YTD(k) - YTD(k-1) with the same fiscal-year start:
                         Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = FY - 9M
      TTM                the annual figure at a fiscal year end, otherwise the
                         sum of four back-to-back quarters
    """
    stats = Counter()
    d = df[df["start"].notna()]
    if d.empty:
        return _frame([]), _frame([]), stats
    st = d["start"].to_numpy("datetime64[D]").astype(np.int64)
    en = d["end"].to_numpy("datetime64[D]").astype(np.int64)
    fi = d["filed"].to_numpy("datetime64[D]").astype(np.int64)
    va = d["val"].to_numpy(float)
    pr = d["prio"].to_numpy(int)
    days = en - st
    cls = np.zeros(len(d), int)
    for k, (_, lo, hi) in enumerate(DUR_CLASSES, 1):
        cls[(days >= lo) & (days <= hi)] = k
    first = {}                       # (start, end) -> first-filed row
    for i in np.lexsort((pr, fi, en, st)):
        if cls[i] and (st[i], en[i]) not in first:
            first[(st[i], en[i])] = i
    q = {}
    for i in first.values():
        if cls[i] == 1 and (en[i] not in q or fi[i] < q[en[i]][3]):
            q[en[i]] = (st[i], en[i], va[i], fi[i], "reported")
    stats["reported"] = len(q)
    chains = {}                      # fiscal-year start -> {class: row}
    for i in first.values():
        c = chains.setdefault(st[i], {})
        if cls[i] not in c or fi[i] < fi[c[cls[i]]]:
            c[cls[i]] = i
    for ks in chains.values():
        for k in (1, 2, 3):
            a, b = ks.get(k), ks.get(k + 1)
            if a is None or b is None or not 75 <= en[b] - en[a] <= 115:
                continue
            if any(en[b] + x in q for x in range(-7, 8)):
                continue                       # a reported quarter exists
            q[en[b]] = (en[a] + 1, en[b], va[b] - va[a], max(fi[a], fi[b]),
                        "derived")
            stats["derived"] += 1
    qs = sorted(q.values(), key=lambda x: x[1])
    ttm = {}
    for i in sorted((i for i in first.values() if cls[i] == 4),
                    key=lambda i: fi[i]):
        if en[i] not in ttm:
            ttm[en[i]] = (st[i], en[i], va[i], fi[i], "annual")
    for j in range(3, len(qs)):
        w = qs[j - 3:j + 1]
        if all(0 <= w[m + 1][0] - w[m][1] <= 7 for m in range(3)) and \
                350 <= w[3][1] - w[0][0] <= 380 and w[3][1] not in ttm:
            ttm[w[3][1]] = (w[0][0], w[3][1], float(sum(x[2] for x in w)),
                            max(x[3] for x in w), "4 quarters")
            stats["ttm_from_quarters"] += 1
    stats["ttm"] = len(ttm)
    return _frame(list(q.values())), _frame(list(ttm.values())), stats


def extract_company(facts):
    """
    All concepts for one company -> one long point-in-time table, plus the
    bookkeeping the audit reports (tags used, alias gain, quarter sources).
    """
    parts, info = [], {}
    for name, (kind, unit, aliases) in CONCEPTS.items():
        raw = raw_rows(facts, aliases, unit)
        rec = {"n_raw": len(raw)}
        if raw.empty:
            info[name] = rec
            continue
        best = raw[raw["prio"] == raw["prio"].min()]
        if kind == "instant":
            S = instant_series(raw)
            rec.update(n=len(S), n_first_tag=len(instant_series(best)))
            if len(S):
                rec["tags"] = Counter(S["tag"])
                parts.append(S.assign(concept=name, series="inst",
                                      src="reported")[PIT_COLS])
        else:
            Q, T, st = flow_series(raw)
            _, T1, _ = flow_series(best)
            rec.update(n=len(T), n_first_tag=len(T1), quarters=len(Q),
                       q_reported=st["reported"], q_derived=st["derived"])
            used = raw.dropna(subset=["start"]).drop_duplicates(
                ["start", "end", "tag"])
            if len(used):
                rec["tags"] = Counter(used["tag"])
            for frame, series in ((Q, "q"), (T, "ttm")):
                if len(frame):
                    parts.append(frame.assign(concept=name,
                                              series=series)[PIT_COLS])
        info[name] = rec
    pit = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=PIT_COLS)
    return pit, info


def company_profile(facts, sub):
    spaces = set(((facts or {}).get("facts") or {}).keys())
    sic = (sub or {}).get("sic")
    try:
        sic = int(sic)
    except (TypeError, ValueError):
        sic = None
    return {"name": (sub or {}).get("name") or (facts or {}).get("entityName"),
            "sic": sic, "sic_desc": (sub or {}).get("sicDescription"),
            "fy_end": (sub or {}).get("fiscalYearEnd"),
            "has_facts": facts is not None,
            "us_gaap": "us-gaap" in spaces, "ifrs": "ifrs-full" in spaces,
            "financial": sic is not None and FIN_SIC[0] <= sic <= FIN_SIC[1]}


def build_pit(sec, ciks, verbose=True):
    """Download (if needed), extract and cache every company."""
    os.makedirs(FUND_CACHE, exist_ok=True)
    tables, infos, profiles = [], {}, {}
    todo = sorted(set(c for c, _ in ciks.values()))
    t0 = time.time()
    for i, cik in enumerate(todo, 1):
        xp = os.path.join(FUND_CACHE, f"x_{EXTRACT_VERSION}_{cik}.pkl")
        fl = sec.local(f"facts_{cik}.json")
        use_cache = (os.path.exists(xp) and fl is not None and sec.fresh(fl)
                     and os.path.getmtime(xp) >= os.path.getmtime(fl))
        if use_cache:
            pit, info, prof = pd.read_pickle(xp)
        else:
            facts = sec.get(COMPANYFACTS.format(cik=cik), f"facts_{cik}.json")
            sub = sec.get(SUBMISSIONS.format(cik=cik), f"sub_{cik}.json")
            pit, info = extract_company(facts)
            prof = company_profile(facts, sub)
            if facts is not None and sub is not None:
                pd.to_pickle((pit, info, prof), xp)
        if len(pit):
            tables.append(pit.assign(cik=cik))
        infos[cik], profiles[cik] = info, prof
        if verbose and (i % 200 == 0 or i == len(todo)):
            el = time.time() - t0
            left = el / i * (len(todo) - i) / 60
            print(f"      {i:,}/{len(todo):,} companies  {el / 60:5.1f} min  "
                  f"({sec.n_fetched:,} downloads)"
                  + (f"  ~{left:.0f} min left" if i < len(todo) else ""))
    P = pd.concat(tables, ignore_index=True) if tables else \
        pd.DataFrame(columns=PIT_COLS + ["cik"])
    for c in ("start", "end", "filed"):
        P[c] = pd.to_datetime(P[c], errors="coerce")
    P["val"] = pd.to_numeric(P["val"], errors="coerce")
    return P, infos, profiles


# =============================================================================
# 4. AS-OF LOOKUPS (reused by V97)
# =============================================================================
def asof(series, dates, max_age=MAX_AGE_DAYS, lag_years=0):
    """
    For each date, the value that was public on that date. `series` has end,
    filed and val for one company, one concept, one series type. The latest
    period known on the date is used if it ended at most `max_age` days
    earlier. lag_years=1 returns, instead, the value for the period one year
    before that latest period (+/- YEAR_AGO_TOL days), also already public.
    Returns (values, period_ends) - NaN / NaT where nothing usable was public.
    """
    d = np.asarray(dates, dtype="datetime64[ns]")
    vals = np.full(len(d), np.nan)
    pend = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    if series is None or not len(series):
        return vals, pend
    s = series.sort_values(["filed", "end"])
    f = s["filed"].to_numpy("datetime64[ns]")
    e = s["end"].to_numpy("datetime64[ns]")
    v = s["val"].to_numpy(float)
    ei = e.astype("int64")
    cm = np.maximum.accumulate(ei)
    best = np.maximum.accumulate(np.where(ei >= cm, np.arange(len(s)), -1))
    k = np.searchsorted(f, d, side="right") - 1
    order = np.argsort(e, kind="mergesort")
    es = e[order]
    day = np.timedelta64(1, "D")
    for j in np.flatnonzero(k >= 0):
        r = best[k[j]]
        if (d[j] - e[r]) / day > max_age:
            continue
        if lag_years == 0:
            vals[j], pend[j] = v[r], e[r]
            continue
        tgt = e[r] - np.timedelta64(365 * lag_years, "D")
        lo = np.searchsorted(es, tgt - np.timedelta64(YEAR_AGO_TOL, "D"))
        hi = np.searchsorted(es, tgt + np.timedelta64(YEAR_AGO_TOL, "D"),
                             side="right")
        cand = [c for c in order[lo:hi] if f[c] <= d[j]]
        if cand:
            c = min(cand, key=lambda c: abs((e[c] - tgt) / day))
            vals[j], pend[j] = v[c], e[c]
    return vals, pend


def series_of(P_one, concept):
    kind = CONCEPTS[concept][0]
    return P_one[(P_one["concept"] == concept) &
                 (P_one["series"] == ("inst" if kind == "instant" else "ttm"))]


def availability(P, dates):
    """Per company: {(concept, lag): bool array over dates} - was a fresh,
    already-public value there? Flows use the TTM series."""
    need = sorted({(c, l) for alts in SIGNALS.values() for grp in alts
                   for alt in grp for c, l in alt} |
                  {(c, 0) for c in CONCEPTS})
    out = {}
    for cik, g in P.groupby("cik"):
        out[cik] = {(c, l): np.isfinite(asof(series_of(g, c), dates,
                                             lag_years=l)[0])
                    for c, l in need}
    return out


def signal_ok(cols, requirements):
    ok = None
    for alts in requirements:
        g_ok = None
        for alt in alts:
            a = np.logical_and.reduce([cols[x] for x in alt])
            g_ok = a if g_ok is None else (g_ok | a)
        ok = g_ok if ok is None else (ok & g_ok)
    return ok


# =============================================================================
# 5. REPORT
# =============================================================================
def quarter_dates(start_year, end_date):
    try:
        return pd.date_range(f"{start_year}-03-31", end_date, freq="QE")
    except (ValueError, TypeError):
        return pd.date_range(f"{start_year}-03-31", end_date, freq="Q")


def coverage_tables(U, ciks, P, profiles, end_date):
    dates = quarter_dates(2009, end_date)
    dv = dates.values
    avail = availability(P, dv)
    yrs = dates.year.to_numpy()
    alive = []
    for r in U.itertuples(index=False):
        cik = ciks.get(r.ticker, (None,))[0]
        alive.append((cik, (dates >= r.first_price) & (dates <= r.last_price),
                      r.group))
    labels = list(CONCEPTS) + list(SIGNALS)

    def tally(keep):
        num = {lab: np.zeros(len(dates)) for lab in labels}
        den = np.zeros(len(dates))
        for cik, a, grp in alive:
            prof = profiles.get(cik, {}) if cik else {}
            if not keep(prof, grp):
                continue
            den += a
            cols = avail.get(cik) if cik else None
            if cols is None:
                continue
            for c in CONCEPTS:
                num[c] += a & cols[(c, 0)]
            for s, req in SIGNALS.items():
                num[s] += a & signal_ok(cols, req)
        return num, den

    rows_c, rows_s, rows_g = [], [], []
    num, den = tally(lambda p, g: True)
    nnum, nden = tally(lambda p, g: not p.get("financial", False))
    for y in sorted(set(yrs)):
        m = yrs == y
        for lab in CONCEPTS:
            rows_c.append({"item": lab, "year": y, "pct": num[lab][m].sum()
                           / max(den[m].sum(), 1) * 100})
        for lab in SIGNALS:
            rows_s.append({"signal": lab, "year": y, "pct": nnum[lab][m].sum()
                           / max(nden[m].sum(), 1) * 100})
    for grp in ("core", "unseen"):
        gnum, gden = tally(lambda p, g, grp=grp: g == grp
                           and not p.get("financial", False))
        m = yrs >= 2012
        for lab in SIGNALS:
            rows_g.append({"signal": lab, "group": grp,
                           "pct": gnum[lab][m].sum() / max(gden[m].sum(), 1)
                           * 100, "stock-quarters": int(gden[m].sum())})
    return (pd.DataFrame(rows_c), pd.DataFrame(rows_s), pd.DataFrame(rows_g),
            dates)


def split_like_jumps(P):
    """Share counts that jump by a split-like ratio between two periods."""
    ratios = (2, 3, 4, 5, 10, 1.5, 20)
    hits = {}
    sh = P[(P["concept"] == "shares") & (P["series"] == "inst")]
    for cik, g in sh.groupby("cik"):
        g = g.sort_values("end")
        v = g["val"].to_numpy(float)
        e = g["end"].to_numpy("datetime64[ns]")
        n = 0
        for a in range(1, len(v)):
            if v[a - 1] <= 0 or (e[a] - e[a - 1]) / np.timedelta64(1, "D") > 200:
                continue
            r = v[a] / v[a - 1]
            if any(abs(r / x - 1) < 0.03 or abs(r * x - 1) < 0.03
                   for x in ratios):
                n += 1
        if n:
            hits[cik] = n
    return hits


def filing_lags(P):
    """Days from period end to first filing, from the balance-sheet dates."""
    a = P[(P["concept"] == "assets") & (P["series"] == "inst")]
    lag = (a["filed"] - a["end"]).dt.days
    return lag[(lag >= 0) & (lag <= 400)]


def fmt_pct(v):
    return f"{v:.0f}" if v is not None and np.isfinite(v) else "-"


def banner(t, W):
    print("\n" + "-" * W + f"\n  {t}\n" + "-" * W)


def pivot_print(df, key):
    T = df.pivot(index=key, columns="year", values="pct")
    T = T.reindex(columns=[y for y in REPORT_YEARS if y in T.columns])
    T = T.reindex(list(df[key].drop_duplicates()))
    D = T.apply(lambda col: col.map(fmt_pct))
    D.columns = [str(c) for c in D.columns]
    D.index.name = None
    print(D.to_string())
    return T


def items_table(infos, n_companies):
    rows = []
    for name, (kind, unit, aliases) in CONCEPTS.items():
        recs = [infos[c][name] for c in infos if name in infos[c]]
        n_all = sum(r.get("n", 0) for r in recs)
        n_first = sum(r.get("n_first_tag", 0) for r in recs)
        tags = Counter()
        for r in recs:
            tags.update(r.get("tags", Counter()))
        tot = max(sum(tags.values()), 1)
        row = {"item": name,
               "kind": "flow (TTM)" if kind == "flow" else "balance sheet",
               "% of companies": sum(1 for r in recs if r.get("n", 0) > 0)
               / max(n_companies, 1) * 100,
               "periods": n_all,
               "gain from aliases %": (n_all / n_first - 1) * 100
               if n_first else np.nan,
               "quarters derived from YTD %": np.nan,
               "main tags": ", ".join(f"{t[:40]} {c / tot:.0%}"
                                      for t, c in tags.most_common(2))}
        if kind == "flow":
            qr = sum(r.get("q_reported", 0) for r in recs)
            qd = sum(r.get("q_derived", 0) for r in recs)
            row["quarters derived from YTD %"] = qd / max(qr + qd, 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def run_v96(core_cache=None, big_cache="price_cache_v68",
            model_in81="entry_model_v81.joblib", sec_contact="",
            edgar_cache="edgar_cache", offline=False, refresh_days=45,
            out_dir=OUT_DIR):
    t0 = time.time()
    W = 118
    os.makedirs(out_dir, exist_ok=True)
    print("=" * W)
    print("V96 - FUNDAMENTALS FROM SEC EDGAR: what can a point-in-time filter be "
          "built from?")
    print("=" * W)
    if core_cache is None:
        import class_ai_entry_model_v81 as M81
        core_cache = M81.load_model(model_in81)["provenance"]["price_cache"]

    # ---- [1] universe and CIKs ----------------------------------------------
    print("  reading the price caches ...")
    U = load_universe(core_cache, big_cache)
    sec = Sec(sec_contact, edgar_cache, offline=offline,
              refresh_days=refresh_days)
    ciks = map_ciks(sec, U["ticker"])
    end_date = U["last_price"].max()
    banner("[1] OUR STOCKS ON EDGAR", W)
    for grp in ("core", "unseen"):
        g = U[U["group"] == grp]
        hit = int(g["ticker"].isin(list(ciks)).sum())
        print(f"  {grp:<7} {len(g):5,} stocks with prices, {hit:5,} found on "
              f"EDGAR ({hit / max(len(g), 1):.0%})")
    miss = sorted(set(U["ticker"]) - set(ciks))
    if miss:
        print(f"  not found ({len(miss)}): {', '.join(miss[:25])}"
              + (" ..." if len(miss) > 25 else ""))

    # ---- [2] downloads and extraction ---------------------------------------
    n_co = len(set(c for c, _ in ciks.values()))
    print(f"\n  reading or downloading XBRL facts and filing profiles for "
          f"{n_co:,} companies ...")
    P, infos, profiles = build_pit(sec, ciks)
    if sec.errors:
        print("  download problems: " + ", ".join(
            f"{k}: {v}" for k, v in sec.errors.items()))
    P.to_pickle(os.path.join(FUND_CACHE, f"pit_{EXTRACT_VERSION}.pkl"))
    grp_of = dict(zip(U["ticker"], U["group"]))
    meta = pd.DataFrame([{"ticker": t, "cik": c, "title": ti,
                          "group": grp_of.get(t), **profiles.get(c, {})}
                         for t, (c, ti) in ciks.items()])
    meta.to_csv(os.path.join(out_dir, "universe_map.csv"), index=False)
    meta.to_pickle(os.path.join(FUND_CACHE, f"meta_{EXTRACT_VERSION}.pkl"))

    # ---- [2] what kind of filers --------------------------------------------
    banner("[2] WHAT KIND OF FILERS", W)
    prof = pd.DataFrame.from_dict(profiles, orient="index")
    for c in ("has_facts", "us_gaap", "ifrs", "financial"):
        if c not in prof:
            prof[c] = False
        prof[c] = prof[c].fillna(False).astype(bool)
    print(f"  {len(prof):,} companies: {int(prof['has_facts'].sum()):,} have "
          f"XBRL facts; {int(prof['us_gaap'].sum()):,} report in US GAAP; "
          f"{int((prof['ifrs'] & ~prof['us_gaap']).sum()):,} only in IFRS "
          f"(foreign filers - not read by this version)")
    print(f"  financials (SIC {FIN_SIC[0]}-{FIN_SIC[1]}: banks, insurers, "
          f"REITs): {int(prof['financial'].sum()):,} - most signals do not "
          f"apply to them, so signal coverage below leaves them out")

    # ---- [3] items -------------------------------------------------------------
    banner("[3] ACCOUNTING ITEMS - share of companies with any point-in-time "
           "value, and what merging tag aliases adds", W)
    I = items_table(infos, len(prof))
    I.to_csv(os.path.join(out_dir, "items.csv"), index=False)
    D = pd.DataFrame({
        "item": I["item"], "kind": I["kind"],
        "% companies": I["% of companies"].map(fmt_pct),
        "periods": I["periods"].map(lambda v: f"{v:,}"),
        "+ from aliases": I["gain from aliases %"].map(
            lambda v: f"+{v:.0f}%" if np.isfinite(v) else "-"),
        "quarters from YTD": I["quarters derived from YTD %"].map(
            lambda v: f"{v:.0f}%" if np.isfinite(v) else ""),
        "main tags": I["main tags"]})
    print(D.to_string(index=False))
    lags = filing_lags(P)
    if len(lags):
        print(f"\n  filing lag, period end -> first public: median "
              f"{lags.median():.0f} days, 90% within {lags.quantile(0.9):.0f} "
              f"days. Every value is used only from its filing date.")

    # ---- [4]-[5] coverage by year -------------------------------------------
    print("\n  checking, quarter by quarter, which values were already "
          "public ...")
    C, S, G, dates = coverage_tables(U, ciks, P, profiles, end_date)
    C.to_csv(os.path.join(out_dir, "coverage_items_by_year.csv"), index=False)
    S.to_csv(os.path.join(out_dir, "coverage_signals_by_year.csv"),
             index=False)
    G.to_csv(os.path.join(out_dir, "coverage_signals_core_vs_unseen.csv"),
             index=False)
    banner(f"[4] ITEMS - % of our stocks (trading that quarter) with a public "
           f"value for a period ended within {MAX_AGE_DAYS} days", W)
    pivot_print(C, "item")
    banner("[5] SIGNALS - % of non-financial stocks where the signal could be "
           "computed from public data", W)
    pivot_print(S, "signal")
    print("\n  2012-2026, non-financial stocks, core vs unseen:")
    Gp = G.pivot(index="signal", columns="group", values="pct") \
        .reindex(list(SIGNALS))
    Gp.index.name = None
    print(Gp.apply(lambda c: c.map(fmt_pct)).to_string())

    # ---- [6] data issues ------------------------------------------------------
    banner("[6] DATA ISSUES V97 MUST HANDLE", W)
    jumps = split_like_jumps(P)
    n_sh = P[P["concept"] == "shares"]["cik"].nunique()
    print(f"  share counts with a split-like jump (x2, x3, x1.5, /10 ...): "
          f"{len(jumps):,} of {n_sh:,} companies. Share issuance must be "
          f"split-adjusted, or a split reads as massive dilution.")
    yt = I.set_index("item")["quarters derived from YTD %"]
    if np.isfinite(yt.get("cfo", np.nan)):
        share = yt["cfo"]
        print(f"  operating cash flow: {share:.0f}% of its quarters exist only "
              f"as year-to-date differences"
              + (". Code that keeps only 3-month cash-flow facts - as Pillar "
                 "2's runway flag does - sees mostly first quarters."
                 if share >= 50 else "."))
    debt = C[(C["item"] == "lt_debt") & (C["year"] >= 2012)]["pct"].mean()
    print(f"  long-term debt has a value for {debt:.0f}% of stock-quarters "
          f"(2012+). A missing debt tag often means no debt; the F-score row "
          f"'debt missing = 0' shows coverage under that reading.")

    # ---- verdict --------------------------------------------------------------
    banner("WHAT THIS MEANS", W)
    for sig in SIGNALS:
        ys = S[S["signal"] == sig].set_index("year")["pct"]
        first = next((y for y, v in ys.items() if v >= 70), None)
        recent = ys[ys.index >= 2012].mean()
        when = f"70%+ from {first}" if first else "never reaches 70%"
        print(f"  {sig:<28} {when:<20} average coverage 2012-2026: "
              f"{recent:.0f}%")
    print(f"\n  point-in-time table for V97: {FUND_CACHE}/pit_{EXTRACT_VERSION}"
          f".pkl ({len(P):,} rows) | tables in {out_dir}/ | "
          f"{(time.time() - t0) / 60:.1f} min")
    return U, ciks, P, profiles


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_SEC_CONTACT  = "ongwork88up@gmail.com"               # "Your Name your.email@example.com"
    RUN_CORE_CACHE   = None             # None = the core cache V86-V95 used
    RUN_BIG_CACHE    = "price_cache_v68"
    RUN_EDGAR_CACHE  = "edgar_cache"    # reuses files already downloaded
    RUN_REFRESH_DAYS = 45               # re-download cached files older than
                                        # this, so recent filings are included
    RUN_OFFLINE      = False            # True = use the cache only
    RUN_OUT_DIR      = "thesis_tables_v96"
    # -------------------------------------------------------------------------

    run_v96(core_cache=RUN_CORE_CACHE, big_cache=RUN_BIG_CACHE,
            sec_contact=RUN_SEC_CONTACT, edgar_cache=RUN_EDGAR_CACHE,
            offline=RUN_OFFLINE, refresh_days=RUN_REFRESH_DAYS,
            out_dir=RUN_OUT_DIR)
