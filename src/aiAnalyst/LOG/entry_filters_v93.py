"""
V93 - CAN THE TARGET HIT RATE GET BETTER?  DON'T BUY FALLING KNIVES.

WHY THE HIT RATE LOOKS ORDINARY
-------------------------------
Two things set it, and neither is a fault of the timing:

  1. The trade's shape. Target = 0.5 sigma of the holding window, stop = 0.6
     x target, so reward:risk is 1.67. For a stock that wanders randomly
     the target is hit first about 1 / (1 + 1.67) = 37.5% of the time. That
     is the break-even rate, and it is why ~37% shows up for ANY day.
  2. Where the model fires. V89 measured it: the model picks a much better
     day (+10 pp) inside stock-months that are worse than average (-9 pp),
     because it fires on stocks that are falling. Net: about level with an
     average entry.

Moving the target closer would raise the hit rate on paper and lower the
payoff by the same amount - no better trading, and the first thing a
reviewer would catch. The honest lever is (2): stop entering dips inside
downtrends. "Buy the 2-day dip only when the stock is above its 200-day
average" is the classic short-term mean-reversion rule (Connors & Alvarez,
2008) - and RSI(2), the feature this model leans on most (V89), is that
rule's own trigger.

WHAT IS TESTED (fixed before any result is seen)
------------------------------------------------
A filter only removes ENTER signals; the model and its cut do not change.

  F0  none                       V88's signals as they are (the reference)
  F1  uptrend                    the stock closes above its 200-day EMA
  F2  market not in a bear phase full-universe breadth >= 0.40
  F3  uptrend AND market         both

  CHOOSE   on the core stocks, 2017-2021: the filter with the highest
           average R per ENTER trade, if it keeps at least 100 signals there
           and beats F0. If none beats F0, no filter is adopted.
  CONFIRM  on the unseen liquid stocks, 2022-2026 - other stocks, later
           years, never used for any choice. The chosen filter is CONFIRMED
           only if all three hold:
             1. its ENTER signals hit the target more often than an average
                entry in the same month (all stocks), lower 90% bound > 0
             2. its average R per trade is at least F0's (it improves the
                trading, not just the hit rate)
             3. its timing edge over other days in the same stock and month
                stays positive (lower 90% bound > 0)

Signals come from the walk-forward copies of V88 (each year scored by a model
trained only on earlier years) - read from V86's caches, nothing retrained.

READING THE TABLES
------------------
  Target / Stop / Expired   share of ENTER trades ending each way (60 days)
  Avg R                     average result per trade in units of the stop
                            distance (+1.67 R = target, -1 R = stop)
  vs same stock-month       ENTER minus the average of ALL days in the same
                            stock and month - the timing claim
  vs average entry          ENTER minus the average of all stocks' days in
                            the same month - "better than buying anything"
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

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = "thesis_tables_v93"
SELECT_YEARS = list(range(2017, 2022))
CONFIRM_YEARS = list(range(2022, 2027))
MIN_SIGNALS = 100
BEAR = V82.BREADTH_CUTS[0]

FILTERS = {
    "F0 none": lambda U: np.ones(len(U), bool),
    "F1 uptrend (close > 200-day EMA)":
        lambda U: U["px_vs_ema200"].to_numpy(float) > 0,
    "F2 market not bear (breadth >= 0.40)":
        lambda U: U["breadth"].to_numpy(float) >= BEAR,
    "F3 uptrend + market not bear":
        lambda U: (U["px_vs_ema200"].to_numpy(float) > 0)
        & (U["breadth"].to_numpy(float) >= BEAR),
}

PREREG = """PRE-REGISTRATION - V93 entry filters
written {now}, before any filter result was computed{replaces}.

FILTERS   applied to the walk-forward V88 ENTER signals; the model and its cut
          are unchanged.
            F0 none | F1 close above its 200-day EMA | F2 full-universe
            breadth >= 0.40 | F3 F1 and F2
CHOOSE    core stocks, {s0}-{s1}: the filter among F1-F3 with the highest
          average R per ENTER trade, with at least {mn} signals, and only if it
          beats F0's average R there. Otherwise no filter is adopted.
CONFIRM   unseen liquid stocks, {c0}-{c1}. CONFIRMED only if all hold:
            1. target hit rate minus the same-month all-stock average, 90%
               month-bootstrap lower bound > 0
            2. average R per trade >= F0's (point estimate)
            3. target hit rate minus the same stock-month average, 90% lower
               bound > 0
METHOD    calendar-month bootstrap, {nb} draws; differences paired by month.
"""


# =============================================================================
# DATA
# =============================================================================
def add_ema200(U, mask, cache):
    """px_vs_ema200 on the rows in `mask`, from the prices, with E64's EMA."""
    out = np.full(len(U), np.nan)
    rows = np.flatnonzero(mask)
    sub = U.iloc[rows]
    for t, g in sub.groupby("ticker"):
        df = E64.P2.load_prices(t, cache)
        if df is None:
            continue
        c = df["Close"].astype(float)
        px = (c / E64._ema(c, 200) - 1).to_numpy()
        pos = pd.DatetimeIndex(df.index).get_indexer(
            pd.DatetimeIndex(g["date"]))
        ok = pos >= 0
        out[g.index.to_numpy()[ok]] = px[pos[ok]]
    return out


def outcome_indicators(U):
    lab = U["label"].to_numpy(float)
    r = U["r_multiple"].to_numpy(float)
    tgt = (lab == 1).astype(float)
    stp = ((lab != 1) & (r <= -0.999)).astype(float)
    exp = 1.0 - tgt - stp
    return {"tgt": tgt, "stp": stp, "exp": exp, "R": r}


def group_means(U, x):
    per = pd.PeriodIndex(pd.DatetimeIndex(U["date"]), freq="M")
    m_key = per.astype(str).to_numpy()
    tm_key = U["ticker"].astype(str).to_numpy() + "|" + m_key
    s = pd.Series(x)
    return (s.groupby(tm_key).transform("mean").to_numpy(),
            s.groupby(m_key).transform("mean").to_numpy())


class Stats:
    """Month-bootstrap statistics on one universe, weights shared by every
    comparison so differences are paired."""

    def __init__(self, U, n_boot):
        self.ev = V85.Evaluator(U, n_boot=n_boot)
        self.ind = outcome_indicators(U)
        self.pool, self.base = {}, {}
        for k, v in self.ind.items():
            self.pool[k], self.base[k] = group_means(U, v)

    def _sum(self, x, m, years):
        ev = self.ev
        idx, W = ev._weights(years)
        mm = m & np.isin(ev.year, years)
        k = len(ev.months)
        s = np.bincount(ev.mcode[mm], x[mm], k)[idx]
        c = np.bincount(ev.mcode[mm], minlength=k)[idx].astype(float)
        return W, s, c, mm

    def mean_ci(self, x, m, years):
        W, s, c, mm = self._sum(x, m, years)
        if c.sum() < 1:
            return np.nan, np.nan, np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (W @ s) / (W @ c)
        return s.sum() / c.sum(), np.nanpercentile(b, 5), \
            np.nanpercentile(b, 95)

    def paired(self, xa, ma, xb, mb, years):
        W, sa, ca, _ = self._sum(xa, ma, years)
        _, sb, cb, _ = self._sum(xb, mb, years)
        if ca.sum() < 1 or cb.sum() < 1:
            return np.nan, np.nan, np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (W @ sa) / (W @ ca) - (W @ sb) / (W @ cb)
        return (sa.sum() / ca.sum() - sb.sum() / cb.sum(),
                np.nanpercentile(d, 5), np.nanpercentile(d, 95))

    def row(self, m, years, f0=None):
        ind, pool, base = self.ind, self.pool, self.base
        n = int((m & np.isin(self.ev.year, years)).sum())
        if n == 0:
            return {"Signals": 0}
        mean = lambda x: self.mean_ci(x, m, years)[0]
        out = {"Signals": n,
               "Target %": mean(ind["tgt"]) * 100,
               "Stop %": mean(ind["stp"]) * 100,
               "Expired %": mean(ind["exp"]) * 100,
               "Avg R": mean(ind["R"])}
        for lab, ref in (("vs same stock-month", pool),
                         ("vs average entry", base)):
            d, lo, hi = self.mean_ci((ind["tgt"] - ref["tgt"]) * 100, m,
                                     years)
            out[f"Target {lab} pp"] = d
            out[f"Target {lab} lo"] = lo
            out[f"Target {lab} hi"] = hi
            dr, lor, hir = self.mean_ci(ind["R"] - ref["R"], m, years)
            out[f"R {lab}"] = dr
            out[f"R {lab} lo"] = lor
            out[f"R {lab} hi"] = hir
        if f0 is not None:
            d, lo, hi = self.paired(ind["R"], m, ind["R"], f0, years)
            out["R vs F0"], out["R vs F0 lo"], out["R vs F0 hi"] = d, lo, hi
        return out

    def baselines(self, m, years):
        """Outcome shares of the comparison groups for the same signals."""
        out = {}
        for lab, ref in (("all days of the same stock & month", self.pool),
                         ("average entry, same month (all stocks)",
                          self.base)):
            mm = m & np.isin(self.ev.year, years)
            out[lab] = {"Signals": int(mm.sum()),
                        "Target %": ref["tgt"][mm].mean() * 100,
                        "Stop %": ref["stp"][mm].mean() * 100,
                        "Expired %": ref["exp"][mm].mean() * 100,
                        "Avg R": ref["R"][mm].mean()}
        return out


# =============================================================================
# PRE-REGISTRATION
# =============================================================================
def preregister(out_dir, n_boot):
    settings = {"filters": list(FILTERS), "select": SELECT_YEARS,
                "confirm": CONFIRM_YEARS, "min": MIN_SIGNALS,
                "bear": BEAR, "n_boot": n_boot}
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
        s0=SELECT_YEARS[0], s1=SELECT_YEARS[-1], c0=CONFIRM_YEARS[0],
        c1=CONFIRM_YEARS[-1], mn=MIN_SIGNALS, nb=n_boot)
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
def ci(lo, hi):
    return f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) and np.isfinite(hi) \
        else ""


def table(stats, fires, U, years):
    f0 = fires & FILTERS["F0 none"](U)
    rows = []
    for name, fn in FILTERS.items():
        m = fires & fn(U)
        r = stats.row(m, years, f0=None if name == "F0 none" else f0)
        rows.append({"Filter": name, **r})
    T = pd.DataFrame(rows)
    n0 = T.loc[T["Filter"] == "F0 none", "Signals"].iloc[0]
    T.insert(2, "Kept %", T["Signals"] / n0 * 100 if n0 else np.nan)
    return T


def show(T, title, W=110):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    D = pd.DataFrame({"Filter": T["Filter"], "Signals": T["Signals"]})
    D["Kept %"] = T["Kept %"].map(lambda v: f"{v:.0f}")
    for c in ("Target %", "Stop %", "Expired %"):
        D[c] = T[c].map(lambda v: f"{v:.1f}")
    D["Avg R"] = T["Avg R"].map(lambda v: f"{v:+.3f}")
    D["Target vs avg entry"] = [
        f"{d:+.1f} {ci(lo, hi)}" for d, lo, hi in zip(
            T["Target vs average entry pp"], T["Target vs average entry lo"],
            T["Target vs average entry hi"])]
    D["Target vs same stock-month"] = [
        f"{d:+.1f} {ci(lo, hi)}" for d, lo, hi in zip(
            T["Target vs same stock-month pp"],
            T["Target vs same stock-month lo"],
            T["Target vs same stock-month hi"])]
    if "R vs F0" in T:
        D["R vs F0"] = [f"{d:+.3f} {ci(lo, hi)}" if np.isfinite(d) else "-"
                        for d, lo, hi in zip(T["R vs F0"], T["R vs F0 lo"],
                                             T["R vs F0 hi"])]
    print(D.to_string(index=False))


def presentable(st, fires, U, T_conf, chosen, out_dir):
    """Target / stop / expired and average R for the ENTER trades and for the
    two comparison groups (same stock-months, same months), on the
    confirmation data - one block per signal set."""
    sets = [("F0 none", "ENTER, no filter (V88 as is)")]
    if chosen and chosen != "F0 none":
        sets.append((chosen, f"ENTER + {chosen.split(' ', 1)[1]}"))
    rows = []
    for key, label in sets:
        r = T_conf[T_conf["Filter"] == key].iloc[0]
        rows.append({"Entry": label, **{k: r[k] for k in (
            "Signals", "Target %", "Stop %", "Expired %", "Avg R")}})
        m = fires & FILTERS[key](U)
        for k, v in st.baselines(m, CONFIRM_YEARS).items():
            rows.append({"Entry": f"    vs {k}", **v})
    P = pd.DataFrame(rows)
    D = P.copy()
    for c in ("Target %", "Stop %", "Expired %"):
        D[c] = P[c].map(lambda v: f"{v:.1f}")
    D["Avg R"] = P["Avg R"].map(lambda v: f"{v:+.3f}")
    print("\n" + "-" * 110)
    print(f"  [3] THE TABLE TO PRESENT - unseen liquid stocks, "
          f"{CONFIRM_YEARS[0]}-{CONFIRM_YEARS[-1]}. Each 'vs' row uses the "
          f"same stock-months / months as the ENTER row above it.")
    print("-" * 110)
    print(D.to_string(index=False))
    P.to_csv(f"{out_dir}/presentable_unseen.csv", index=False)


def run_v93(model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
            out_dir=OUT_DIR, verbose=False):
    t0 = time.time()
    print("=" * 110)
    print("V93 - CAN THE TARGET HIT RATE GET BETTER? Filters on V88's ENTER "
          "signals (pre-registered)")
    print("=" * 110)
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    print("  reading V86's data and walk-forward models (cached) ...")
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    print("  market breadth from the core stocks ...")
    b, _ = V82.full_breadth(core_cache, V86.cache_names(core_cache))
    CORE = f"core {n_core}"
    unis = {}
    for name, frame, mask, tag, cache in (
            (CORE, core, np.ones(len(core), bool), "core", core_cache),
            ("unseen, liquid", fresh, fresh["liquid"].to_numpy(bool),
             "fresh", big_cache)):
        U, w = V87.universe(frame, mask, wf, tag)
        fires = V85.model_fire(U, *w["combo"])
        U["breadth"] = pd.DatetimeIndex(U["date"]).normalize().map(b) \
            .to_numpy(float)
        print(f"  {name}: 200-day EMA on {int(fires.sum()):,} ENTER rows ...")
        U["px_vs_ema200"] = add_ema200(U, fires, cache)
        unis[name] = (U, fires, Stats(U, n_boot))

    print("\n  PRE-REGISTRATION")
    when, prereg_ok = preregister(out_dir, n_boot)

    # ---- choose on core 2017-2021 --------------------------------------------
    U, fires, st = unis[CORE]
    T_sel = table(st, fires, U, SELECT_YEARS)
    show(T_sel, f"[1] CHOOSE - {CORE}, {SELECT_YEARS[0]}-{SELECT_YEARS[-1]} "
                f"(walk-forward signals)")
    T_sel.to_csv(f"{out_dir}/choose_core.csv", index=False)
    r0 = float(T_sel.loc[T_sel["Filter"] == "F0 none", "Avg R"].iloc[0])
    cand = T_sel[(T_sel["Filter"] != "F0 none")
                 & (T_sel["Signals"] >= MIN_SIGNALS)
                 & (T_sel["Avg R"] > r0)]
    chosen = cand.loc[cand["Avg R"].idxmax(), "Filter"] if len(cand) else None
    print(f"\n  chosen: {chosen or 'none - no filter beats F0 on the core'}")

    # ---- confirm on unseen 2022-2026 ----------------------------------------
    U, fires, st = unis["unseen, liquid"]
    T_conf = table(st, fires, U, CONFIRM_YEARS)
    show(T_conf, f"[2] CONFIRM - unseen liquid stocks, {CONFIRM_YEARS[0]}-"
                 f"{CONFIRM_YEARS[-1]} (all filters shown; only the chosen "
                 f"one is the claim)")
    T_conf.to_csv(f"{out_dir}/confirm_unseen.csv", index=False)

    lines = []
    if chosen:
        r = T_conf[T_conf["Filter"] == chosen].iloc[0]
        r0c = T_conf[T_conf["Filter"] == "F0 none"].iloc[0]
        c1 = r["Target vs average entry lo"] > 0
        c2 = r["Avg R"] >= r0c["Avg R"]
        c3 = r["Target vs same stock-month lo"] > 0
        lines += [
            f"  chosen on the core: {chosen}",
            f"  1. target vs an average entry in the same month: "
            f"{r['Target vs average entry pp']:+.1f} pp "
            f"{ci(r['Target vs average entry lo'], r['Target vs average entry hi'])}"
            f"  -> {'PASS' if c1 else 'FAIL'}",
            f"  2. average R per trade: {r['Avg R']:+.3f} vs F0 "
            f"{r0c['Avg R']:+.3f}  -> {'PASS' if c2 else 'FAIL'}",
            f"  3. target vs all days of the same stock-month: "
            f"{r['Target vs same stock-month pp']:+.1f} pp "
            f"{ci(r['Target vs same stock-month lo'], r['Target vs same stock-month hi'])}"
            f"  -> {'PASS' if c3 else 'FAIL'}"]
        ok = c1 and c2 and c3
        lines.append(f"\n  VERDICT: {'CONFIRMED' if ok else 'NOT CONFIRMED'}"
                     + (f" - adopt '{chosen}' in the recommender."
                        if ok else " - keep V88's signals unfiltered."))
    else:
        lines.append("\n  VERDICT: NOT ADOPTED - no filter beat F0 on the core "
                     "stocks; keep V88's signals unfiltered.")
    presentable(st, fires, U, T_conf, chosen, out_dir)
    if not prereg_ok:
        lines.append("  (settings changed after an earlier result - "
                     "exploratory)")

    # ---- secondary: every universe and period --------------------------------
    for name, yrs, key in ((CORE, CONFIRM_YEARS, "core_2022_26"),
                           ("unseen, liquid", SELECT_YEARS,
                            "unseen_2017_21"),
                           ("unseen, liquid", SELECT_YEARS + CONFIRM_YEARS,
                            "unseen_2017_26")):
        U, fires, st = unis[name]
        T = table(st, fires, U, yrs)
        show(T, f"secondary - {name}, {yrs[0]}-{yrs[-1]}")
        T.to_csv(f"{out_dir}/secondary_{key}.csv", index=False)

    print("\n" + "=" * 110 + "\n  THE PRE-REGISTERED VERDICT\n" + "=" * 110)
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
    RUN_OUT_DIR   = "thesis_tables_v93"
    # -------------------------------------------------------------------------

    run_v93(big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK, n_boot=RUN_N_BOOT,
            out_dir=RUN_OUT_DIR)
