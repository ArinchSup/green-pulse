"""
V99 - A RED-FLAG VETO: BLOCK ENTRY ONLY ON CLEAR WARNING SIGNS (pre-registered)

WHY
---
V98's broad filter (drop the bottom 30% by a quality score) did not pass. Two
things stood out, both found after looking, so neither is a claim yet:

  - companies raising money or burning cash did worse (share growth, net
    issuance, cash runway)
  - the quality score's very worst stocks (~1%) hit their target 5 points
    less often

And your friend's veto (health score <= 4) removed stocks that did slightly
better. So V99 tests a narrow veto instead of a broad filter: block entry
only when a clear red flag is present. Two candidates are Pillar 2's OWN hard
flags, which the pipeline already uses to refuse trades - measured here with
V97's corrected data (split-proof share counts, a cash runway that reads all
four quarters).

CANDIDATES (fixed before any V99 result; a stock with no data is never vetoed)
------------------------------------------------------------------------------
  D   heavy dilution   shares up more than 15% in a year
                       (Pillar 2's HIGH_DILUTION threshold)
  R   short runway     under 6 quarters of cash + short-term investments at
                       the current burn (Pillar 2's SHORT_RUNWAY threshold)
  DR  either flag      Pillar 2's hard-flag veto, as the pipeline uses it
  T2  worst 2%         of the month's stocks by V98's quality score
  T5  worst 5%         of the month's stocks by V98's quality score

CHOOSE AND CONFIRM
------------------
  CHOOSE   unseen LIQUID non-financial stocks, 2012-2026 - V98's data,
           already seen, used only to choose. Eligible: vetoes 0.5%-15% of
           candidate days, and the vetoed days did worse. Chosen: the one
           whose gap is most clearly above zero (highest lower 90% bound).
           None eligible -> NOT ADOPTED; the confirmation outcomes are not
           read.
  CONFIRM  unseen non-financial stocks that were NEVER liquid in 2012-2026:
           different companies from the choice data, never used for any
           fundamental question. Power gate (counted before any outcome is
           read): at least 1,000 vetoed candidate days, else INCONCLUSIVE.
           CONFIRMED only if all hold - the same bar as V98:
             1. kept minus vetoed, target hit rate: 90% lower bound > 0
             2. kept minus vetoed, average R: 90% lower bound > 0
             3. rule 1 inside volatility thirds: estimate > 0
  REPORTED your friend's veto on the same data; V88's ENTER signals with the
           veto; the core stocks - descriptive.

Outcome, signal timing and bootstrap exactly as in V98: V88's 60-day target /
stop label on every candidate day, scores as of the previous month-end,
calendar-month bootstrap with comparisons paired by month.
"""

import hashlib
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import class_ai_entry_model_v81 as M81
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import entry_filters_v93 as V93
import entry_filters_v94 as V94
import fundamentals_audit_v96 as F96
import fundamental_signals_v97 as F97
import fundamental_filter_v98 as V98

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = "thesis_tables_v99"
YEARS = V98.YEARS
DILUTION = 0.15                  # Pillar 2's HIGH_DILUTION: shares +15% in a year
RUNWAY = 6.0                     # Pillar 2's SHORT_RUNWAY: under 6 quarters
VETO_SHARE = (0.005, 0.15)       # a veto, not a filter
MIN_VETOED = 1000                # power gate on the confirmation data
MIN_SHOW = 100                   # fewer vetoed days -> no confidence interval
FRIEND_VETO = V98.VETO           # his health score <= 4

CANDIDATES = {
    "D  shares +15% in a year (Pillar 2 HIGH_DILUTION)":
        lambda J: J["share_growth"].to_numpy(float) > DILUTION,
    "R  runway under 6 quarters (Pillar 2 SHORT_RUNWAY)":
        lambda J: J["runway_q"].to_numpy(float) < RUNWAY,
    "DR either flag (Pillar 2's hard-flag veto)":
        lambda J: (J["share_growth"].to_numpy(float) > DILUTION)
        | (J["runway_q"].to_numpy(float) < RUNWAY),
    "T2 worst 2% by V98's quality score":
        lambda J: J["comp_pct"].to_numpy(float) <= 0.02,
    "T5 worst 5% by V98's quality score":
        lambda J: J["comp_pct"].to_numpy(float) <= 0.05,
}

PREREG = """PRE-REGISTRATION - V99 red-flag veto
written {now}{replaces}.

CANDIDATES  D shares +{dil:.0%} in a year | R runway < {rw:g} quarters | DR
            either | T2 worst 2% by V98's quality score | T5 worst 5%.
            No data -> never vetoed. Scores from V97 {ver} as of the
            previous month-end.
CHOOSE      unseen liquid non-financial stocks, {y0}-{y1} (seen in V98):
            eligible if it vetoes {lo:.1%}-{hi:.0%} of candidate days and the
            vetoed days did worse; chosen = highest lower 90% bound of kept
            minus vetoed target hit rate. None -> NOT ADOPTED, confirmation
            outcomes not read.
CONFIRM     unseen non-financial stocks never liquid in {y0}-{y1}. Power gate:
            >= {gate:,} vetoed candidate days, else INCONCLUSIVE. CONFIRMED only
            if: 1. kept - vetoed target hit rate, lower bound > 0; 2. kept -
            vetoed average R, lower bound > 0; 3. rule 1 within volatility
            thirds, estimate > 0.
METHOD      calendar-month bootstrap, {nb} draws, paired by month.
"""


# =============================================================================
# DATA
# =============================================================================
def attach(U, P, group):
    """Scores and flags as of the previous month-end, for each candidate day."""
    prev = (pd.DatetimeIndex(U["date"]).to_period("M") - 1) \
        .to_timestamp(how="end").normalize()
    cols = ["ticker", "month", "financial", "comp_pct", "share_growth",
            "runway_q", "fr_health", "vt2"]
    Pg = P[P["group"] == group][cols]
    key = pd.DataFrame({"ticker": U["ticker"].to_numpy(), "month": prev})
    J = key.merge(Pg, on=["ticker", "month"], how="left")
    J["financial"] = J["financial"].fillna(False).astype(bool)
    return J


def prepare_panel(core_cache, big_cache):
    panel_p = os.path.join(F96.FUND_CACHE, f"signals_{F97.VERSION}.pkl")
    if not os.path.exists(panel_p):
        raise SystemExit(f"{panel_p} not found - run V97 ({F97.VERSION}) first.")
    P = pd.read_pickle(panel_p)
    P["month"] = pd.DatetimeIndex(P["month"]).normalize()
    P["financial"] = P["financial"].fillna(False).astype(bool)
    P = V98.add_scores(P)
    PM = pd.concat([V98.price_monthly(sorted(P.loc[P["group"] == g, "ticker"]
                                             .unique()), c)
                    for g, c in (("core", core_cache), ("unseen", big_cache))],
                   ignore_index=True)
    PM["month"] = pd.DatetimeIndex(PM["month"]).normalize()
    P = P.merge(PM, on=["ticker", "month"], how="left")
    ok = ~P["financial"] & P["vol60"].notna()
    P["vt2"] = np.nan
    P.loc[ok, "vt2"] = P[ok].groupby(["group", "month"])["vol60"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 3, labels=False)
        if len(v) >= 3 else np.nan)
    return P


# =============================================================================
# TABLES
# =============================================================================
def veto_table(B, ind, J, base, years, names=None):
    """One row per candidate: how much it vetoes, how vetoed vs kept days did."""
    rows, flags = [], {}
    yr = np.isin(B.year, years)
    n_base = int((base & yr).sum())
    for name, fn in CANDIDATES.items():
        if names is not None and name not in names:
            continue
        f = base & np.nan_to_num(fn(J), nan=0).astype(bool)
        flags[name] = f
        kept = base & ~f
        p1, d1, nk, nv = B.diff(ind["tgt"], kept, f, years)
        p2, d2, _, _ = B.diff(ind["R"], kept, f, years)
        lo1, hi1 = V98.ci(d1 * 100)
        lo2, hi2 = V98.ci(d2)
        if nv < MIN_SHOW:            # a handful of days: no honest interval
            p1 = p2 = lo1 = hi1 = lo2 = hi2 = np.nan
        rows.append({"Veto": name, "vetoed days": nv,
                     "vetoed %": nv / max(n_base, 1) * 100,
                     "Target % kept": ind["tgt"][kept & yr].mean() * 100,
                     "Target % vetoed": ind["tgt"][f & yr].mean() * 100
                     if nv else np.nan,
                     "gap pp": p1 * 100, "gap lo": lo1, "gap hi": hi1,
                     "gap R": p2, "gap R lo": lo2, "gap R hi": hi2})
    return pd.DataFrame(rows), flags


def show_vetoes(T, title, W=118):
    print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
    D = pd.DataFrame({
        "Veto": T["Veto"],
        "vetoed days": T["vetoed days"].map(lambda v: f"{int(v):,}"),
        "vetoed %": T["vetoed %"].map(lambda v: f"{v:.1f}"),
        "Target % kept": T["Target % kept"].map(V94.pct),
        "Target % vetoed": T["Target % vetoed"].map(V94.pct),
        "kept - vetoed, target pp [90%]": [
            f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]" if np.isfinite(p) else "-"
            for p, lo, hi in zip(T["gap pp"], T["gap lo"], T["gap hi"])],
        "kept - vetoed, R [90%]": [
            f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(p) else "-"
            for p, lo, hi in zip(T["gap R"], T["gap R lo"], T["gap R hi"])]})
    print(D.to_string(index=False))
    if (T["vetoed days"] < MIN_SHOW).any():
        print(f"  '-' = fewer than {MIN_SHOW} vetoed days, too few to measure")


def choose(T):
    notes, best = [], None
    for _, r in T.iterrows():
        share = r["vetoed %"] / 100
        if not VETO_SHARE[0] <= share <= VETO_SHARE[1]:
            why = (f"vetoes {share:.1%} - outside {VETO_SHARE[0]:.1%}-"
                   f"{VETO_SHARE[1]:.0%}, not eligible")
        elif not np.isfinite(r["gap pp"]):
            why = "too few vetoed days to measure - not eligible"
        elif not r["gap pp"] > 0:
            why = "vetoed days did not do worse - not eligible"
        else:
            lo = r["gap lo"] if np.isfinite(r["gap lo"]) else -np.inf
            why = f"eligible: lower bound {r['gap lo']:+.2f} pp"
            if best is None or lo > best[1]:
                best = (r["Veto"], lo)
        notes.append(f"    {r['Veto']:<52} {why}")
    return (best[0] if best else None), notes


def preregister(out_dir, n_boot):
    settings = {"cands": list(CANDIDATES), "dil": DILUTION, "rw": RUNWAY,
                "share": VETO_SHARE, "gate": MIN_VETOED, "show": MIN_SHOW,
                "years": YEARS,
                "n_boot": n_boot, "panel": F97.VERSION,
                "composite": V98.COMPONENTS}
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
    when = {"fresh": ", before any V99 result was computed",
            "changed": ", before any V99 result was computed (REPLACES an "
                       "earlier pre-registration)",
            "changed-after-result": " - REPLACES an earlier pre-registration "
                                    "AFTER a V99 result was seen "
                                    "(exploratory)"}[status]
    txt = PREREG.format(now=now, replaces=when,
                        dil=DILUTION, rw=RUNWAY, ver=F97.VERSION, y0=YEARS[0],
                        y1=YEARS[-1], lo=VETO_SHARE[0], hi=VETO_SHARE[1],
                        gate=MIN_VETOED, nb=n_boot)
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
# RUNNER
# =============================================================================
def run_v99(model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
            out_dir=OUT_DIR, verbose=False):
    t0 = time.time()
    W = 118
    print("=" * W)
    print("V99 - A RED-FLAG VETO: block entry only on clear warning signs "
          "(pre-registered)")
    print("=" * W)
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    print("  reading V86's data and walk-forward models (cached) ...")
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    print(f"  V97's panel ({F97.VERSION}), V98's quality score, volatility ...")
    P = prepare_panel(core_cache, big_cache)

    liq = fresh["liquid"].to_numpy(bool)
    yrs_f = fresh["year"].to_numpy()
    ever_liquid = set(fresh.loc[liq & np.isin(yrs_f, YEARS), "ticker"])
    never = ~liq & ~fresh["ticker"].isin(ever_liquid).to_numpy()
    unis = {}
    for name, frame, mask, tag, grp in (
            ("choose: unseen, liquid", fresh, liq, "fresh", "unseen"),
            ("confirm: unseen, never liquid", fresh, never, "fresh", "unseen"),
            (f"core {n_core}", core, np.ones(len(core), bool), "core",
             "core")):
        U, w = V87.universe(frame, mask, wf, tag)
        fires = V85.model_fire(U, *w["combo"])
        J = attach(U, P, grp)
        unis[name] = (U, fires, J)
    CH, CF, CO = list(unis)

    print("\n  PRE-REGISTRATION")
    when, prereg_ok = preregister(out_dir, n_boot)

    # ---- [0] who is where (counts only) ---------------------------------------
    print("\n" + "-" * W)
    print("  [0] THE TWO DATA SETS - counts only, no outcomes read")
    print("-" * W)
    rows = []
    for name in (CH, CF):
        U, fires, J = unis[name]
        yr = np.isin(U["year"].to_numpy(), YEARS)
        base = ~J["financial"].to_numpy() & yr
        rec = {"data": name, "stocks": f"{U.loc[base, 'ticker'].nunique():,}",
               "candidate days": f"{int(base.sum()):,}"}
        for cname, fn in CANDIDATES.items():
            f = base & np.nan_to_num(fn(J), nan=0).astype(bool)
            rec[cname.split(" ")[0]] = f"{int(f.sum()):,} ({f.sum() / max(base.sum(), 1):.1%})"
        rows.append(rec)
    C = pd.DataFrame(rows)
    print(C.to_string(index=False))
    C.to_csv(os.path.join(out_dir, "counts.csv"), index=False)
    def names_in_years(name):
        Ux = unis[name][0]
        return set(Ux.loc[np.isin(Ux["year"].to_numpy(), YEARS), "ticker"])
    print(f"  the two sets share no stocks in {YEARS[0]}-{YEARS[-1]}: "
          f"{not (names_in_years(CH) & names_in_years(CF))}")

    # ---- [1] choose ---------------------------------------------------------------
    U, fires, J = unis[CH]
    ind = V93.outcome_indicators(U)
    B = V98.Boot(U["date"], n_boot)
    base = ~J["financial"].to_numpy()
    T_ch, _ = veto_table(B, ind, J, base, YEARS)
    show_vetoes(T_ch, f"[1] CHOOSE - {CH}, {YEARS[0]}-{YEARS[-1]} (seen in "
                      f"V98; used only to choose)")
    T_ch.to_csv(os.path.join(out_dir, "choose.csv"), index=False)
    chosen, notes = choose(T_ch)
    print("\n  choice rule:")
    print("\n".join(notes))
    print(f"\n  chosen: {chosen or 'none'}")

    # ---- [2] confirm ----------------------------------------------------------------
    lines, ran = [], False
    if chosen is None:
        lines.append("  VERDICT: NOT ADOPTED - no red flag separated stocks on "
                     "the choice data. The confirmation outcomes were not read.")
    else:
        U2, fires2, J2 = unis[CF]
        yr2 = np.isin(U2["year"].to_numpy(), YEARS)
        base2 = ~J2["financial"].to_numpy()
        n_v = int((base2 & yr2 & np.nan_to_num(CANDIDATES[chosen](J2), nan=0)
                   .astype(bool)).sum())
        print(f"\n  power gate: '{chosen}' vetoes {n_v:,} candidate days in the "
              f"confirmation data (needs {MIN_VETOED:,}) -> "
              f"{'OK' if n_v >= MIN_VETOED else 'TOO FEW'}")
        if n_v < MIN_VETOED:
            lines.append(f"  VERDICT: INCONCLUSIVE - only {n_v:,} vetoed "
                         "candidate days; the confirmation outcomes were not "
                         "read.")
        else:
            ran = True
            ind2 = V93.outcome_indicators(U2)
            B2 = V98.Boot(U2["date"], n_boot)
            T_cf, flags2 = veto_table(B2, ind2, J2, base2, YEARS)
            show_vetoes(T_cf, f"[2] CONFIRM - {CF}, {YEARS[0]}-{YEARS[-1]} "
                              f"(all shown; only '{chosen.split(' ')[0]}' is "
                              f"the claim)")
            T_cf.to_csv(os.path.join(out_dir, "confirm.csv"), index=False)
            r = T_cf[T_cf["Veto"] == chosen].iloc[0]
            f = flags2[chosen]
            pv, _ = V98.vol_neutral(B2, ind2["tgt"], base2 & ~f, f,
                                    J2["vt2"].to_numpy(float), YEARS)
            c1, c2 = r["gap lo"] > 0, r["gap R lo"] > 0
            c3 = bool(np.isfinite(pv) and pv > 0)
            pf = lambda c: "PASS" if c else "FAIL"
            lines += [
                f"  chosen on {CH}: {chosen}",
                f"  confirmed on {CF} ({int(r['vetoed days']):,} vetoed days, "
                f"{r['vetoed %']:.1f}%)",
                f"  1. kept minus vetoed, target hit rate: {r['gap pp']:+.1f} pp "
                f"[{r['gap lo']:+.1f}, {r['gap hi']:+.1f}]  -> {pf(c1)}",
                f"  2. kept minus vetoed, average R: {r['gap R']:+.3f} "
                f"[{r['gap R lo']:+.3f}, {r['gap R hi']:+.3f}]  -> {pf(c2)}",
                f"  3. inside volatility thirds: {pv * 100:+.1f} pp  -> "
                f"{pf(c3)}",
                "\n  VERDICT: " + (
                    f"CONFIRMED - adopt '{chosen}' as Pillar 2's fundamental "
                    "veto (it replaces your friend's health <= 4 veto)."
                    if c1 and c2 and c3 else
                    "NOT CONFIRMED - no fundamental veto is adopted; keep the "
                    "flags as information only.")]

            # ---- your friend's veto on the confirmation data ------------------
            fh = J2["fr_health"].to_numpy(float)
            fv = base2 & np.isfinite(fh) & (fh <= FRIEND_VETO)
            ph, dh, _, nvh = B2.diff(ind2["tgt"], base2 & ~fv, fv, YEARS)
            po, do, _, _ = B2.diff(ind2["tgt"], base2 & ~f, f, YEARS)
            his = (f"{V98.fmt(ph, dh, 100)} pp" if nvh >= MIN_SHOW else
                   "too few vetoed days to measure")
            print(f"\n  your friend's veto (health <= {FRIEND_VETO:g}) on the same "
                  f"data: vetoes {nvh:,} days; kept minus vetoed {his}")
            if nvh >= MIN_SHOW:
                print(f"  chosen veto minus his, paired by month: "
                      f"{V98.fmt(po - ph, do - dh, 100)} pp")
            lines.append(f"  (your friend's veto on the same data, {nvh:,} "
                         f"vetoed days: {his} - reported, not required)")

    if chosen is not None:
        # ---- V88's ENTER signals with the veto -------------------------------------
        for name, label in ((CH, "seen in V98"), (CF, "confirmation data")):
            if name == CF and not ran:
                continue
            Ux, fx, Jx = unis[name]
            st = V93.Stats(Ux, n_boot)
            bx = ~Jx["financial"].to_numpy()
            valid = fx & bx
            fl = np.nan_to_num(CANDIDATES[chosen](Jx), nan=0).astype(bool)
            fh = Jx["fr_health"].to_numpy(float)
            M = {V94.F0: valid,
                 f"X {chosen.split(' ', 1)[1].strip()} vetoed": valid & ~fl,
                 "V friend's veto (health <= 4)":
                     valid & ~(np.isfinite(fh) & (fh <= FRIEND_VETO))}
            Tv = V94.table(st, valid, M, YEARS)
            V94.show(Tv, f"[3] WITH V88 - ENTER signals, {name} ({label})")
            Tv.to_csv(os.path.join(out_dir, f"with_v88_{name.split(':')[0]}"
                                            f".csv"), index=False)
        # ---- core, descriptive --------------------------------------------------------
        Uc, fc, Jc = unis[CO]
        bc = ~Jc["financial"].to_numpy()
        Tc, _ = veto_table(V98.Boot(Uc["date"], n_boot),
                           V93.outcome_indicators(Uc), Jc, bc, YEARS,
                           names=[chosen])
        show_vetoes(Tc, f"[4] SECONDARY - {CO} (hand-picked stocks; "
                        f"descriptive)")
        Tc.to_csv(os.path.join(out_dir, "core.csv"), index=False)

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
    RUN_OUT_DIR   = "thesis_tables_v99"
    # -------------------------------------------------------------------------

    run_v99(big_cache=RUN_BIG_CACHE, chunk=RUN_CHUNK, n_boot=RUN_N_BOOT,
            out_dir=RUN_OUT_DIR)
