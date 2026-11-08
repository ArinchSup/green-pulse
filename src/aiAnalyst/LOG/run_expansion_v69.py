#!/usr/bin/env python3
"""
run_expansion_v69.py - the whole expansion, one run, resumable.

WHAT IT DOES

Four steps in order, each skippable, each checkpointed so a crash in step 3 does
not cost you steps 1 and 2:

    1 audit      run V68's quality gates over the cache you already have. Two
                 minutes, and it is the only way to find out whether anything
                 already in there is bad. On this project's fixture cache it
                 refused 62 of 62 frames.
    2 pit        build the point-in-time universe from SEC XBRL frames. ~70
                 requests, cached to disk, so a rerun is free.
    3 download   fetch daily OHLCV for the expanded list. Names that fail are
                 RECORDED with a reason - that list is the delisting estimate
                 V58 needs, not an error log to ignore.
    4 filters    re-run V67 on the NEW cache and put it beside the OLD result,
                 so the only question that matters gets a direct answer.

THE ONLY QUESTION THAT MATTERS

V67 on 337 names gave the `spaced` rule a clustered interval of [42.8%, 51.2%] -
8.4 points wide, just clear of the 42.1% blind rate, on n_eff 547. The expansion
is worth doing if and only if that interval TIGHTENS.

    it tightens      -> the sniper is established and can be called a buy signal
    it stays wide    -> the effect is smaller than it looks, and the honest
                        label stays "favourable setup"

Stated before the run, because a prediction made afterwards is not a prediction:
337 to ~1,200 names is 3.6x the rows and NOT 3.6x the evidence. Large US equities
share a market factor. Expect n_eff to grow 1.5-2x, and expect most of that from
the names least like the ones already cached.

REUSING THE OLD RESULT

Step 4 does not re-run V67 on the old cache - it reads the numbers out of the
filters_v67.json that run already wrote. Re-deriving them would cost an hour and
change nothing. Point OLD_RESULT at that file, or set RERUN_OLD to compute it.

USAGE
  python run_expansion_v69.py                 # all four steps
  python run_expansion_v69.py --steps audit   # just one
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
OLD_CACHE = "price_cache_v43"
NEW_CACHE = "price_cache_v68"
PIT_FILE = "pit_universe.csv.gz"
OLD_RESULT = "filters_v67.json"
NEW_RESULT = "filters_v69.json"
STATE_FILE = "expansion_v69_state.json"

TARGET_NAMES = 1200
MIN_ASSETS_USD = 3e8
EMAIL = "you@example.com"          # SEC wants a real contact; it throttles without

N_RANDOM = 30                      # matched random draws; 8 was too few to clear 5%
BASE_QUANTILE = 0.10               # the V66 cut V67 builds on
FEATURE_SET = "no_confirm"

STEPS = ("audit", "pit", "download", "filters")

# Where the candidate list comes from.
#   "listing" Nasdaq Trader's free symbol directory. No key, no email, no SEC.
#             ~6,000 common stocks; liquidity and the quality gates cut it down.
#   "sec"     SEC XBRL frames, point-in-time by total assets. Needs SEC access.
#   "prices"  rank the cache you already have. Weakest - survivorship by design.
UNIVERSE_SOURCE = "listing"

# the figure to beat, from the 337-name run
REFERENCE = {"filter": "spaced", "n_eff": 547, "lo": 0.428, "hi": 0.512,
             "win_rate": 0.466, "blind": 0.421, "n_names": 337}


# =============================================================================
# STATE
# =============================================================================
def load_state(path=STATE_FILE):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "started": time.time()}


def save_state(st, path=STATE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, default=str)


def banner(n, name, note=""):
    print("\n" + "=" * 92)
    print(f"  STEP {n}/4 - {name.upper()}" + (f"   {note}" if note else ""))
    print("=" * 92)


# =============================================================================
# STEPS
# =============================================================================
def step_audit(old_cache, verbose=True):
    import build_universe_v68 as B
    if not os.path.isdir(old_cache):
        print(f"  {old_cache} not found - nothing to audit, skipping")
        return {"skipped": True}
    bad = B.audit_cache(old_cache, verbose=verbose)
    n = len([f for f in os.listdir(old_cache) if f.endswith(".pkl")])
    if bad and len(bad) == n:
        print(f"\n  EVERY frame was refused. Do not build on this cache - "
              f"whatever is in it is not")
        print(f"  market data. Fix that before step 3, or the expansion just "
              f"adds more of the same.")
    elif bad:
        print(f"\n  {len(bad)} of {n} frames would be refused. They are "
              f"excluded automatically downstream,")
        print(f"  but check the reasons: a cluster of one reason usually means a "
              f"source problem.")
    else:
        print(f"\n  all {n} frames pass. The cache is sound.")
    return {"n_frames": n, "n_refused": len(bad),
            "refused": [{"ticker": t, "reason": w} for t, w in bad[:200]]}


def step_pit(pit_file, target, min_assets, email, source=UNIVERSE_SOURCE,
             old_cache=OLD_CACHE, verbose=True):
    import build_universe_v68 as B
    B.EMAIL = email

    if source == "listing":
        syms = B.universe_from_listing(verbose=verbose)
        print(f"\n  {len(syms):,} candidates from the exchange listing. The "
              f"download step's quality")
        print(f"  gates and a liquidity cut take it to roughly {target:,} "
              f"usable names.")
        print(f"\n  NOTE what this does and does not fix: it removes the "
              f"'rank by today's size'")
        print(f"  problem, but delisted companies are still absent - no free "
              f"price source keeps")
        print(f"  them. That bias is measured by V58, not cured here.")
        return {"n_universe": len(syms), "tickers": syms, "source": source}

    if source == "prices":
        u = B.universe_from_prices(old_cache, target, verbose)
        return {"n_universe": int(len(u)), "source": source,
                "tickers": sorted(u["ticker"].astype(str))}

    ok, why = B.check_sec(email, verbose=verbose)
    if not ok:
        print(f"\n  SEC is not reachable ({why}). Falling back to the exchange "
              f"listing, which needs")
        print(f"  no SEC access. Set UNIVERSE_SOURCE = 'listing' to skip this "
              f"probe next time.")
        syms = B.universe_from_listing(verbose=verbose)
        return {"n_universe": len(syms), "tickers": syms,
                "source": "listing (sec unavailable)"}
    if os.path.exists(pit_file):
        df = pd.read_csv(pit_file)
        print(f"  {pit_file} already present: {len(df):,} rows, "
              f"{df['cik'].nunique():,} companies - reusing it")
    else:
        df = B.build_pit_universe(out=pit_file, email=email, verbose=verbose)
    u = B.point_in_time_universe(pit_file, min_assets, target, verbose=verbose,
                                 build=False)
    cmap = B.sec_ticker_map(email=email)
    u["ticker"] = u["cik"].map(cmap)
    unmapped = int(u["ticker"].isna().sum())
    u = u.dropna(subset=["ticker"]).drop_duplicates("ticker")
    u.to_csv("universe_v69.csv", index=False)
    print(f"\n  {len(u):,} tickers written to universe_v69.csv")
    print(f"  {unmapped:,} CIKs had no current ticker - most of those are "
          f"DELISTED companies,")
    print(f"  which is the population a survivor-only cache is missing.")
    return {"n_universe": int(len(u)), "n_unmapped": unmapped,
            "tickers": sorted(u["ticker"].astype(str))}


def step_download(tickers, new_cache, verbose=True):
    import build_universe_v68 as B
    kept, failed = B.download(tickers, out_cache=new_cache, verbose=verbose)
    from collections import Counter
    print(f"\n  cached {len(kept):,} | refused {len(failed):,}")
    for reason, k in Counter(w.split(" (")[0] for _, w in failed).most_common():
        print(f"    {reason:26s} {k:5d}")
    if len(kept) < 400:
        print(f"\n  only {len(kept):,} names made it. That is not enough to "
              f"move n_eff much - check the")
        print(f"  refusal reasons above before running step 4.")
    return {"n_kept": len(kept), "n_failed": len(failed),
            "failed": [{"ticker": t, "reason": w} for t, w in failed[:400]]}


def _spaced_row(rows):
    for r in rows:
        if r.get("filter") == "spaced" and r.get("n_eff"):
            return r
    return None


def step_filters(new_cache, old_result, n_random, quantile, feature_set,
                 out=NEW_RESULT, rerun_old=False, verbose=True):
    import evaluate_filters_v67 as F67

    print(f"  running V67 on {new_cache} with {n_random} random draws")
    rows, _ = F67.run_filters_v67(price_cache=new_cache, quantile=quantile,
                                  feature_set=feature_set, n_random=n_random,
                                  out=out, verbose=verbose)
    new_base = rows[0]
    new_spaced = _spaced_row(rows)

    old_base, old_spaced = None, None
    if rerun_old:
        print(f"\n  re-running V67 on {OLD_CACHE} for a like-for-like baseline")
        orows, _ = F67.run_filters_v67(price_cache=OLD_CACHE,
                                       quantile=quantile,
                                       feature_set=feature_set,
                                       n_random=n_random,
                                       out="filters_v69_old.json",
                                       verbose=False)
        old_base, old_spaced = orows[0], _spaced_row(orows)
    elif os.path.exists(old_result):
        with open(old_result, encoding="utf-8") as f:
            prev = json.load(f)
        old_base = prev.get("base")
        old_spaced = _spaced_row(prev.get("filters", []))

    return {"new_base": new_base, "new_spaced": new_spaced,
            "old_base": old_base, "old_spaced": old_spaced}


# =============================================================================
# THE COMPARISON
# =============================================================================
def compare(res, n_new_names, reference=REFERENCE):
    print("\n" + "=" * 92)
    print("  DID MORE NAMES TIGHTEN THE INTERVAL?  (the only question)")
    print("=" * 92)

    old_s = res.get("old_spaced")
    new_s = res.get("new_spaced")
    if new_s is None:
        print("  the `spaced` rule did not produce a scorable set on the new "
              "cache - nothing to compare")
        return

    def line(tag, r, names):
        if r is None:
            print(f"  {tag:22s} (not available)")
            return None
        w = (r["wr_hi"] - r["wr_lo"]) * 100
        print(f"  {tag:22s} {names:>6} names  n={r['n']:>7,d}  "
              f"n_eff={r['n_eff']:>6,.0f}  {r['win_rate']:6.1%}  "
              f"[{r['wr_lo']:.1%}, {r['wr_hi']:.1%}]  width {w:4.1f}pp")
        return w

    ref_w = (reference["hi"] - reference["lo"]) * 100
    print(f"  {'reference (V67 run)':22s} {reference['n_names']:>6} names  "
          f"n_eff={reference['n_eff']:>6,.0f}  {reference['win_rate']:6.1%}  "
          f"[{reference['lo']:.1%}, {reference['hi']:.1%}]  width {ref_w:4.1f}pp")
    old_w = line("BEFORE (from json)", old_s, reference["n_names"])
    new_w = line("AFTER", new_s, n_new_names)

    base_w = old_w if old_w is not None else ref_w
    blind = (res.get("new_base") or {}).get("win_rate")
    print()
    if blind is not None:
        print(f"  blind win rate on the new cache: {blind:.1%} "
              f"(it will move - a wider universe is a different population)")

    grew = new_s["n_eff"] / (old_s["n_eff"] if old_s else reference["n_eff"])
    print(f"  n_eff changed by {grew:.2f}x on {n_new_names / reference['n_names']:.1f}x "
          f"the names. Predicted 1.5-2.0x.")
    if grew < 1.3:
        print(f"    -> the extra names added little independent information. "
              f"They are probably too")
        print(f"       similar to the ones you had: same size band, same "
              f"sectors, same market factor.")

    tightened = new_w < base_w - 0.5
    clears = (blind is not None and new_s["wr_lo"] > blind)
    print()
    if tightened and clears:
        print(f"  -> ESTABLISHED. The interval tightened from {base_w:.1f}pp to "
              f"{new_w:.1f}pp and its lower")
        print(f"     bound ({new_s['wr_lo']:.1%}) is above the blind rate "
              f"({blind:.1%}).")
        print(f"     This is a buy signal you can defend. Size it on the lower "
              f"bound, not the point")
        print(f"     estimate, and re-check it under the rolling split before "
              f"anyone trades it.")
    elif clears:
        print(f"  -> CLEARS BUT DOES NOT TIGHTEN. Lower bound "
              f"{new_s['wr_lo']:.1%} beats the blind "
              f"{blind:.1%},")
        print(f"     but the interval is still {new_w:.1f}pp wide. The effect is "
              f"real and small. Ship it as")
        print(f"     a 'favourable setup' tag, not as a buy signal.")
    else:
        print(f"  -> NOT ESTABLISHED. The lower bound "
              f"({new_s['wr_lo']:.1%}) does not clear the blind rate on the")
        print(f"     wider universe. More data made the estimate more honest, "
              f"not better. The 337-name")
        print(f"     result was optimistic, and this is the number to believe.")
    print(f"\n  Whatever the verdict: the interval on 1,200 names is the one to "
          f"quote. It is built on")
    print(f"  more independent evidence than the 337-name version, in both "
          f"directions.")


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_expansion(steps=STEPS, old_cache=OLD_CACHE, new_cache=NEW_CACHE,
                  pit_file=PIT_FILE, target=TARGET_NAMES,
                  min_assets=MIN_ASSETS_USD, email=EMAIL, n_random=N_RANDOM,
                  quantile=BASE_QUANTILE, feature_set=FEATURE_SET,
                  old_result=OLD_RESULT, rerun_old=False, fresh=False,
                  universe_source=UNIVERSE_SOURCE, state_file=STATE_FILE,
                  verbose=True):
    print("=" * 92)
    print("V69 - EXPAND THE UNIVERSE AND RE-TEST THE SNIPER")
    print("=" * 92)
    print(f"  steps: {', '.join(steps)}")
    print(f"  {old_cache} -> {new_cache}, aiming for {target:,} names")

    st = {"done": [], "started": time.time()} if fresh else load_state(state_file)
    if st.get("done"):
        print(f"  resuming; already done: {', '.join(st['done'])}")

    t0 = time.time()
    if "audit" in steps:
        banner(1, "audit the existing cache", f"({old_cache})")
        st["audit"] = step_audit(old_cache, verbose)
        st["done"] = sorted(set(st["done"] + ["audit"]))
        save_state(st, state_file)

    if "pit" in steps:
        banner(2, "candidate universe", f"(source: {universe_source})")
        if universe_source == "sec" and email == "you@example.com":
            print("  REFUSING: 'sec' source needs a real contact address in "
                  "EMAIL. Or set")
            print("  UNIVERSE_SOURCE = 'listing', which needs no email at all.")
            return st
        st["pit"] = step_pit(pit_file, target, min_assets, email,
                             source=universe_source, old_cache=old_cache,
                             verbose=verbose)
        st["done"] = sorted(set(st["done"] + ["pit"]))
        save_state(st, state_file)

    if "download" in steps:
        banner(3, "download prices", f"(-> {new_cache})")
        tickers = (st.get("pit") or {}).get("tickers")
        if not tickers:
            if os.path.exists("universe_v69.csv"):
                tickers = sorted(pd.read_csv("universe_v69.csv")["ticker"]
                                 .astype(str))
                print(f"  reusing universe_v69.csv ({len(tickers):,} names)")
            else:
                print("  no ticker list - run the pit step first")
                return st
        st["download"] = step_download(tickers, new_cache, verbose)
        st["done"] = sorted(set(st["done"] + ["download"]))
        save_state(st, state_file)

    if "filters" in steps:
        banner(4, "re-test the sniper", f"({n_random} random draws)")
        if not os.path.isdir(new_cache):
            print(f"  {new_cache} does not exist - run the download step first")
            return st
        n_names = len([f for f in os.listdir(new_cache) if f.endswith(".pkl")])
        print(f"  {n_names:,} names in {new_cache}")
        res = step_filters(new_cache, old_result, n_random, quantile,
                           feature_set, rerun_old=rerun_old, verbose=verbose)
        compare(res, n_names)
        st["filters"] = {k: v for k, v in res.items()}
        st["done"] = sorted(set(st["done"] + ["filters"]))
        save_state(st, state_file)

    mins = (time.time() - t0) / 60
    print(f"\n  finished {', '.join(steps)} in {mins:.1f} min | state in "
          f"{state_file}")
    return st


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=list(STEPS), choices=STEPS)
    ap.add_argument("--old-cache", default=OLD_CACHE)
    ap.add_argument("--new-cache", default=NEW_CACHE)
    ap.add_argument("--pit-file", default=PIT_FILE)
    ap.add_argument("--target", type=int, default=TARGET_NAMES)
    ap.add_argument("--min-assets", type=float, default=MIN_ASSETS_USD)
    ap.add_argument("--email", default=EMAIL)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--quantile", type=float, default=BASE_QUANTILE)
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--old-result", default=OLD_RESULT)
    ap.add_argument("--rerun-old", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the checkpoint and start over")
    ap.add_argument("--universe-source", default=UNIVERSE_SOURCE,
                    choices=["listing", "sec", "prices"])
    ap.add_argument("--check-sec", action="store_true",
                    help="one request to SEC, then exit, to diagnose a 403")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    if c.check_sec:
        import build_universe_v68 as B
        ok, why = B.check_sec(c.email)
        print(f"\n  verdict: {'SEC works' if ok else why}")
        return 0 if ok else 1
    run_expansion(tuple(c.steps), c.old_cache, c.new_cache, c.pit_file,
                  c.target, c.min_assets, c.email, c.n_random, c.quantile,
                  c.feature_set, c.old_result, c.rerun_old, c.fresh,
                  c.universe_source)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # Drop steps you have already done - the checkpoint also skips them, but
    # being explicit is clearer than trusting a state file.
    RUN_STEPS       = ("audit", "pit", "download", "filters")

    RUN_OLD_CACHE   = "price_cache_v43"      # the 337-name cache to audit
    RUN_NEW_CACHE   = "price_cache_v68"      # where the expanded one goes
    RUN_PIT_FILE    = "pit_universe.csv.gz"
    RUN_TARGET      = 1200
    RUN_MIN_ASSETS  = 3e8                    # size floor to enter the universe

    # WHERE THE CANDIDATE LIST COMES FROM
    #   "listing" Nasdaq Trader's free symbol directory - no key, no email, no
    #             SEC. ~6,000 common stocks. This is the default because SEC
    #             returns 403 on some networks and the email is never the cause.
    #   "sec"     SEC XBRL frames, point-in-time by total assets. Needs access.
    #   "prices"  rank the cache you already have. Survivorship by design.
    RUN_UNIVERSE_SRC = "listing"

    # Only used when RUN_UNIVERSE_SRC is "sec". Any real address works - SEC
    # does not validate it, and a gmail is fine.
    RUN_EMAIL       = "you@example.com"

    RUN_N_RANDOM    = 30        # 8 draws could not clear 5%; 30 can
    RUN_QUANTILE    = 0.10      # the V66 cut
    RUN_FEATURE_SET = "no_confirm"

    RUN_OLD_RESULT  = "filters_v67.json"   # reuse the 337-name numbers
    RUN_RERUN_OLD   = False     # True recomputes them instead (adds ~1 hour)
    RUN_FRESH       = False     # True ignores the checkpoint
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_expansion(steps=RUN_STEPS, old_cache=RUN_OLD_CACHE,
                      new_cache=RUN_NEW_CACHE, pit_file=RUN_PIT_FILE,
                      target=RUN_TARGET, min_assets=RUN_MIN_ASSETS,
                      email=RUN_EMAIL, n_random=RUN_N_RANDOM,
                      quantile=RUN_QUANTILE, feature_set=RUN_FEATURE_SET,
                      old_result=RUN_OLD_RESULT, rerun_old=RUN_RERUN_OLD,
                      fresh=RUN_FRESH, universe_source=RUN_UNIVERSE_SRC)
