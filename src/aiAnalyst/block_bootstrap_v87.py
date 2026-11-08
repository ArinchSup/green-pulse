"""
V87 - DOES V86'S WIN SURVIVE OVERLAPPING TRADES?

WHY THIS CHECK
--------------
V86's interval resampled calendar MONTHS as if they were independent. They are
not quite: a trade can run 60 trading days (~3 months), so a signal in March
and one in May can share the same weeks. If neighbouring months move together,
the month bootstrap is too narrow. The combo's lead over `low bb_position` on
unseen names cleared zero by only +0.34 pp after the two-shot correction, so a
modest widening could erase it.

WHAT THIS FILE DOES
-------------------
  CHECK 1  CIRCULAR BLOCK BOOTSTRAP. Instead of single months, it resamples
           runs of consecutive months - 3 (one holding period, a quarter),
           6, and 12 (a year) - wrapping around the end so every month is
           equally likely. Whatever dependence exists inside a block is kept
           in every draw, so the interval widens exactly as much as the
           dependence requires. Block 1 is V86's month bootstrap again, as a
           reference (the same method, different random draws).
  CHECK 2  THE CLEAN SLICE. Fresh names in 2022-2026 are the only rows that
           are out of universe AND out of the period the combo was chosen on
           (V85 picked it on core 2017-2021, and fresh names in those months
           share the same market).
  DIAGNOSTIC  the autocorrelation of the month-by-month difference between
           the arms. If it is near zero, the blocks will barely move the
           interval - and that is a finding, not a failure of the check.

Both arms always get the same resampled months, so every comparison stays
paired. Nothing is retrained: V86's cached data and fold models are read back,
and the fire counts and point estimates must match V86's tables exactly.

PRE-REGISTRATION (written before any block result is computed)
--------------------------------------------------------------
These checks were designed after V86's month-bootstrap result was known. The
decision below is fixed before the block results are, and is written to
thesis_tables_v87/preregistration.txt on the first run.

  CHECK 1 PASSES if the combo's Bonferroni (k=2) lower bound vs
          low bb_position, fresh liquid names, 2017-2026, stays above zero
          with BOTH 3-month and 6-month blocks. 12-month blocks are shown but
          do not decide: ten years give only ~10 independent blocks, too few
          for a reliable 2.5th percentile.
  CHECK 2 PASSES if the combo's point estimate on fresh liquid names,
          2022-2026, is above zero (3-month blocks). It is CLEAN if the lower
          90% bound is above zero too. Half the months cannot be expected to
          clear zero on their own, so "clean" is a bonus, not a requirement.

  CONFIRMED (clean)  both pass and check 2 is clean -> promote the combo
  CONFIRMED          both pass                      -> promote the combo
  NOT CONFIRMED      either fails                   -> keep V81; report the
                                                       combo as suggestive

V81 is reported with the same blocks for completeness; its pre-registered
verdict was set in V86 and is not re-decided here.

COST: a few minutes - everything is read from V86's caches (oou_cache_v86/).
Keep RUN_CHUNK equal to the value V86 ran with, or the fresh-name cache misses
and the build starts again (~15 min).
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

import class_ai_entry_model_v81 as M81
import measure_v83 as V83
import sweep_options_v85 as V85
import out_of_universe_v86 as V86

warnings.filterwarnings("ignore", message="Mean of empty slice")

SEED = 87
OUT_DIR = "thesis_tables_v87"
V86_DIR = "thesis_tables_v86"
OPPONENT = V86.OPPONENT
COMBO = V86.COMBO_NAME
HELD_YEARS = V86.HELD_YEARS
TEST_YEARS = V86.TEST_YEARS
DECIDING_BLOCKS = (3, 6)
ALL_BLOCKS = (1, 3, 6, 12)
K_BONF = 2
PRIMARY = "fresh, liquid (PRIMARY)"


# =============================================================================
# 1. READ V86 BACK FROM ITS CACHES
# =============================================================================
def prepare(model_in, core_cache, big_cache, n_seeds, chunk, liq_pct,
            dup_corr, verbose=True):
    """V86's steps [1], [2] and [4], with V86's exact cache keys."""
    m81 = M81.load_model(model_in)
    prov = m81["provenance"]
    core_cache = core_cache or prov["price_cache"]
    if not os.path.isdir(big_cache):
        raise RuntimeError(f"{big_cache} not found")
    core_set = set(V86.cache_names(core_cache))
    big = V86.cache_names(big_cache)
    cand = [t for t in big if t not in core_set]
    dkey = V86._hash(["dups", core_cache, sorted(core_set), big_cache, cand,
                      dup_corr, V86.DUP_MIN_OVERLAP])
    dpath = os.path.join(V86.CACHE_DIR, f"dups_{dkey}.pkl")
    if os.path.exists(dpath):
        dups = pd.read_pickle(dpath)
    else:
        print("      (duplicate check not cached - recomputing)")
        os.makedirs(V86.CACHE_DIR, exist_ok=True)
        dups = V86.find_duplicates(core_cache, sorted(core_set), big_cache,
                                   cand, thr=dup_corr)
        dups.to_pickle(dpath)
    drop = set(dups["fresh"])
    fresh_names = [t for t in cand if t not in drop]

    core, core_names = V86.build_core(prov, core_cache, verbose=verbose)
    fresh = V86.build_fresh(big_cache, fresh_names, prov, chunk=chunk,
                            verbose=verbose)
    fresh["liq_floor"] = V86.liquidity_floor(core, fresh["date"], liq_pct)
    fresh["liquid"] = (fresh["log_dollar_vol"].to_numpy(float)
                       >= fresh["liq_floor"].to_numpy(float))
    years_all = sorted(core["year"].unique())
    fold_years = years_all[int(prov["min_train_years"]):]
    sig = V86._hash([len(core), core_names, len(fresh), fresh_names,
                     str(fresh["date"].max())])
    wf = {k: V86.walk_forward_oou(c, core, fresh, fold_years, n_seeds,
                                  prov["horizon"], k, sig, verbose=verbose)
          for k, c in (("V81", V86.V81_CFG), ("combo", V86.COMBO_CFG))}
    return core, fresh, wf, len(core_set)


def universe(frame, mask, wf, tag):
    pos = np.full(len(frame), -1, int)
    pos[mask] = np.arange(int(mask.sum()))
    U = frame[mask].reset_index(drop=True).copy()
    U["row_id"] = np.arange(len(U))
    return U, {k: V86.restrict(*v[tag], pos) for k, v in wf.items()}


def fires_for(U, w):
    s = -U["bb_position"].to_numpy(float)
    return {"V81": V85.model_fire(U, *w["V81"]),
            COMBO: V85.model_fire(U, *w["combo"]),
            OPPONENT: V83.apply_rule(s, V83.rolling_cut(U["date"], s),
                                     strict=True)}


# =============================================================================
# 2. CIRCULAR BLOCK BOOTSTRAP, PAIRED
# =============================================================================
class BlockBoot:
    """
    Excess per signal = outcome minus its own stock-month pool mean (V85's
    analytic matched control). Months are resampled in circular blocks of L
    consecutive months; both arms get the same months, so the difference is
    paired. L = 1 is the ordinary month bootstrap V86 used.
    """

    def __init__(self, U, n_boot=10000, seed=SEED):
        ev = V85.Evaluator(U, n_boot=1)
        self.ex_w, self.year, self.mcode = ev.ex_w, ev.year, ev.mcode
        self.months = ev.months
        self.n_boot, self.seed = n_boot, seed
        self._W = {}

    def _idx(self, years):
        idx = np.flatnonzero(np.isin(np.array([m.year for m in self.months]),
                                     years))
        return idx[np.argsort(self.months[idx].to_timestamp().values)]

    def weights(self, years, L):
        key = (tuple(years), L)
        if key not in self._W:
            idx = self._idx(years)
            n = len(idx)
            rng = np.random.default_rng([self.seed, L, years[0], years[-1]])
            k = int(np.ceil(n / L))
            starts = rng.integers(0, n, size=(self.n_boot, k))
            flat = ((starts[:, :, None] + np.arange(L)) % n) \
                .reshape(self.n_boot, -1)[:, :n]
            flat = flat + np.arange(self.n_boot)[:, None] * n
            W = np.bincount(flat.ravel(), minlength=self.n_boot * n) \
                .reshape(self.n_boot, n).astype(float)
            self._W[key] = (idx, W)
        return self._W[key]

    def _sums(self, fire, years, idx):
        m = fire & np.isin(self.year, years)
        k = len(self.months)
        sw = np.bincount(self.mcode[m], self.ex_w[m], k)[idx]
        cn = np.bincount(self.mcode[m], minlength=k)[idx].astype(float)
        return sw, cn

    def paired(self, fa, fb, years, L, k_tests=K_BONF):
        idx, W = self.weights(years, L)
        swa, cna = self._sums(fa, years, idx)
        swb, cnb = self._sums(fb, years, idx)
        if cna.sum() < 30 or cnb.sum() < 30:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            d = ((W @ swa) / (W @ cna) - (W @ swb) / (W @ cnb)) * 100
        a = 5.0 / k_tests
        return {"Signals": int(cna.sum()), "bb signals": int(cnb.sum()),
                "Months": len(idx),
                "Diff": (swa.sum() / cna.sum() - swb.sum() / cnb.sum()) * 100,
                "90% lo": np.nanpercentile(d, 5),
                "90% hi": np.nanpercentile(d, 95),
                "Bonf lo": np.nanpercentile(d, a),
                "Bonf hi": np.nanpercentile(d, 100 - a),
                "P(diff<=0)": float(np.nanmean(d <= 0))}

    def monthly_diff(self, fa, fb, years):
        idx = self._idx(years)
        k = len(self.months)
        out = []
        for f in (fa, fb):
            m = f & np.isin(self.year, years)
            s = np.bincount(self.mcode[m], self.ex_w[m], k)
            c = np.bincount(self.mcode[m], minlength=k).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append((s / c)[idx])
        return (out[0] - out[1]) * 100


def autocorr(x, lags=(1, 2, 3, 6)):
    x = np.asarray(x, float)
    r = {}
    for L in lags:
        a, b = x[:-L], x[L:]
        ok = np.isfinite(a) & np.isfinite(b)
        r[L] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 \
            else np.nan
    return r


# =============================================================================
# 3. PRE-REGISTRATION
# =============================================================================
PREREG = """PRE-REGISTRATION - V87 block-bootstrap check of V86
written {now}, before any block-bootstrap result was computed{replaces}.
Designed after V86's month-bootstrap result was known.

QUESTION   does the combo's lead over {opp} on fresh liquid names survive
           resampling that respects overlapping trades?
METHOD     circular block bootstrap over calendar months, {n_boot} draws, the
           same resampled months for both arms (paired). Block lengths
           {blocks} months; 1 = V86's month bootstrap, for reference.
CHECK 1    PASSES if the combo's Bonferroni (k=2) lower bound, fresh liquid,
           {h0}-{h1}, is above zero with BOTH {d0}- and {d1}-month blocks.
           12-month blocks are reported, not decisive (~10 blocks in ten years).
CHECK 2    PASSES if the combo's point estimate, fresh liquid, {t0}-{t1}, is
           above zero ({d0}-month blocks); CLEAN if its lower 90% bound is
           above zero too.
DECISION   CONFIRMED (clean) / CONFIRMED -> promote the combo to the served
           model. NOT CONFIRMED -> keep V81, report the combo as suggestive.
V81        reported with the same blocks; its verdict was set in V86.
"""


def preregister(out_dir, settings):
    sig = V86._hash(settings)
    meta_p = os.path.join(out_dir, "preregistration.json")
    txt_p = os.path.join(out_dir, "preregistration.txt")
    status = "fresh"
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            meta = json.load(fh)
        if meta.get("sig") == sig:
            print(f"  pre-registered {meta['written']} - settings unchanged "
                  f"since")
            return meta["written"], meta.get("status") != "changed-after-result"
        status = ("changed-after-result"
                  if os.path.exists(os.path.join(out_dir, "verdict.txt"))
                  else "changed")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = PREREG.format(
        now=now, replaces="" if status == "fresh"
        else " (REPLACES an earlier pre-registration)",
        opp=OPPONENT, n_boot=settings["n_boot"],
        blocks=", ".join(map(str, ALL_BLOCKS)),
        h0=HELD_YEARS[0], h1=HELD_YEARS[-1], t0=TEST_YEARS[0],
        t1=TEST_YEARS[-1], d0=DECIDING_BLOCKS[0], d1=DECIDING_BLOCKS[1])
    if os.path.exists(txt_p):
        with open(txt_p) as fh, open(os.path.join(
                out_dir, "preregistration_history.txt"), "a") as h:
            h.write(fh.read() + "\n" + "-" * 80 + "\n")
    with open(txt_p, "w") as fh:
        fh.write(text)
    with open(meta_p, "w") as fh:
        json.dump({"sig": sig, "written": now, "status": status}, fh)
    print(f"  wrote {txt_p} ({now})")
    if status == "changed-after-result":
        print("  *** the decision settings changed AFTER a result was written"
              " - this run's verdict is exploratory")
    return now, status != "changed-after-result"


# =============================================================================
# RUNNER
# =============================================================================
def run_v87(model_in="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250,
            liq_pct=V86.LIQ_PCT, dup_corr=V86.DUP_CORR, n_boot=10000,
            show_other_universes=True, out_dir=OUT_DIR, v86_dir=V86_DIR,
            verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    W_ = 104
    print("=" * W_)
    print("V87 - DOES V86'S WIN SURVIVE OVERLAPPING TRADES? "
          "(block bootstrap + the clean slice)")
    print("=" * W_)

    # ---- [1] V86 back from its caches ---------------------------------------
    print("\n  [1] READING V86 (cached data and fold models) ...")
    core, fresh, wf, n_core = prepare(model_in, core_cache, big_cache,
                                      n_seeds, chunk, liq_pct, dup_corr,
                                      verbose=verbose)
    CORE = f"core {n_core}"
    unis = {PRIMARY: universe(fresh, fresh["liquid"].to_numpy(bool), wf,
                              "fresh")}
    if show_other_universes:
        unis["fresh, all"] = universe(fresh, np.ones(len(fresh), bool), wf,
                                      "fresh")
        unis[CORE] = universe(core, np.ones(len(core), bool), wf, "core")
    fires = {n: fires_for(U, w) for n, (U, w) in unis.items()}
    boots = {n: BlockBoot(U, n_boot=n_boot) for n, (U, _) in unis.items()}

    # ---- [2] integrity: identical to V86 ----------------------------------
    print(f"\n  [2] INTEGRITY - fires and point estimates must equal V86's")
    ok = True
    side_p = os.path.join(v86_dir, "side_by_side.csv")
    arms_p = os.path.join(v86_dir, "arms_all_universes.csv")
    ref = pd.read_csv(side_p) if os.path.exists(side_p) else None
    arms = pd.read_csv(arms_p) if os.path.exists(arms_p) else None
    if ref is None:
        print(f"      {side_p} not found - cannot compare (run V86 first)")
        ok = False
    for name in unis:
        f, bb = fires[name], boots[name]
        for arm in ("V81", COMBO):
            p = bb.paired(f[arm], f[OPPONENT], HELD_YEARS, 1)
            if ref is None or p is None:
                continue
            r = ref[(ref["Universe"] == name) & (ref["Arm"] == arm)]
            if not len(r):
                continue
            r = r.iloc[0]
            same = (p["Signals"] == int(r["Signals"])
                    and abs(p["Diff"] - float(r["vs bb"])) < 1e-6)
            ok &= same
            print(f"      {name:<24} {arm:<26} signals {p['Signals']:>5,} vs "
                  f"{int(r['Signals']):>5,}   diff {p['Diff']:+.4f} vs "
                  f"{float(r['vs bb']):+.4f}   "
                  f"{'REPRODUCED' if same else '*** DIFFERENT ***'}")
        if arms is not None:
            a = arms[(arms["Universe"] == name) & (arms["Arm"] == OPPONENT)]
            if len(a):
                n_now = int((f[OPPONENT]
                             & np.isin(unis[name][0]["year"], HELD_YEARS))
                            .sum())
                same = n_now == int(a["Signals"].iloc[0])
                ok &= same
                print(f"      {name:<24} {OPPONENT:<26} signals {n_now:>5,} "
                      f"vs {int(a['Signals'].iloc[0]):>5,}   "
                      f"{'REPRODUCED' if same else '*** DIFFERENT ***'}")
    if not ok:
        print("      the data or models differ from V86's run - check "
              "RUN_CHUNK, RUN_N_SEEDS and the caches before trusting below")

    # ---- [3] pre-registration ----------------------------------------------
    print(f"\n  [3] PRE-REGISTRATION")
    settings = {"deciding": DECIDING_BLOCKS, "blocks": ALL_BLOCKS,
                "k": K_BONF, "n_boot": n_boot, "seed": SEED,
                "held": HELD_YEARS, "test": TEST_YEARS, "opp": OPPONENT,
                "arm": COMBO, "universe": PRIMARY}
    prereg_when, prereg_ok = preregister(out_dir, settings)

    # ---- [4] dependence diagnostic -------------------------------------------
    U, _ = unis[PRIMARY]
    f, bb = fires[PRIMARY], boots[PRIMARY]
    print(f"\n  [4] DO NEIGHBOURING MONTHS MOVE TOGETHER? autocorrelation of the "
          f"monthly difference vs {OPPONENT},")
    print(f"      {PRIMARY}, {HELD_YEARS[0]}-{HELD_YEARS[-1]} "
          f"(|r| above ~{2 / np.sqrt(len(bb._idx(HELD_YEARS))):.2f} would be "
          f"notable)")
    ac_rows = []
    for arm in ("V81", COMBO):
        ac = autocorr(bb.monthly_diff(f[arm], f[OPPONENT], HELD_YEARS))
        ac_rows.append({"Arm": arm, **{f"lag {k}": v for k, v in ac.items()}})
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(f"{out_dir}/autocorrelation.csv", index=False)
    print("      " + ac.to_string(index=False, float_format=lambda v:
                                   f"{v:+.3f}").replace("\n", "\n      "))

    # ---- [5] the block table ------------------------------------------------
    rows = []
    for name in unis:
        f, bb = fires[name], boots[name]
        for arm in (COMBO, "V81"):
            for tag, yrs in ((f"{HELD_YEARS[0]}-{HELD_YEARS[-1]}", HELD_YEARS),
                             (f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}", TEST_YEARS)):
                for L in ALL_BLOCKS:
                    p = bb.paired(f[arm], f[OPPONENT], yrs, L)
                    if p:
                        rows.append({"Universe": name, "Arm": arm,
                                     "Years": tag, "Block (months)": L, **p})
    T = pd.DataFrame(rows)
    T.to_csv(f"{out_dir}/block_bootstrap.csv", index=False)
    fmt = lambda v: f"{v:+.2f}"
    cols = ["Arm", "Years", "Block (months)", "Signals", "Diff", "90% lo",
            "90% hi", "Bonf lo", "P(diff<=0)"]
    print("\n" + "=" * W_)
    print(f"  [5] PAIRED DIFFERENCE vs {OPPONENT} (pp) under each block length "
          f"- {n_boot:,} draws; 'Bonf lo' = k=2")
    print("=" * W_)
    for name in unis:
        t = T[T["Universe"] == name]
        if not len(t):
            continue
        print(f"\n  {name.upper()}" + ("" if name == PRIMARY
                                       else "  (information only)"))
        t = t[cols].copy()
        t["P(diff<=0)"] = t["P(diff<=0)"].map(lambda v: f"{v:.3f}")
        print(t.to_string(index=False, float_format=fmt))
    if ref is not None:
        r = ref[(ref["Universe"] == PRIMARY) & (ref["Arm"] == COMBO)]
        if len(r):
            r = r.iloc[0]
            print(f"\n      V86's month bootstrap for the combo, for comparison "
                  f"with block 1: [{r['lo']:+.2f}, {r['hi']:+.2f}], Bonf lo "
                  f"{r['lo (Bonf k=2)']:+.2f} (different random draws, same "
                  f"method)")

    # ---- [6] verdict -------------------------------------------------------------
    def get(arm, yrs, L, col):
        t = T[(T["Universe"] == PRIMARY) & (T["Arm"] == arm)
              & (T["Years"] == yrs) & (T["Block (months)"] == L)]
        return float(t[col].iloc[0]) if len(t) else np.nan
    H = f"{HELD_YEARS[0]}-{HELD_YEARS[-1]}"
    Tt = f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}"
    c1 = {L: get(COMBO, H, L, "Bonf lo") for L in DECIDING_BLOCKS}
    pass1 = all(np.isfinite(v) and v > 0 for v in c1.values())
    d0 = DECIDING_BLOCKS[0]
    c2_diff = get(COMBO, Tt, d0, "Diff")
    c2_lo, c2_hi = get(COMBO, Tt, d0, "90% lo"), get(COMBO, Tt, d0, "90% hi")
    pass2 = np.isfinite(c2_diff) and c2_diff > 0
    clean = pass2 and np.isfinite(c2_lo) and c2_lo > 0

    lines = []
    lines.append(f"  CHECK 1  combo vs {OPPONENT}, {PRIMARY}, {H}, Bonferroni "
                 f"k=2 lower bound: "
                 + ", ".join(f"{L}-month blocks {v:+.2f}"
                             for L, v in c1.items())
                 + f"  -> {'PASSES' if pass1 else 'FAILS'}")
    lines.append(f"  CHECK 2  combo vs {OPPONENT}, {PRIMARY}, {Tt}: "
                 f"{c2_diff:+.2f} pp, 90% [{c2_lo:+.2f}, {c2_hi:+.2f}] "
                 f"({d0}-month blocks)  -> "
                 + ("PASSES, CLEAN" if clean else
                    "PASSES (interval includes zero)" if pass2 else "FAILS"))
    if pass1 and pass2:
        v = ("CONFIRMED (clean)" if clean else "CONFIRMED") + \
            (f" - the combo beats the pre-chosen rule on names it has never "
             f"seen, and the lead survives overlapping trades. Promote it to "
             f"the served model.")
    else:
        why = ("its lower bound does not clear zero once overlapping trades "
               "are allowed for" if not pass1 else
               "it is not ahead on the clean 2022-2026 slice")
        v = (f"NOT CONFIRMED - {why}. Keep V81 as the served model and "
             f"report the combo as suggestive.")
    lines.append(f"\n  VERDICT: {v}")
    if not prereg_ok:
        lines.append("  (decision settings changed after an earlier result - "
                     "exploratory)")
    if not ok:
        lines.append("  (integrity check failed - these are not V86's models "
                     "or data)")
    v1 = {L: get("V81", H, L, "90% lo") for L in DECIDING_BLOCKS}
    lines.append(f"\n  for completeness, V81 (verdict set in V86): "
                 f"{get('V81', H, d0, 'Diff'):+.2f} pp, lower 90% bound "
                 + ", ".join(f"{v:+.2f} ({L}-month)" for L, v in v1.items()))
    lines.append(f"\n  pre-registered {prereg_when}; tables in {out_dir}/")

    print("\n" + "=" * W_)
    print("  [6] THE PRE-REGISTERED VERDICT")
    print("=" * W_)
    print("\n".join(lines))
    with open(os.path.join(out_dir, "verdict.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  total {(time.time() - t0) / 60:.1f} min")
    return T


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
    RUN_BIG_CACHE   = "price_cache_v68" # same as V86
    RUN_N_SEEDS     = 5                 # same as V86
    RUN_CHUNK       = 250               # MUST equal V86's RUN_CHUNK (cache key)
    RUN_N_BOOT      = 10000             # more draws: the 2.5th percentile decides
    RUN_SHOW_OTHERS = True              # also fresh-all and core, information only
    RUN_OUT_DIR     = "thesis_tables_v87"
    RUN_V86_DIR     = "thesis_tables_v86"
    # -------------------------------------------------------------------------

    run_v87(model_in=RUN_MODEL_IN, core_cache=RUN_CORE_CACHE,
            big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS, chunk=RUN_CHUNK,
            n_boot=RUN_N_BOOT, show_other_universes=RUN_SHOW_OTHERS,
            out_dir=RUN_OUT_DIR, v86_dir=RUN_V86_DIR)
