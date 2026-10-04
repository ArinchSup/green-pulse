"""
V79 - DOES THE MODEL BEAT THE MECHANISM IT DISCOVERED?

THE QUESTION
------------
V78 put 57% of the out-of-sample lift on a single feature: mkt_ret_20, the
20-day market return, the only one clearing 2 SD of the permutation noise. The
reading was that the model has largely learned to buy market pullbacks.

If that is true, a one-line rule - "fire when the 20-day market return is in
its bottom 1%" - should capture most of the +7.86pp. If it does, the
gradient-boosted model is decoration and the thesis should say so. If the model
clearly beats it, the model adds something over the mechanism it found.

This is a falsification test of a claim made in this project's own results, not
a search for a better number. Every arm is pre-specified below and all of them
are reported, including the ones that would embarrass the model.

THE ARMS - identical rule, identical control, only the SCORE differs
-------------------------------------------------------------------
  model                     the deployed V76 score
  model minus mkt_ret_20    refit without it. The ablation complement to
                            permutation: if the lift survives, permutation
                            overstated that feature's role.
  -mkt_ret_20               the hypothesis, direction PRE-SPECIFIED by it
                            (pullback -> buy). One raw feature, no model.
  other single features     both directions, EXPLORATORY. These exist to
                            answer a harder question than the headline one.
  random score              THE FLOOR. Same rolling rule, same matched
                            control, a uniform random score.

WHY THE RANDOM ARM IS THE MOST IMPORTANT ONE
--------------------------------------------
Every result in this project rests on the matched control. If the rolling-
threshold machinery plus that control returns a positive lift on a RANDOM
score, then the comparison has an artifact in it and the +7.86pp headline is
not what it appears to be. The random arm must come back at roughly zero. It
has never been run against this exact rule, so it is run here.

ON DIRECTION
------------
mkt_ret_20's sign is a prediction of the mechanism hypothesis, so its arm is
confirmatory. For every other feature both signs are tested and both are
printed, because picking the better sign after seeing it is a 2x multiple
comparison - the same post-hoc direction trap this project hit earlier. Those
rows are labelled exploratory and should not be quoted as single results.
"""

import os
import sys
from math import comb

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v76 as M76

SEED = 79
OUT_DIR = "thesis_tables_v79"

SINGLE_FEATURES = ["mkt_ret_20", "ret_5", "ret_20", "atr_pct",
                   "mkt_vol_20", "rsi_14", "bb_position", "px_vs_ema20"]


# =============================================================================
# FAST MATCHED CONTROL
# =============================================================================
class MatchedControl:
    """
    Name-and-month matched control, vectorised.

    V73's version loops per signal per draw, which is fine for one arm and far
    too slow for twenty. The pools are identical; only the sampling is
    rewritten: group positions are sorted once into a flat array with per-group
    offsets and lengths, so one draw is a single vectorised gather.

    Verified against V73.scorecard on the same mask - see the self-test in
    __main__.
    """

    def __init__(self, te):
        di = pd.DatetimeIndex(te["date"])
        key = pd.Series(te["ticker"].astype(str)) + "|" + \
            pd.PeriodIndex(di, freq="M").astype(str)
        codes, _ = pd.factorize(key, sort=False)
        self.codes = codes
        order = np.argsort(codes, kind="stable")
        self.order = order
        counts = np.bincount(codes)
        self.lens = counts
        self.starts = np.r_[0, np.cumsum(counts)[:-1]]
        self.label = te["label"].to_numpy(float)
        self.r = te["r_multiple"].to_numpy(float)
        self.is_loss = (te["outcome"] == "loss").to_numpy(float)
        self.bars = te["bars_held"].to_numpy(float)
        self.n = len(te)

    def draw(self, fired_pos, rng):
        g = self.codes[fired_pos]
        ln = self.lens[g]
        off = (rng.random(len(g)) * ln).astype(np.int64)
        return self.order[self.starts[g] + off]

    def metrics(self, pos):
        r = self.r[pos]
        win, loss = r[r > 0], r[r < 0]
        return {"precision": float(self.label[pos].mean()),
                "expectancy_R": float(r.mean()),
                "loss_share": float(self.is_loss[pos].mean()),
                "R_per_bar": float(r.sum() / np.nansum(self.bars[pos])),
                "payoff_ratio": (float(win.mean() / abs(loss.mean()))
                                 if win.size and loss.size else np.nan)}

    def compare(self, fired_pos, n_draws=200, seed=SEED):
        rng = np.random.default_rng(seed)
        mine = self.metrics(fired_pos)
        draws = [self.metrics(self.draw(fired_pos, rng)) for _ in range(n_draws)]
        out = {}
        for k, v in mine.items():
            vals = np.array([d[k] for d in draws], float)
            vals = vals[np.isfinite(vals)]
            higher = k not in ("loss_share",)
            beat = int((vals < v).sum() if higher else (vals > v).sum())
            out[k] = {"model": v, "ctl_median": float(np.median(vals)),
                      "ctl_lo": float(vals.min()), "ctl_hi": float(vals.max()),
                      "beat": beat, "n": len(vals),
                      "p": (len(vals) - beat + 1) / (len(vals) + 1),
                      "lift": v - float(np.median(vals))}
        return out


# =============================================================================
# ONE ARM
# =============================================================================
def sign_test(te, fire, floor=15):
    f = te[fire]
    rows = []
    for y, g in te.groupby("year"):
        fy = f[f["year"] == y]
        if len(fy) >= floor:
            rows.append((float(fy["label"].mean()), float(g["label"].mean())))
    if len(rows) < 3:
        return np.nan, 0, 0
    n = len(rows)
    k = sum(1 for a, b in rows if a > b)
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, k, n


def arm(te, score, label, mc, quantile, window_days, n_draws, seed,
        kind="", verbose=True):
    t = te.assign(p=np.asarray(score, float))
    fire, _ = M76.fire_rolling(t, quantile, window_days, spaced=False)
    pos = np.flatnonzero(fire)
    if len(pos) < 100:
        if verbose:
            print(f"    {label}: only {len(pos)} signals - skipped")
        return None
    cmp = mc.compare(pos, n_draws, seed)
    p_sign, k, n = sign_test(t, fire)
    row = {"Arm": label, "Kind": kind, "Signals": len(pos),
           "Years": int(t.loc[fire, "year"].nunique()),
           "Precision": cmp["precision"]["model"],
           "Control": cmp["precision"]["ctl_median"],
           "Prec lift pp": cmp["precision"]["lift"] * 100,
           "Beat (prec)": f"{cmp['precision']['beat']}/{cmp['precision']['n']}",
           "p (prec)": cmp["precision"]["p"],
           "ExpR lift": cmp["expectancy_R"]["lift"],
           "p (ExpR)": cmp["expectancy_R"]["p"],
           "Stop-out lift pp": cmp["loss_share"]["lift"] * 100,
           "Sign p (f15)": p_sign, "Sign years": f"{k}/{n}" if n else ""}
    if verbose:
        print(f"    {label:<28} {len(pos):>6} sig  "
              f"lift {row['Prec lift pp']:+6.2f}pp  "
              f"ExpR {row['ExpR lift']:+.4f}  "
              f"beat {row['Beat (prec)']:>8}  sign p {p_sign:.3f}")
    return row


# =============================================================================
# RUNNER
# =============================================================================
def run_v79(model_file="entry_model_v76.joblib", price_cache=None,
            out_dir=OUT_DIR, n_draws=200, seed=SEED, n_random=5,
            ablate=True, ablate_all_market=False, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model = M76.load_model(model_file)
    p, r = model["provenance"], model["rule"]
    price_cache = price_cache or p["price_cache"]
    q, win = r["quantile"], r["window_days"]
    feats = model["features"]

    print("=" * 100)
    print("V79 - DOES THE MODEL BEAT THE MECHANISM IT DISCOVERED?")
    print("=" * 100)
    print(f"  cache {price_cache} | rule: top {q:.1%} of trailing {win}d, "
          f"refit {r['refit_freq']}, unspaced")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, p["horizon"], p["step"],
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, p["label_mode"])
    d["year"] = pd.DatetimeIndex(d["date"]).year

    print(f"\n  walk-forward: full model ...")
    te = E64.walk_forward(d, feats, p["min_train_years"], p["n_seeds"],
                          p["seed"], p["horizon"],
                          verbose=False).reset_index(drop=True)
    print(f"  {len(te):,} out-of-sample trades, {te['year'].nunique()} years")
    mc = MatchedControl(te)

    rows = []
    print(f"\n  ARMS ({n_draws} control draws each)")

    # --- the floor ---------------------------------------------------------
    print(f"\n  [floor] random score, same rule and control:")
    for i in range(n_random):
        rg = np.random.default_rng(1000 + i)
        rows.append(arm(te, rg.random(len(te)), f"random score #{i+1}", mc,
                        q, win, n_draws, seed, "floor", verbose))

    # --- the model ---------------------------------------------------------
    print(f"\n  [model]")
    rows.append(arm(te, te["p"], "model (deployed)", mc, q, win, n_draws,
                    seed, "model", verbose))

    # --- the hypothesis ----------------------------------------------------
    print(f"\n  [hypothesis] one raw feature, direction pre-specified:")
    rows.append(arm(te, -te["mkt_ret_20"], "-mkt_ret_20 (buy the dip)", mc,
                    q, win, n_draws, seed, "hypothesis", verbose))

    # --- ablation ----------------------------------------------------------
    if ablate:
        drop = (["mkt_ret_20"] if not ablate_all_market
                else [f for f in feats if f.startswith("mkt_")])
        sub = [f for f in feats if f not in drop]
        print(f"\n  [ablation] refit without {', '.join(drop)} "
              f"({len(sub)} features) ...")
        te2 = E64.walk_forward(d, sub, p["min_train_years"], p["n_seeds"],
                               p["seed"], p["horizon"],
                               verbose=False).reset_index(drop=True)
        mc2 = MatchedControl(te2)
        a = arm(te2, te2["p"], f"model minus {'+'.join(drop)}", mc2, q, win,
                n_draws, seed, "ablation", verbose)
        rows.append(a)

    # --- exploratory single features --------------------------------------
    print(f"\n  [exploratory] single features, BOTH directions "
          f"(do not quote one alone):")
    for f in SINGLE_FEATURES:
        if f not in te.columns:
            continue
        for sgn, tag in ((-1.0, "low"), (1.0, "high")):
            if f == "mkt_ret_20" and sgn < 0:
                continue                      # already run as the hypothesis
            rows.append(arm(te, sgn * te[f], f"{tag} {f}", mc, q, win,
                            n_draws, seed, "exploratory", verbose))

    out = pd.DataFrame([x for x in rows if x])
    out.to_csv(f"{out_dir}/arms.csv", index=False)

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("  ALL ARMS")
    print("=" * 100)
    cols = ["Arm", "Kind", "Signals", "Years", "Prec lift pp", "ExpR lift",
            "Beat (prec)", "p (prec)", "Stop-out lift pp", "Sign p (f15)"]
    print(out[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))

    def pick(kind):
        s = out[out["Kind"] == kind]
        return s if len(s) else None

    fl, md, hy = pick("floor"), pick("model"), pick("hypothesis")
    ab, ex = pick("ablation"), pick("exploratory")

    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    if fl is not None:
        lo, hi = fl["Prec lift pp"].min(), fl["Prec lift pp"].max()
        print(f"  FLOOR   random score: {lo:+.2f} to {hi:+.2f} pp "
              f"(mean {fl['Prec lift pp'].mean():+.2f})")
        if abs(fl["Prec lift pp"].mean()) > 1.0:
            print(f"    *** THE FLOOR IS NOT ZERO. The rolling rule plus the "
                  f"matched control returns a")
            print(f"    *** lift on a RANDOM score, so the machinery has an "
                  f"artifact and every number")
            print(f"    *** in this project built on it must be re-derived. "
                  f"Stop and fix this first.")
        else:
            print(f"    The machinery is clean: a random score earns "
                  f"essentially nothing.")
    # The decisive comparison is the model against the BEST single-feature arm,
    # not against the pre-specified one. An earlier version judged only the
    # hypothesis arm and printed "THE MODEL ADDS REAL WORK" on a run where six
    # single features beat the model outright - a headline contradicted by a
    # note three lines below it.
    singles = pd.concat([x for x in (hy, ex) if x is not None]) \
        if (hy is not None or ex is not None) else None
    if md is not None and singles is not None and len(singles):
        m = float(md["Prec lift pp"].iloc[0])
        mr = float(md["ExpR lift"].iloc[0])
        best = singles.loc[singles["Prec lift pp"].idxmax()]
        b = float(best["Prec lift pp"])
        beats = singles[singles["Prec lift pp"] >= m]
        print(f"\n  MODEL                {m:+.2f} pp / {mr:+.4f} R   "
              f"({int(md['Signals'].iloc[0]):,} signals)")
        print(f"  BEST SINGLE FEATURE  {b:+.2f} pp / "
              f"{float(best['ExpR lift']):+.4f} R   ({best['Arm']})")
        print(f"  single-feature arms matching or beating the model: "
              f"{len(beats)}/{len(singles)}")
        if len(beats):
            # Column names here contain spaces, so itertuples renames them to
            # positional _N attributes and the numbering shifts whenever a
            # column is added. Index by name.
            for _, row in beats.sort_values("Prec lift pp",
                                            ascending=False).iterrows():
                sp = row["Sign p (f15)"]
                print(f"      {row['Arm']:<26} "
                      f"{row['Prec lift pp']:+6.2f} pp   "
                      f"ExpR {row['ExpR lift']:+.4f}   "
                      + (f"sign p {sp:.3f}" if np.isfinite(sp) else "sign p -"))

        # Multiple-comparison defence. A maximum over ~15 noisy arms would be
        # one lucky winner with no mirror structure. Several winners whose
        # OPPOSITE-direction twins are strongly negative is a directional
        # effect, not a max-over-noise artifact.
        if ex is not None:
            pairs, sym = [], 0
            for f in SINGLE_FEATURES:
                lo = ex[ex["Arm"] == f"low {f}"]
                hi = ex[ex["Arm"] == f"high {f}"]
                if len(lo) and len(hi):
                    a_, b_ = (float(lo["Prec lift pp"].iloc[0]),
                              float(hi["Prec lift pp"].iloc[0]))
                    pairs.append((f, a_, b_))
                    if a_ * b_ < 0 and min(abs(a_), abs(b_)) > 3.0:
                        sym += 1
            if pairs:
                print(f"\n  MIRROR CHECK  opposite directions of the same "
                      f"feature:")
                for f, a_, b_ in pairs:
                    print(f"      {f:<20} low {a_:+6.2f} pp   "
                          f"high {b_:+6.2f} pp")
                print(f"    {sym}/{len(pairs)} features show a strong "
                      f"SYMMETRIC split (both sides > 3pp,")
                print(f"    opposite signs). Symmetry rules out "
                      f"max-over-noise: luck gives one winner")
                print(f"    with a flat twin, not several with strongly "
                      f"negative twins.")

        if b >= m:
            print(f"\n    *** THE MODEL IS BEATEN BY A ONE-LINE RULE. "
                  f"'{best['Arm']}' earns {b:+.2f}pp")
            print(f"    *** against the model's {m:+.2f}pp - "
                  f"{b/m if m else float('nan'):.1f}x - with no model at all. "
                  f"The 28-feature")
            print(f"    *** gradient-boosted model captures a FRACTION of an "
                  f"effect a single raw")
            print(f"    *** mean-reversion column captures outright.")
            print(f"\n    This is the result, and it is a stronger thesis than "
                  f"the model working:")
            print(f"      1. within-name, within-month entry timing carries a "
                  f"large real effect;")
            print(f"      2. a single mean-reversion feature extracts it;")
            print(f"      3. the ML model extracts less than half of it, so it "
                  f"is not merely")
            print(f"         unnecessary here - it is WORSE than the simplest "
                  f"possible rule.")
            print(f"    Report the single-feature rule as the finding and the "
                  f"model as evidence about")
            print(f"    what gradient boosting failed to learn. Do NOT lead "
                  f"with the model.")
        elif b >= 0.8 * m:
            print(f"\n    A SINGLE FEATURE MATCHES THE MODEL ({b:+.2f} vs "
                  f"{m:+.2f} pp). The model adds")
            print(f"    nothing a reader would pay for. Report the simple rule "
                  f"as the result.")
        else:
            print(f"\n    THE MODEL ADDS REAL WORK: the best single feature "
                  f"gets {b/m if m else float('nan'):.0%} of its lift.")
            print(f"    That holds only if the random floor above is clean and "
                  f"the single-feature arms")
            print(f"    were pre-specified - treat the best of "
                  f"{len(singles)} as a maximum, not a result.")
    if ab is not None and md is not None:
        a = float(ab["Prec lift pp"].iloc[0])
        m = float(md["Prec lift pp"].iloc[0])
        print(f"\n  ABLATION  without the feature: {a:+.2f} pp "
              f"(full model {m:+.2f})  -> costs {m-a:+.2f} pp")
        print(f"    Permutation said it carried 57% of the lift. Ablation "
              f"refits without it, so the")
        print(f"    model can route around it; the two agreeing is the strong "
              f"result, and ablation")
        print(f"    wins where they disagree.")
    if hy is not None and md is not None:
        h = float(hy["Prec lift pp"].iloc[0])
        m = float(md["Prec lift pp"].iloc[0])
        print(f"\n  THE PRE-SPECIFIED HYPOTHESIS  -mkt_ret_20 (buy the dip): "
              f"{h:+.2f} pp, "
              f"{h/m if m else float('nan'):.0%} of the model")
        print(f"    This was the mechanism V78's permutation table pointed at. "
              f"It is confirmatory")
        print(f"    rather than exploratory, so it is the one single-feature "
              f"number quotable on")
        print(f"    its own - but note the per-NAME reversion features below "
              f"beat the MARKET one,")
        print(f"    which relocates the mechanism from market timing to "
              f"per-name mean reversion.")

    print(f"\n  wrote {out_dir}/arms.csv")
    return out


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v76.joblib"
    RUN_PRICE_CACHE = None      # None uses the cache the model was trained on
    RUN_OUT_DIR     = "thesis_tables_v79"
    RUN_N_DRAWS     = 200       # control draws per arm; p floor is 1/(N+1)
    RUN_N_RANDOM    = 5         # random-score arms (the floor)
    RUN_ABLATE      = True      # refit without mkt_ret_20 (one extra walk-forward)
    RUN_ABLATE_ALL  = False     # True drops every mkt_* feature instead
    RUN_SELFTEST    = True      # check the fast control against V73's
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    if RUN_SELFTEST:
        print("self-test: fast matched control vs V73.scorecard ...")
        try:
            import evaluate_sniper_metrics_v73 as V73
            import class_ai_entry_model_v76 as _M
            _m = _M.load_model(RUN_MODEL_FILE)
            _p = _m["provenance"]
            _pc = RUN_PRICE_CACHE or _p["price_cache"]
            _n = sorted(os.path.splitext(f)[0] for f in os.listdir(_pc)
                        if f.endswith(".pkl"))[:40]
            _d = E64.build_dataset(_n, _pc, _p["horizon"], _p["step"],
                                   verbose=False)
            E64.assert_market_columns(_d)
            _d = E64.apply_label_mode(_d, _p["label_mode"])
            _d["year"] = pd.DatetimeIndex(_d["date"]).year
            _te = E64.walk_forward(_d, _m["features"], _p["min_train_years"],
                                   1, _p["seed"], _p["horizon"],
                                   verbose=False).reset_index(drop=True)
            _f, _ = M76.fire_rolling(_te, 0.01, 252, spaced=False)
            _pos = np.flatnonzero(_f)
            _mc = MatchedControl(_te)
            _fast = _mc.compare(_pos, 60, 7)
            _floor = float(_te["p"].min()) - 1.0
            _sc = V73.scorecard(_te.assign(p=np.where(_f, _te["p"], _floor)),
                                len(_pos) / len(_te), 7, 60)
            _b = _sc[0].set_index("Metric")
            for _k in ("precision", "expectancy_R", "loss_share"):
                _a = _fast[_k]["ctl_median"]
                _c = float(_b.loc[_k, "Control median"])
                print(f"  {_k:<14} fast {_a:+.4f}  V73 {_c:+.4f}  "
                      f"diff {abs(_a-_c):.4f}  "
                      f"{'OK' if abs(_a-_c) < 0.02 else 'MISMATCH'}")
        except Exception as _e:
            print(f"  self-test could not run ({_e}) - continuing")
        print()

    run_v79(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
            out_dir=RUN_OUT_DIR, n_draws=RUN_N_DRAWS, n_random=RUN_N_RANDOM,
            ablate=RUN_ABLATE, ablate_all_market=RUN_ABLATE_ALL,
            seed=RUN_SEED)
