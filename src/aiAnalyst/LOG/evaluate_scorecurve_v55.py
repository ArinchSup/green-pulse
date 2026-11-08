#!/usr/bin/env python3
"""
evaluate_scorecurve_v55.py - keep the curve, improve the score.

WHAT V54 ACTUALLY FOUND

Handing market-state features to a free-form model produced this:

    tracking Spearman   +0.638 -> +0.721      best in the project
    spread ratio          0.66 -> 0.76        the flatness V52 flagged, narrowing
    years within 5pp     13/16 -> 14/16       year-level calibration improved
    AUC             0.6907 -> 0.6713          CI -0.047 to +0.003, NOT established
    pooled MAE        2.15pp -> 4.82pp        much worse
    Brier skill      +0.0514 -> -0.0371       negative

Years-within-5pp bins by YEAR; pooled MAE bins by PREDICTED PROBABILITY. So the
model got the level right for each year - exactly what V52 could not do - while
getting the magnitudes wrong within a year. And the AUC interval spanning zero
says the RANKING survived; only the calibration broke.

Market state has real temporal signal and about fifteen to twenty independent
market episodes behind it. A free-form model memorises the regimes it trained on
and extrapolates hard. Right ordering, wrong numbers.

THE HYPOTHESIS

two_stage works BECAUSE of the twelve-bin empirical curve. That curve is what
turns a ranking into calibrated probabilities, and nothing has ever been fed
through it except a volatility forecast. So:

    two_stage        OLS(3 vol features)   ->  12-bin curve
    score_curve_vol  gbm(3 vol features)   ->  12-bin curve
    score_curve_all  gbm(9 features)       ->  12-bin curve
    two_stage_x_mkt  two_stage probability x a shrunk date-level multiplier

score_curve_vol isolates the first stage: same features, better learner. If it
ties two_stage, then any gain in score_curve_all belongs to the FEATURES rather
than to gradient boosting, which is the cleaner claim.

two_stage_x_mkt is the conservative alternative: leave the deployed model alone
and spend the market signal on five shrunk numbers instead of a tree ensemble.
Fifteen episodes of information can pay for five parameters; it cannot pay for
three hundred trees.

THE CURVE MUST BE FITTED ON HELD-OUT SCORES

A curve built on a model's own in-sample fitted values inherits that model's
overfitting. It barely matters for OLS; for gradient boosting it is fatal - the
in-sample scores separate the classes far better than out-of-sample scores ever
will, so the curve maps them to event rates that never materialise.

So inside each training period: fit on the earlier part, score a held-out tail,
and build the curve from those genuinely out-of-sample scores. Then refit on the
whole training period for the test prediction. The curve is indexed by score
PERCENTILE rather than by raw score, so a shift in the score distribution between
the inner model and the final one cannot move the mapping.

USAGE
  python evaluate_scorecurve_v55.py
  python evaluate_scorecurve_v55.py --model logit --shrink 0.5
  python evaluate_scorecurve_v55.py --quick
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk as P2
import evaluate_calibration_v52 as V52
from evaluate_direct_v54 import (MARKET, ONE_SIDED, POSITION, VOL, build_panel,
                                paired_stat, summarise, two_stage)

ALL_FEATURES = VOL + ONE_SIDED + POSITION + MARKET
N_CURVE_BINS = 12
INNER_TAIL = 0.25           # share of the training period held out for the curve

# Shrink is swept rather than chosen. It is a prior on how much fifteen market
# episodes can tell you, not a fitted parameter, so the honest thing is to show
# the whole trade-off and say which point was picked. shrink=0 must reproduce
# two_stage exactly, which makes it a free correctness check on the arm.
SHRINKS = [0.0, 0.25, 0.5, 1.0]

ARMS = {"two_stage": None, "score_curve_vol": VOL, "score_curve_all": ALL_FEATURES}
ARMS.update({f"mkt_x{sh:g}": ("multiplier", sh) for sh in SHRINKS})
BASELINE = "two_stage"


# =============================================================================
# SCORE -> CURVE
# =============================================================================
def fit_score(Xtr, ytr, Xte, kind):
    """Raw ranking score. Only its ORDER matters - the curve supplies the
    calibration, so no probability interpretation is needed here."""
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    if ok.sum() < 500 or len(np.unique(ytr[ok])) < 2:
        return None, None
    Xtr, ytr = Xtr[ok], ytr[ok]
    mu = Xtr.mean(axis=0)
    sd = np.where(Xtr.std(axis=0) > 1e-9, Xtr.std(axis=0), 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    finite = np.isfinite(Zte).all(axis=1)
    Zte = np.nan_to_num(Zte, nan=0.0, posinf=0.0, neginf=0.0)
    if kind == "gbm":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=2.0, n_jobs=4, verbosity=0,
                              eval_metric="logloss")
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(Ztr, ytr)
    s_tr = m.predict_proba(Ztr)[:, 1]
    s_te = np.where(finite, m.predict_proba(Zte)[:, 1], np.nan)
    return s_te, (s_tr, ytr)


def pct_curve(scores, events, n_bins=N_CURVE_BINS):
    """
    Map score PERCENTILE to event rate, ISOTONICALLY.

    The monotonicity is not cosmetic. AUC is invariant under any monotone
    transformation, so a calibration curve that preserves the ranking cannot
    change AUC at all. A first version of this binned the empirical event rate
    and interpolated without forcing the result increasing; noisy bins inverted,
    the map stopped being monotone, and AUC fell by 0.076 on features whose raw
    ranking was fine. Isotonic regression fixes it properly - it pools adjacent
    violators, which both enforces monotonicity and denoises the small bins.

    Percentiles rather than raw scores, because the fold models and the final
    model are fitted on different amounts of data and their score scales differ;
    their orderings are comparable, their levels are not.
    """
    d = pd.DataFrame({"s": scores, "e": np.asarray(events, float)}).dropna()
    if len(d) < 500:
        return None
    from sklearn.isotonic import IsotonicRegression
    q = d["s"].rank(pct=True).to_numpy()
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True,
                            y_min=0.0, y_max=1.0)
    ir.fit(q, d["e"].to_numpy())
    xs = np.linspace(0.0, 1.0, max(n_bins * 8, 64))
    return {"x": xs, "y": np.maximum.accumulate(ir.predict(xs))}


def apply_curve(test_scores, train_scores, cv):
    """Test score -> its percentile in the TRAINING score distribution -> the
    curve. Percentiles taken against training scores only; using the test set's
    own distribution would leak its composition into every prediction."""
    ref = np.sort(np.asarray(train_scores, float))
    q = np.searchsorted(ref, np.asarray(test_scores, float), side="left") / max(
        len(ref), 1)
    p = np.interp(q, cv["x"], cv["y"])
    return np.where(np.isfinite(test_scores), p, np.nan)


def oof_scores(tr, feats, kind, n_folds, embargo):
    """
    Out-of-fold scores covering the WHOLE training period.

    A single held-out tail gives a curve built on one contiguous era from a
    quarter of the data. Sequential purged folds cover every era instead, so the
    curve is not specific to whichever regime happened to sit at the end of the
    training window. Each fold trains on data outside itself, embargoed on both
    sides so a training label cannot resolve inside the fold it is scoring.
    """
    dates = tr["date"]
    edges = dates.quantile(np.linspace(0, 1, n_folds + 1)).to_numpy()
    emb = pd.Timedelta(days=embargo)
    out_s, out_y = [], []
    for k in range(n_folds):
        a, b = pd.Timestamp(edges[k]), pd.Timestamp(edges[k + 1])
        hold = (dates >= a) & (dates < b) if k < n_folds - 1 else (dates >= a)
        keep = (dates < a - emb) | (dates > b + emb)
        if keep.sum() < 3000 or hold.sum() < 300:
            continue
        sc, _ = fit_score(tr.loc[keep, feats].to_numpy(float),
                          tr.loc[keep, "event"].to_numpy(float),
                          tr.loc[hold, feats].to_numpy(float), kind)
        if sc is None:
            continue
        out_s.append(sc)
        out_y.append(tr.loc[hold, "event"].to_numpy(float))
    if not out_s:
        return None, None
    return np.concatenate(out_s), np.concatenate(out_y)


def score_curve(tr, te, feats, kind, n_folds=3, embargo=180, audit=None):
    """Out-of-fold scores build the curve; a final fit on all training data
    supplies the test scores."""
    s_oof, y_oof = oof_scores(tr, feats, kind, n_folds, embargo)
    if s_oof is None:
        return None
    cv = pct_curve(s_oof, y_oof)
    if cv is None:
        return None
    s_te, trained = fit_score(tr[feats].to_numpy(float),
                              tr["event"].to_numpy(float),
                              te[feats].to_numpy(float), kind)
    if s_te is None:
        return None
    p = apply_curve(s_te, trained[0], cv)
    if audit is not None:
        # A monotone map cannot change AUC. Keeping both is the check that would
        # have caught the non-monotone curve on the first run instead of the
        # third.
        audit.append((s_te, p, te["event"].to_numpy(float)))
    return p


# =============================================================================
# DATE-LEVEL MULTIPLIER
# =============================================================================
def two_stage_x_market(tr, te, shrink, n_buckets=5):
    """
    The deployed model, times a date-level multiplier read off market state.

    Per training bucket of market volatility: the ratio of realised event rate to
    mean predicted probability. Above 1.0 means the deployed model under-predicts
    in that regime. Shrunk toward 1.0, because five bucket ratios estimated from
    roughly fifteen independent market episodes are noisy, and an unshrunk ratio
    is how the V54 market rung produced a negative Brier skill.
    """
    p_tr = two_stage(tr, tr)
    p_te = two_stage(tr, te)
    if p_tr is None or p_te is None:
        return None
    mv_tr = tr["mkt_vol"].to_numpy(float)
    mv_te = te["mkt_vol"].to_numpy(float)
    ok = np.isfinite(mv_tr) & np.isfinite(p_tr)
    if ok.sum() < 5000:
        return None
    edges = np.quantile(mv_tr[ok], np.linspace(0, 1, n_buckets + 1)[1:-1])
    b_tr = np.digitize(mv_tr, edges)
    b_te = np.digitize(mv_te, edges)
    y = tr["event"].to_numpy(float)
    mult = np.ones(n_buckets + 1)
    for b in range(n_buckets):
        m = ok & (b_tr == b)
        if m.sum() < 500 or p_tr[m].mean() <= 0:
            continue
        raw = y[m].mean() / p_tr[m].mean()
        mult[b] = 1.0 + shrink * (raw - 1.0)
    out = p_te * mult[np.clip(b_te, 0, n_buckets)]
    return np.where(np.isfinite(mv_te), np.clip(out, 1e-6, 1 - 1e-6), p_te)


# =============================================================================
# WALK FORWARD
# =============================================================================
def walk(panel, name, feats, cfg):
    """One record per test year, pooled."""
    yrs = sorted(panel["date"].dt.year.unique())
    out = []
    for yr in [y for y in yrs if y >= yrs[0] + int(cfg.min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = panel[panel["date"] <= opens - pd.Timedelta(days=cfg.embargo_days)]
        te = panel[(panel["date"] >= opens) &
                   (panel["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        if name == BASELINE:
            p = two_stage(tr, te)
        elif isinstance(feats, tuple) and feats[0] == "multiplier":
            p = two_stage_x_market(tr, te, feats[1])
        else:
            p = score_curve(tr, te, feats, cfg.model, cfg.folds,
                            cfg.embargo_days, cfg.audit.setdefault(name, []))
        if p is None:
            continue
        out.append(pd.DataFrame({"rid": te.index.to_numpy(),
                                 "date": te["date"].to_numpy(), "p": p,
                                 "y": te["event"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else None


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_scorecurve_v55() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v54.pkl",
                    help="shares V54's panel - the features are identical")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=6)
    ap.add_argument("--embargo-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--model", default="gbm", choices=["gbm", "logit"])
    ap.add_argument("--folds", type=int, default=3,
                    help="purged folds inside each training period for the curve")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="scorecurve_eval_v55.json")
    return ap


def run_scorecurve_v55(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, embargo_days=None, model=None, folds=None, n_boot=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_scorecurve_v55()
        run_scorecurve_v55(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "panel_cache": panel_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "embargo_days": embargo_days, "model": model, "folds": folds, "n_boot": n_boot, "max_tickers": max_tickers, "seed": seed, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step, cfg.n_boot = 60, 20, 200
    V52.N_BOOT = cfg.n_boot
    cfg.audit = {}

    print("=" * 96)
    print("V55 - KEEP THE CURVE, IMPROVE THE SCORE")
    print("=" * 96)

    if os.path.exists(cfg.panel_cache) and not cfg.rebuild:
        panel = pd.read_pickle(cfg.panel_cache)
        print(f"  loaded {len(panel):,} rows from {cfg.panel_cache}")
    else:
        tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                         for p in glob.glob(os.path.join(cfg.cache, "*.pkl")))
        if cfg.max_tickers:
            tickers = tickers[:cfg.max_tickers]
        if not tickers:
            sys.exit(f"No .pkl files in {cfg.cache}.")
        print(f"  building the panel from {len(tickers)} tickers")
        panel = build_panel(tickers, cfg.cache, cfg.step)
        panel.to_pickle(cfg.panel_cache)
    if panel.empty:
        sys.exit("Empty panel.")

    missing = [f for f in ALL_FEATURES if f not in panel.columns]
    if missing:
        sys.exit(f"Panel is missing {missing} - rebuild it with V54.")
    mkn = panel.attrs.get("mkt_median_n")
    print(f"  {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"  base rate {panel['event'].mean():.1%} | market cross-section "
          f"median {mkn if mkn is not None else 'not recorded'} tickers/date")
    if mkn is None:
        print("  NOTE the cached panel does not record its cross-section size.")
        print("       If it predates V54's fix the market features are noise -")
        print("       rebuild with V54 --rebuild to be sure.")

    print(f"\n  first stage {cfg.model} | isotonic curve from {cfg.folds} "
          f"purged out-of-fold splits of each training period")
    print(f"  date multiplier shrink swept over {SHRINKS}")

    print("\n" + "=" * 96)
    print("WALK-FORWARD")
    print("=" * 96)
    runs = {}
    for name, feats in ARMS.items():
        rec = walk(panel, name, feats, cfg)
        if rec is None:
            print(f"  {name}: no usable windows")
            continue
        runs[name] = rec
        tag = ("curve" if feats is None else "mult" if feats == "multiplier"
               else f"{len(feats)}f")
        print(f"  {name:<18}{len(rec):>8,} predictions   {tag}")
    if BASELINE not in runs:
        sys.exit("Baseline did not run.")

    # Every arm must be scored on the SAME rows. An arm whose inner split failed
    # in the early years would otherwise be graded on a different, later set of
    # years, and a tracking or calibration difference between two arms would be
    # partly a difference of sample. Intersect first, report what it cost.
    common = set(runs[BASELINE]["rid"])
    for rec in runs.values():
        common &= set(rec["rid"])
    dropped = {k: len(v) - len(common) for k, v in runs.items()}
    if any(d > 0 for d in dropped.values()):
        print(f"\n  aligning arms on the {len(common):,} rows every arm scored "
              f"(an arm short of this skipped years):")
        for k, d in dropped.items():
            if d > 0:
                print(f"    {k}: {d:,} rows set aside")
    runs = {k: v[v["rid"].isin(common)].reset_index(drop=True)
            for k, v in runs.items()}
    yrs = sorted(pd.to_datetime(runs[BASELINE]["date"]).dt.year.unique())
    print(f"  all arms graded on {len(common):,} rows, "
          f"{yrs[0]}-{yrs[-1]} ({len(yrs)} years)")

    base = runs[BASELINE]
    ref = float(np.nanmean(base["y"].to_numpy()))
    stats = {k: summarise(v, ref) for k, v in runs.items()}

    # Self-checks. Both are theorems, not judgements: a monotone calibration map
    # cannot change AUC, and a zero-shrink multiplier is the identity.
    print("\n  SELF-CHECKS")
    for name, rec in cfg.audit.items():
        if not rec:
            continue
        # Monotonicity is a property of ONE fit. Each test year has its own
        # curve and its own training score distribution, so pooling years and
        # sorting by raw score mixes incomparable scales and invents inversions
        # that are not there. Check inside each fit, then total.
        bad = 0
        for sc_i, pm_i, _ in rec:
            o = np.argsort(sc_i, kind="stable")
            bad += int((np.diff(pm_i[o]) < -1e-12).sum())
        sc = np.concatenate([r[0] for r in rec])
        pm = np.concatenate([r[1] for r in rec])
        yy = np.concatenate([r[2] for r in rec])
        a_raw, a_map = V52.auc(sc, yy), V52.auc(pm, yy)
        lvl = len(np.unique(np.round(pm, 10)))
        verdict = "monotone" if bad == 0 else f"NOT MONOTONE ({bad:,} inversions)"
        print(f"    {name}: {verdict} within every fit | {lvl:,} distinct levels")
        print(f"      pooled AUC: raw score {a_raw:.4f}, calibrated "
              f"{a_map:.4f} ({a_map - a_raw:+.4f})")
    if cfg.audit:
        print("      Within a fit the map cannot reorder anything, so any AUC it")
        print("      loses there is ties from isotonic's flat regions - the curve")
        print("      saying those scores are indistinguishable given the data. The")
        print("      pooled figures also mix year-varying raw score scales, so the")
        print("      calibrated column is the comparable one.")
    if "mkt_x0" in runs:
        m0 = runs["mkt_x0"][["rid", "p"]].merge(base[["rid", "p"]], on="rid",
                                                suffixes=("_m", "_b"))
        gap = float(np.nanmax(np.abs(m0["p_m"] - m0["p_b"]))) if len(m0) else np.nan
        print(f"    mkt_x0 vs two_stage: max |difference| {gap:.2e}   "
              f"{'ok' if gap < 1e-9 else 'SHOULD BE IDENTICAL'}")

    print("\n" + "=" * 96)
    print("CALIBRATION AND DISCRIMINATION")
    print("=" * 96)
    print(f"  {'model':<18}{'MAE pp':>8}{'within 5pp':>12}{'worst yr':>10}"
          f"{'skill':>9}{'resolution':>12}{'AUC':>8}")
    for name in ARMS:
        if name not in stats:
            continue
        s = stats[name]
        print(f"  {name:<18}{s['mae']:>8.2f}{s['within5']:>12}"
              f"{s['worst_year']:>10.1f}{s['skill']:>+9.4f}"
              f"{s['resolution']:>12.5f}{s['auc']:>8.4f}")

    print("\n" + "=" * 96)
    print("TRACKING ACROSS YEARS")
    print("=" * 96)
    print(f"  {'model':<18}{'Spearman':>10}{'Pearson':>10}{'spread':>12}"
          f"{'reality':>10}{'ratio':>8}")
    for name in ARMS:
        if name not in stats:
            continue
        s = stats[name]
        ratio = (s["pred_spread"] / s["real_spread"]
                 if np.isfinite(s["real_spread"]) and s["real_spread"] > 0
                 else np.nan)
        tag = ("" if not np.isfinite(ratio) else
               "flat" if ratio < 0.85 else "over" if ratio > 1.15 else "ok")
        print(f"  {name:<18}{s['spearman']:>+10.3f}{s['pearson']:>+10.3f}"
              f"{s['pred_spread']:>11.1f}pp{s['real_spread']:>9.1f}pp"
              f"{ratio:>8.2f}{tag:>6}")

    print("\n" + "=" * 96)
    print(f"PAIRED AGAINST {BASELINE}  (same rows, half-year block bootstrap)")
    print("=" * 96)
    print("  Calibration error: NEGATIVE is better. Skill and AUC: positive is")
    print("  better. An interval spanning zero means not established.")
    print(f"\n  {'model':<18}{'metric':<10}{'diff':>9}{'95% CI':>24}"
          f"{'verdict':>18}")
    gains = {}
    for name in ARMS:
        if name == BASELINE or name not in runs:
            continue
        m = runs[name][["rid", "p"]].merge(base[["rid", "p", "y", "date"]],
                                          on="rid", suffixes=("_m", "_b"))
        if len(m) < 1000:
            print(f"  {name:<18}{'too few shared rows':>50}")
            continue
        pa, pb = m["p_m"].to_numpy(), m["p_b"].to_numpy()
        y, dts = m["y"].to_numpy(), m["date"].to_numpy()
        gains[name] = {}
        for label, fn, low_better in (
                ("MAE pp", lambda a, b: V52.mae(a, b), True),
                ("skill", lambda a, b: V52.skill(a, b, ref), False),
                ("AUC", V52.auc, False)):
            d, lo, hi = paired_stat(pa, pb, y, dts, fn, cfg.n_boot, cfg.seed)
            if not np.isfinite(d):
                continue
            gains[name][label] = (d, lo, hi)
            good = (hi < 0) if low_better else (lo > 0)
            bad = (lo > 0) if low_better else (hi < 0)
            v = "better" if good else "WORSE" if bad else "not estab."
            fmt = "{:+.2f}" if label == "MAE pp" else "{:+.3f}"
            print(f"  {name:<18}{label:<10}{fmt.format(d):>9}"
                  f"{fmt.format(lo) + ' to ' + fmt.format(hi):>24}{v:>18}")

    print("\n" + "=" * 96)
    print("HOW TO READ IT")
    print("=" * 96)
    print("  score_curve_all HOLDS MAE AND SKILL WHILE TRACKING RISES")
    print("      -> the win. V54's market signal was real and only its")
    print("         calibration was broken; the curve fixes exactly that. This")
    print("         is a genuine improvement to the deployed model: replace the")
    print("         OLS first stage with the scored one and recalibrate.")
    print("  score_curve_vol ALREADY BEATS two_stage")
    print("      -> the gain is the LEARNER, not the features, and the honest")
    print("         claim shrinks to 'boosting ranks better than OLS here'.")
    print("  score_curve_all STILL LOSES MAE OR SKILL")
    print("      -> the miscalibration was not a monotone distortion, so a")
    print("         monotone curve cannot repair it. The market signal exists in")
    print("         the ordering and cannot be converted into honest numbers with")
    print("         the data available. Report it as a measured ceiling.")
    print("  two_stage_x_mkt BEATS two_stage ON TRACKING AT EQUAL MAE")
    print("      -> the conservative win, and the one to prefer if it lands: five")
    print("         shrunk numbers on top of an unchanged deployed model is far")
    print("         easier to defend than a boosted first stage, and it degrades")
    print("         gracefully because shrink=0 returns the current model exactly.")
    print("  NOTHING BEATS two_stage")
    print("      -> the flatness is a property of what daily price data can say")
    print("         about the next six months, not of the estimator. That closes")
    print("         the question and V52's reactive characterisation stands as the")
    print("         ceiling.")
    print("\n  Try --shrink 0.25 and --shrink 1.0 before concluding on the")
    print("  multiplier arm: shrink is a prior on how much fifteen market")
    print("  episodes can tell you, not a fitted parameter, and it should be")
    print("  reported as the choice it is.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg),
                   "stats": {k: {a: b for a, b in v.items() if a != "by_year"}
                             for k, v in stats.items()},
                   "by_year": {k: v["by_year"] for k, v in stats.items()},
                   "paired": gains}, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_scorecurve_v55(),
# and each value shown is that argument's own default, except QUICK which starts
# True so that a plain run finishes fast. Set it False for the real experiment.
#
# Passing any command-line flag still works and takes over, so the old CLI is
# not lost: it just is not the default way in any more.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # <<< small, fast sanity run. False = the real thing.
    RUN_QUICK           = True
    RUN_CACHE           = 'price_cache_v43'
    RUN_PANEL_CACHE     = 'panel_v54.pkl'   # shares V54's panel - the features are identical
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_EMBARGO_DAYS    = 180
    RUN_MODEL           = 'gbm'
    RUN_FOLDS           = 3                 # purged folds inside each training period for the curve
    RUN_N_BOOT          = 500
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'scorecurve_eval_v55.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_scorecurve_v55.py --quick
    else:
        run_scorecurve_v55(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            panel_cache=RUN_PANEL_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            embargo_days=RUN_EMBARGO_DAYS,
            model=RUN_MODEL,
            folds=RUN_FOLDS,
            n_boot=RUN_N_BOOT,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            out=RUN_OUT,
        )
