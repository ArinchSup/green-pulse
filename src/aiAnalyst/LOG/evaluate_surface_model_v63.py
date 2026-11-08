#!/usr/bin/env python3
"""
evaluate_surface_model_v63.py - is the multi-horizon surface model any good?

TWO SEPARATE QUESTIONS, AND THEY NEED DIFFERENT DATA

    DOES THE MACHINERY WORK?  Given data whose true answer is known, does the
    model recover it? This needs SIMULATED data, because only there is the truth
    known. It is answered by synthetic_truth_test().

    DOES IT BEAT THE ALTERNATIVES ON REAL PRICES?  This needs a real price cache
    and is answered by walk_forward().

They are run separately because they fail for different reasons and a single
number that mixes them tells you nothing. A model can recover simulated truth
perfectly and still add nothing over the closed form on real prices - that would
mean the machinery is sound and the correction is not there, which is a useful
result, not a bug.

THE KNOWN-TRUTH TEST

Driftless geometric Brownian motion touches a barrier d below its start within a
window of volatility sigma_h with probability exactly 2*Phi(ln(1-d)/sigma_h).
So:

    ARM 1  constant-volatility GBM.  The closed form is the truth, EXCEPT for
           one known bias: the model sees daily closes and lows, not the
           continuous path, so a barrier touched and recovered between
           observations is missed. Discrete monitoring therefore makes the
           realised rate slightly LOWER than the closed form. A correct model
           learns that small negative correction and nothing else.

    ARM 2  Student-t innovations, same volatility.  Fat tails put more mass in
           the extreme moves that trigger a barrier, so the realised rate is
           HIGHER than the Gaussian closed form at deep thresholds. A correct
           model learns a positive correction that grows with the threshold.

If the model tracks truth in arm 1 and moves the right way in arm 2, the
machinery is sound. If it cannot do that on data it was handed the answer to, no
result on real prices is worth reading.

THE BASELINES ON REAL PRICES

    closed_form        the barrier formula alone, zero fitted parameters
    per_cell_curve     the DEPLOYED architecture, generalised: a separate
                       twelve-bin empirical curve fitted per (horizon,
                       threshold) cell. This is the thing to beat - it is what
                       you would build without a surface model.
    surface_model      V63

PROTOCOL

Purged walk-forward. Year Y is predicted by a model fitted only on observations
whose forward window closed before Y opened, with an embargo equal to the
LONGEST horizon on the grid - not the horizon of the cell being scored, because
one fit serves every horizon and the fit must be clean for all of them.

Intervals are half-year block bootstrap, because overlapping forward windows make
neighbouring rows anything but independent.

USAGE
  python evaluate_surface_model_v63.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2
import class_ai_pillar2_surface_v63 as S63

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = S63.PRICE_CACHE
STEP = 10
MIN_TRAIN_YEARS = 5
N_BOOT = 300
SEED = 63
CURVE_BINS = 12

# known-truth simulation
SIM_TICKERS = 120
SIM_BARS = 1400
SIM_VOLS = (0.20, 0.35, 0.55, 0.80, 1.10)     # annual, constant within a path
SIM_T_DF = 3.0                                 # Student-t degrees of freedom


# =============================================================================
# METRICS
# =============================================================================
def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill(y, p, base):
    b = brier(y, p)
    b0 = brier(y, np.full(len(y), base))
    return float(1 - b / b0) if b0 > 0 else np.nan


def calib_mae_pp(y, p, n_bins=10):
    """Mean absolute gap between predicted and realised, over probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < n_bins * 5:
        return np.nan
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float(abs(p.mean() - y.mean()) * 100)
    idx = np.clip(np.searchsorted(edges, p, "right") - 1, 0, len(edges) - 2)
    gaps, wts = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        gaps.append(abs(p[m].mean() - y[m].mean()) * 100)
        wts.append(m.sum())
    return float(np.average(gaps, weights=wts)) if gaps else np.nan


def auc(y, p):
    y = np.asarray(y, float)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_bootstrap(dates, values, stat, n_boot, seed):
    """Resample half-year blocks, so overlapping windows stay together."""
    rng = np.random.default_rng(seed)
    blocks = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="6M")
    uniq = blocks.unique()
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        if len(idx) < 50:
            continue
        out.append(stat(idx))
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# =============================================================================
# BASELINE: the deployed architecture, one curve per cell
# =============================================================================
def fit_per_cell_curves(train, vol_ts, n_bins=CURVE_BINS):
    """A twelve-bin empirical curve per (horizon, threshold), P2-style."""
    curves = {}
    for (h, d), g in train.groupby(["horizon_days", "drawdown"]):
        if len(g) < 200:
            continue
        fv = S63.forecast_vol_h_vec(g.assign(_h=h), vol_ts, "_h")
        ok = np.isfinite(fv)
        fv, ev = fv[ok], g["event"].to_numpy(float)[ok]
        if len(fv) < 200:
            continue
        qs = np.unique(np.quantile(fv, np.linspace(0, 1, n_bins + 1)))
        xs, ys = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (fv >= lo) & (fv <= hi)
            if m.sum() < 20:
                continue
            xs.append(float(fv[m].mean()))
            ys.append(float(ev[m].mean()))
        if len(xs) >= 3:
            curves[(int(h), float(d))] = (xs, _pava(ys))
    return curves


def predict_per_cell(test, vol_ts, curves):
    out = np.full(len(test), np.nan)
    for (h, d), g in test.groupby(["horizon_days", "drawdown"]):
        c = curves.get((int(h), float(d)))
        if c is None:
            continue
        fv = S63.forecast_vol_h_vec(g.assign(_h=h), vol_ts, "_h")
        out[test.index.get_indexer(g.index)] = np.interp(fv, c[0], c[1])
    return out


def _pava(y):
    y = [float(v) for v in y]
    val, wt = list(y), [1.0] * len(y)
    i = 0
    while i < len(val) - 1:
        if val[i] <= val[i + 1]:
            i += 1
            continue
        tot = wt[i] + wt[i + 1]
        val[i] = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / tot
        wt[i] = tot
        del val[i + 1], wt[i + 1]
        if i > 0:
            i -= 1
    out = []
    for v, w in zip(val, wt):
        out += [v] * int(round(w))
    return out[:len(y)]


def predict_closed_form(test, vol_ts):
    fv = S63.forecast_vol_h_vec(test, vol_ts)
    sig = S63._sigma_h(fv, test["h_bars"].to_numpy(float))
    return S63.barrier_prob(test["drawdown"].to_numpy(float), sig)


# =============================================================================
# KNOWN-TRUTH SIMULATION
# =============================================================================
def simulate_paths(n_paths, n_bars, vols, mode, seed):
    """
    Daily OHLC where Close is the path and High/Low are the intrabar extremes of
    a finer walk, so the barrier test sees realistic lows rather than closes only.

    ZERO DRIFT IN THE LOG, deliberately. The closed form assumes log-Brownian
    motion with no drift, so the simulation has to match that or the test
    measures the mismatch instead of the model. The first version of this
    function used the risk-neutral step  -0.5*sigma^2*dt + sigma*sqrt(dt)*z,
    which is driftless in the PRICE and therefore carries a -0.5*sigma^2*T drift
    in the LOG - at 55% volatility over a year that is a 15% downward push, and
    it showed up as a clean -11.6pp "closed form reads low" that was entirely my
    own construction.

        mode "gaussian"   the closed form is exactly right, up to discrete
                          monitoring. The model should leave it alone.
        mode "fat_tailed" Student-t innovations per step.
        mode "stoch_vol"  volatility itself follows a mean-reverting log-OU
                          process, so each path is a MIXTURE over volatilities.
    """
    rng = np.random.default_rng(seed)
    sub = 8                                    # intrabar steps
    out = {}
    for i in range(n_paths):
        vol = vols[i % len(vols)]
        dt = 1.0 / 252.0 / sub
        n = n_bars * sub

        if mode == "fat_tailed":
            z = rng.standard_t(SIM_T_DF, n)
            z /= np.sqrt(SIM_T_DF / (SIM_T_DF - 2.0))   # unit variance
        else:
            z = rng.standard_normal(n)

        if mode == "stoch_vol":
            # log-vol mean-reverts to log(vol) with its own noise. Unlike fat
            # tails at the step level, this does NOT wash out as the horizon
            # grows: the horizon return is a mixture over volatility paths, and
            # a mixture of normals is fatter than the normal with the same mean
            # variance no matter how many steps are aggregated.
            kappa, eta = 3.0, 0.9
            lv = np.empty(n)
            lv[0] = np.log(vol)
            w = rng.standard_normal(n)
            for k in range(1, n):
                lv[k] = (lv[k - 1] + kappa * (np.log(vol) - lv[k - 1]) * dt
                         + eta * np.sqrt(dt) * w[k])
            sig = np.exp(lv)
        else:
            sig = np.full(n, vol)

        steps = sig * np.sqrt(dt) * z          # no drift term, by design
        path = 100.0 * np.exp(np.cumsum(steps))
        p = path.reshape(n_bars, sub)
        idx = pd.bdate_range("2008-01-01", periods=n_bars)
        out[f"SIM{i:03d}"] = pd.DataFrame(
            {"Open": p[:, 0], "High": p.max(axis=1), "Low": p.min(axis=1),
             "Close": p[:, -1], "Volume": 1e6}, index=idx)
    return out


class _CacheShim:
    """Lets build_surface_observations read simulated frames as if from disk."""
    def __init__(self, frames):
        self.frames = frames
        self._orig = None

    def __enter__(self):
        self._orig = P2.load_prices
        P2.load_prices = lambda t, cache_dir=None: self.frames.get(t)
        return self

    def __exit__(self, *a):
        P2.load_prices = self._orig


def synthetic_truth_test(n_paths=SIM_TICKERS, n_bars=SIM_BARS, seed=SEED,
                         verbose=True):
    """Arm 1 constant-vol Gaussian, arm 2 fat-tailed. See the module docstring."""
    def say(*a):
        if verbose:
            print(*a)

    say("\n" + "=" * 88)
    say("  KNOWN-TRUTH TEST - does the machinery recover an answer we already know?")
    say("=" * 88)

    results = {}
    for arm in ("gaussian", "fat_tailed", "stoch_vol"):
        frames = simulate_paths(n_paths, n_bars, SIM_VOLS, arm, seed)
        with _CacheShim(frames):
            obs = S63.build_surface_observations(
                list(frames), cache_dir=None, step=STEP,
                horizons=[30, 90, 180, 365],
                thresholds=[0.10, 0.20, 0.30, 0.40], verbose=False)
            vol_ts = S63.fit_vol_term_structure(obs, verbose=False)
            model = S63.fit_surface(obs, vol_ts=vol_ts, verbose=False)
            p_th = predict_closed_form(obs, vol_ts)
            X, margin, y, _, _, _ = S63._design(obs, vol_ts)
            p_md = model["booster"].predict_proba(X, base_margin=margin)[:, 1]

        say(f"\n  ARM: {arm}   {len(obs):,} cells from "
            f"{obs.groupby(['ticker','date']).ngroups:,} dates")
        say(f"  {'horizon':>8s} {'fall':>6s} {'realised':>9s} "
            f"{'closed form':>12s} {'MODEL':>8s} "
            f"{'form err':>9s} {'model err':>10s}")
        rows = []
        for (h, d), g in obs.groupby(["horizon_days", "drawdown"]):
            m = obs.index.get_indexer(g.index)
            real = float(g["event"].mean())
            th, md = float(p_th[m].mean()), float(p_md[m].mean())
            rows.append({"horizon": int(h), "drawdown": float(d),
                         "realised": real, "closed_form": th, "model": md,
                         "form_err_pp": (th - real) * 100,
                         "model_err_pp": (md - real) * 100})
            say(f"  {int(h):7d}d {d:6.0%} {real*100:8.1f}% {th*100:11.1f}% "
                f"{md*100:7.1f}% {(th-real)*100:+8.1f}pp "
                f"{(md-real)*100:+9.1f}pp")
        r = pd.DataFrame(rows)
        f_mae = float(r["form_err_pp"].abs().mean())
        m_mae = float(r["model_err_pp"].abs().mean())
        say(f"\n    mean |error|   closed form {f_mae:5.2f}pp    "
            f"model {m_mae:5.2f}pp")
        say(f"    closed-form bias is {'NEGATIVE' if r['form_err_pp'].mean() < 0 else 'POSITIVE'} "
            f"on average ({r['form_err_pp'].mean():+.2f}pp)")
        results[arm] = {"cells": r.to_dict("records"),
                        "form_mae_pp": f_mae, "model_mae_pp": m_mae,
                        "form_bias_pp": float(r["form_err_pp"].mean()),
                        "model_bias_pp": float(r["model_err_pp"].mean())}

    say("\n" + "-" * 88)
    say("  WHAT EACH ARM WAS SUPPOSED TO SHOW")
    say("-" * 88)
    g, f, v = (results["gaussian"], results["fat_tailed"],
               results["stoch_vol"])

    say("  1. gaussian - the closed form is the TRUTH here, so the test is "
        "whether the")
    say("     learned stage leaves a correct baseline alone instead of "
        "'correcting' noise.")
    say(f"       closed form {g['form_mae_pp']:.2f}pp   "
        f"model {g['model_mae_pp']:.2f}pp")
    if g["model_mae_pp"] <= g["form_mae_pp"] + 0.30:
        say("       -> PASS: the model did not damage a baseline that was "
            "already right.")
    else:
        say("       -> FAIL: the model made a correct baseline worse. That is "
            "overfitting,")
        say("          and nothing below should be believed until it is fixed.")

    say("  2. fat_tailed - Student-t steps. The honest expectation is that this "
        "barely")
    say("     moves anything: 2,016 steps aggregate to near-Gaussian by the "
        "central limit")
    say("     theorem, so daily fat tails cannot explain long-horizon barrier "
        "risk.")
    say(f"       closed-form bias {g['form_bias_pp']:+.2f}pp (gaussian) -> "
        f"{f['form_bias_pp']:+.2f}pp (fat tails)")
    say(f"       -> moved {abs(f['form_bias_pp'] - g['form_bias_pp']):.2f}pp. "
        + ("small, as the CLT says it should be"
           if abs(f["form_bias_pp"] - g["form_bias_pp"]) < 2.0
           else "larger than the CLT argument suggests - worth a look"))

    say("  3. stoch_vol - volatility itself moves. A mixture of normals is "
        "fatter than")
    say("     the normal with the same average variance, and unlike step-level "
        "fat tails")
    say("     this does NOT wash out with horizon. The closed form should now "
        "read LOW,")
    say("     and the learned stage should be able to see it.")
    say(f"       closed-form bias {v['form_bias_pp']:+.2f}pp   "
        f"MAE {v['form_mae_pp']:.2f}pp -> model {v['model_mae_pp']:.2f}pp")
    if v["form_bias_pp"] < g["form_bias_pp"] - 0.5:
        say("       -> the bias appeared, in the predicted direction.")
    else:
        say("       -> the bias did NOT appear; the vol-of-vol may be too "
            "small to bite.")
    if v["model_mae_pp"] < v["form_mae_pp"]:
        say(f"       -> and the model removed "
            f"{100*(1 - v['model_mae_pp']/v['form_mae_pp']):.0f}% of it. "
            f"This is the case the surface model exists for.")
    else:
        say("       -> but the model did not remove it.")
    return results


# =============================================================================
# WALK-FORWARD ON REAL PRICES
# =============================================================================
def walk_forward(obs, min_train_years=MIN_TRAIN_YEARS, n_boot=N_BOOT,
                 seed=SEED, verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    obs = obs.sort_values("date").reset_index(drop=True)
    obs["year"] = pd.DatetimeIndex(obs["date"]).year
    years = sorted(obs["year"].unique())
    embargo = pd.Timedelta(days=max(obs["horizon_days"]))

    preds = {k: [] for k in ("closed_form", "per_cell_curve", "surface_model")}
    keep = []
    say("\n" + "=" * 88)
    say("  PURGED WALK-FORWARD ON REAL PRICES")
    say("=" * 88)
    say(f"  embargo {embargo.days}d - the LONGEST horizon on the grid, because "
        f"one fit serves them all")

    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = obs[pd.DatetimeIndex(obs["date"]) < start - embargo]
        te = obs[obs["year"] == y]
        if len(tr) < 5000 or len(te) < 500:
            continue
        vol_ts = S63.fit_vol_term_structure(tr, verbose=False)
        try:
            model = S63.fit_surface(tr, vol_ts=vol_ts, verbose=False)
        except RuntimeError:
            continue
        curves = fit_per_cell_curves(tr, vol_ts)

        te = te.reset_index(drop=True)
        X, margin, _, _, _, _ = S63._design(te, vol_ts)
        preds["surface_model"].append(
            model["booster"].predict_proba(X, base_margin=margin)[:, 1])
        preds["closed_form"].append(predict_closed_form(te, vol_ts))
        preds["per_cell_curve"].append(predict_per_cell(te, vol_ts, curves))
        keep.append(te)
        say(f"    {y}: train {len(tr):,} cells -> test {len(te):,}")

    if not keep:
        raise RuntimeError("no walk-forward folds were usable")
    te = pd.concat(keep, ignore_index=True)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    y = te["event"].to_numpy(int)
    base = float(y.mean())

    say(f"\n  {len(te):,} out-of-sample cells, base rate {base:.4f}")
    say(f"  {'arm':16s} {'brier':>8s} {'skill':>8s} {'calib MAE':>10s} {'AUC':>7s}")
    summary = {}
    for k, p in P.items():
        ok = np.isfinite(p)
        summary[k] = {"brier": brier(y[ok], p[ok]),
                      "skill": brier_skill(y[ok], p[ok], base),
                      "calib_mae_pp": calib_mae_pp(y[ok], p[ok]),
                      "auc": auc(y[ok], p[ok]), "n": int(ok.sum())}
        s = summary[k]
        say(f"  {k:16s} {s['brier']:8.5f} {s['skill']:8.4f} "
            f"{s['calib_mae_pp']:9.2f}pp {s['auc']:7.4f}")
    say("  NOTE the pooled AUC is near-meaningless on a surface and is printed "
        "only to stop")
    say("       anyone computing it again. Cells run from a 0.3% base rate "
        "(14d/50%) to 70%")
    say("       (365d/10%), so most of the apparent discrimination is just "
        "knowing WHICH CELL")
    say("       a row sits in. This project has already read a pooled metric "
        "wrong three")
    say("       times (V55, V59, V61). Judge discrimination inside a cell or "
        "not at all.")

    say(f"\n  {'comparison':38s} {'delta MAE pp':>13s} {'95% block bootstrap':>24s}")
    for a, b in (("surface_model", "closed_form"),
                 ("surface_model", "per_cell_curve"),
                 ("per_cell_curve", "closed_form")):
        pa, pb = P[a], P[b]
        ok = np.isfinite(pa) & np.isfinite(pb)
        d = calib_mae_pp(y[ok], pa[ok]) - calib_mae_pp(y[ok], pb[ok])
        lo, hi = block_bootstrap(
            te["date"].to_numpy()[ok], None,
            lambda idx: calib_mae_pp(y[ok][idx], pa[ok][idx])
            - calib_mae_pp(y[ok][idx], pb[ok][idx]), n_boot, seed)
        verdict = ("BETTER" if hi < 0 else "WORSE" if lo > 0
                   else "not established")
        say(f"  {a + ' vs ' + b:38s} {d:+13.3f} "
            f"{f'[{lo:+.3f}, {hi:+.3f}]':>24s}  {verdict}")

    say(f"\n  PER-CELL CALIBRATION ERROR (pp), surface model")
    te2 = te.copy()
    te2["p"] = P["surface_model"]
    piv = te2.pivot_table(index="horizon_days", columns="drawdown",
                          values=["p", "event"], aggfunc="mean")
    err = (piv["p"] - piv["event"]) * 100
    say("    " + err.round(1).to_string().replace("\n", "\n    "))

    bad_h = S63.monotone_violations(te2.assign(p_model=te2["p"]),
                                    key=("ticker", "date", "drawdown"))
    say(f"\n  monotonicity across every out-of-sample row: "
        f"{'clean' if bad_h == 0 else f'{bad_h} violations'}")
    return {"summary": summary, "n_test": int(len(te)), "base_rate": base}


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_surface_model_v63(price_cache=PRICE_CACHE, step=STEP,
                          min_train_years=MIN_TRAIN_YEARS, n_boot=N_BOOT,
                          seed=SEED, sim_paths=SIM_TICKERS, sim_bars=SIM_BARS,
                          skip_real=False, out="surface_model_v63.json",
                          verbose=True):
    print("=" * 88)
    print("V63 EVALUATION - MULTI-HORIZON RISK SURFACE")
    print("=" * 88)

    payload = {}
    payload["known_truth"] = synthetic_truth_test(sim_paths, sim_bars, seed,
                                                  verbose=verbose)

    if not skip_real:
        tickers = S63._tickers_from_cache(price_cache)
        if not tickers:
            print(f"\n  no price cache at {price_cache}; skipping the real-price run")
        else:
            obs = S63.build_surface_observations(tickers, price_cache,
                                                 step=step, verbose=verbose)
            payload["walk_forward"] = walk_forward(obs, min_train_years,
                                                   n_boot, seed, verbose)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n  wrote {out}")
    return payload


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sim-paths", type=int, default=SIM_TICKERS)
    ap.add_argument("--sim-bars", type=int, default=SIM_BARS)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default="surface_model_v63.json")
    return ap


def main(argv=None):
    cfg = _parser().parse_args(argv)
    return run_surface_model_v63(
        price_cache=cfg.price_cache, step=cfg.step,
        min_train_years=cfg.min_train_years, n_boot=cfg.n_boot, seed=cfg.seed,
        sim_paths=cfg.sim_paths, sim_bars=cfg.sim_bars,
        skip_real=cfg.skip_real, out=cfg.out)


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE     = PRICE_CACHE
    RUN_STEP            = STEP
    RUN_MIN_TRAIN_YEARS = MIN_TRAIN_YEARS
    RUN_N_BOOT          = N_BOOT
    RUN_SEED            = SEED
    RUN_SIM_PATHS       = SIM_TICKERS   # simulated names for the known-truth test
    RUN_SIM_BARS        = SIM_BARS      # bars per simulated name
    RUN_SKIP_REAL       = False         # True runs only the known-truth test
    RUN_OUT             = "surface_model_v63.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                     # e.g. python evaluate_surface_model_v63.py --skip-real
    else:
        run_surface_model_v63(price_cache=RUN_PRICE_CACHE, step=RUN_STEP,
                              min_train_years=RUN_MIN_TRAIN_YEARS,
                              n_boot=RUN_N_BOOT, seed=RUN_SEED,
                              sim_paths=RUN_SIM_PATHS, sim_bars=RUN_SIM_BARS,
                              skip_real=RUN_SKIP_REAL, out=RUN_OUT)
