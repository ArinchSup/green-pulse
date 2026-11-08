#!/usr/bin/env python3
"""
deployment_preflight.py - check that the parts still agree with each other.

WHY THIS EXISTS

Two failures in one evening, both of the same kind:

  A forward-window floor of 80 bars, correct for a 180-day horizon, silently
  rejected EVERY observation at 30 days. The calibration reported zero
  observations and no reason.

  Tier thresholds set against a 10.7% base rate were left in place when the base
  rate moved to 13.2% and the curve's top end moved from 33.5% to 41.4%. Nothing
  errored; twice as many names quietly dropped to the smallest position size.

Neither was a bug in the code. Both were constants that had stopped agreeing with
each other, in a system where the horizon lives in one file, the threshold in
another, the trade geometry in a third and the fitted curve in a JSON file on
disk. Nothing checks that those four still describe the same thing, so this does.

Every check below either passes, warns, or fails, and says what to do about it.
A FAIL means a number somewhere downstream is wrong rather than missing, which is
the dangerous case. Exit code is non-zero if anything fails.

USAGE
  python deployment_preflight.py
  python deployment_preflight.py --tickers NVDA AMD MU --equity 25000
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

# --- config ------------------------------------------------------------------
DEFAULT_TICKERS = None      # None -> a few names from the deploy list
DEFAULT_EQUITY = 10000.0

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results = []


def check(name, status, detail="", fix=""):
    _results.append((name, status, detail, fix))
    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name}")
    if detail:
        print(f"          {detail}")
    if fix and status != PASS:
        print(f"          -> {fix}")


class _Cfg:
    """The checks read cfg.tickers and cfg.equity. Keeping that shape means
    run_preflight() can expose real parameters without rewriting every check."""

    def __init__(self, tickers, equity):
        self.tickers = tickers
        self.equity = equity


def run_preflight(tickers=DEFAULT_TICKERS, equity=DEFAULT_EQUITY):
    """
    Run every consistency check. Returns (exit_code, results) where results is a
    list of (name, status, detail, fix) and status is "PASS" | "WARN" | "FAIL".

        code, results = run_preflight(["NVDA", "AMD"], equity=25000)
        [r for r in results if r[1] == "FAIL"]

    Safe to call repeatedly in one process - the results list is reset here
    rather than accumulating across calls.
    """
    _results.clear()
    cfg = _Cfg(tickers, equity)

    print("=" * 88)
    print("DEPLOYMENT PREFLIGHT")
    print("=" * 88)

    # ---------------------------------------------------------------- imports
    try:
        import class_ai_pillar2_risk_v2 as P2
        check("risk module imports", PASS,
              f"HORIZON_DAYS={P2.HORIZON_DAYS}, DRAWDOWN={P2.DRAWDOWN:.0%}, "
              f"HORIZON={P2.HORIZON}")
    except Exception as e:
        check("risk module imports", FAIL, f"{type(e).__name__}: {e}",
              "run this next to class_ai_pillar2_risk_v2.py")
        return 1, list(_results)

    PL = None
    for mod in ("class_ai_pipeline_v3", "class_ai_pipeline_v2"):
        try:
            PL = __import__(mod)
            break
        except ImportError:
            continue
        except Exception as e:
            check("pipeline imports", FAIL, f"{mod}: {type(e).__name__}: {e}")
            PL = None
            break
    if PL is None:
        check("pipeline imports", FAIL, "neither v3 nor v2 could be imported")
    else:
        same = PL.P2 is P2
        check("pipeline imports the v2 risk module", PASS if same else FAIL,
              f"{PL.__name__} uses {PL.P2.__name__}",
              f"change the import in {PL.__name__}.py to "
              "class_ai_pillar2_risk_v2")
        has_pf = hasattr(PL, "apply_portfolio")
        check("pipeline has a portfolio layer", PASS if has_pf else WARN,
              f"{PL.__name__}"
              + ("" if has_pf else " sizes each name in isolation"),
              "v2 has no portfolio budget: six names at the concentration "
              "ceiling is 120% of equity. Use class_ai_pipeline_v3.")

    try:
        import class_ai_portfolio as PF
        check("portfolio module imports the v2 risk module",
              PASS if PF.P2 is P2 else FAIL, f"uses {PF.P2.__name__}")
    except Exception as e:
        check("portfolio module imports", FAIL, f"{type(e).__name__}: {e}")
        PF = None

    # ------------------------------------------------------ calibration file
    if not os.path.exists(P2.CALIB_FILE):
        check("calibration file present", FAIL, f"{P2.CALIB_FILE} not found",
              "python class_ai_pillar2_risk_v2.py --calibrate")
        return 1, list(_results)
    with open(P2.CALIB_FILE, encoding="utf-8") as f:
        calib = json.load(f)
    check("calibration file present", PASS,
          f"{P2.CALIB_FILE}, {calib.get('n_observations', '?')} observations, "
          f"built {calib.get('built', 'unknown')}")

    # The check that matters most: the curve on disk was fitted for ONE question.
    # If the module now asks a different one, every probability is an answer to
    # the wrong question and nothing errors.
    ch, cd = calib.get("horizon_days"), calib.get("drawdown")
    if ch is None or cd is None:
        check("calibration matches the module's question", WARN,
              "the file does not record its horizon or threshold",
              "recalibrate so the file carries them")
    elif int(ch) != int(P2.HORIZON_DAYS) or abs(float(cd) - P2.DRAWDOWN) > 1e-9:
        check("calibration matches the module's question", FAIL,
              f"curve fitted for {cd:.0%} in {ch}d, module asks "
              f"{P2.DRAWDOWN:.0%} in {P2.HORIZON_DAYS}d",
              "python class_ai_pillar2_risk_v2.py --calibrate")
    else:
        check("calibration matches the module's question", PASS,
              f"both {P2.DRAWDOWN:.0%} in {P2.HORIZON_DAYS}d")

    sh = calib.get("shrinkage", {})
    if "features" in sh:
        check("volatility model is the V53 feature set", PASS,
              f"{'+'.join(sh['features'])}, in-sample R2 {sh.get('r2', float('nan')):.3f}")
    else:
        check("volatility model is the V53 feature set", WARN,
              "calibration predates V53; the module falls back to one "
              "close-to-close feature",
              "recalibrate to pick up the Yang-Zhang forecast")

    # ------------------------------------------------- horizon vs trade plan
    try:
        from trade_config import HORIZON_CONFIGS
        tc = HORIZON_CONFIGS[P2.HORIZON]
        trade_days = tc.get("eval_days") or int(
            round(tc.get("lookahead_bars", 60) * 365.25 / 252))
        if P2.HORIZON_DAYS < 0.9 * trade_days:
            check("risk window covers the trade", FAIL,
                  f"risk window {P2.HORIZON_DAYS}d, {P2.HORIZON} trade runs "
                  f"{trade_days}d",
                  f"set HORIZON_DAYS near {trade_days}, or shorten the trade "
                  f"horizon - a probability about a shorter window than the "
                  f"position is open for understates the exposure")
        else:
            check("risk window covers the trade", PASS,
                  f"risk {P2.HORIZON_DAYS}d against a {trade_days}d trade")
    except Exception as e:
        check("risk window covers the trade", WARN, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------ the curve
    curve = calib.get("curve") or []
    xs = [p["fvol"] for p in curve]
    ys = [p["p"] for p in curve]
    if len(curve) < 4:
        check("drawdown curve usable", FAIL, f"{len(curve)} points")
    else:
        # An empirical curve from a dozen bins can dip a little on noise. A
        # decrease worth acting on is one large relative to the curve's own
        # range, since that is what actually reorders names.
        rng_y = max(ys) - min(ys)
        drops = [a - b for a, b in zip(ys, ys[1:]) if b < a]
        worst = max(drops) if drops else 0.0
        if rng_y > 0 and worst > 0.05 * rng_y:
            check("drawdown curve monotone", FAIL,
                  f"probability falls by {worst:.1%} at one step, "
                  f"{worst / rng_y:.0%} of the curve's range",
                  "a curve that falls where volatility rises reorders names; "
                  "refit, or pass the bin rates through isotonic regression")
        elif drops:
            check("drawdown curve monotone", WARN,
                  f"largest dip {worst:.2%}, {worst / rng_y:.0%} of the range",
                  "small enough to be bin noise; watch it if it grows")
        else:
            check("drawdown curve monotone", PASS,
                  f"{len(curve)} points, {min(ys):.1%} to {max(ys):.1%} over "
                  f"{min(xs):.0f}% to {max(xs):.0f}% volatility")
    base = calib.get("base_rate")
    if base:
        check("base rate recorded", PASS, f"{base:.1%} of observations")

    # --------------------------------------------- tier thresholds vs the curve
    # These are absolute probabilities in the pipeline, but they were chosen as
    # MULTIPLES of the base rate. When the question changes the base rate moves
    # and the thresholds do not follow, so the same numbers bite harder or softer
    # than intended without anything erroring.
    if PL is not None and base:
        bounds = [b for b, _, _ in PL.RISK_TIERS]
        # Do NOT test these against the multiples they happened to be set at -
        # that encodes one historical accident and cries wolf on any deliberate
        # choice. Test the property that matters: do the tiers actually divide
        # the population, or has the curve moved out from under them?
        print(f"          tiers at " + ", ".join(f"{b:.0%}" for b in bounds)
              + f" = " + ", ".join(f"{b / base:.2f}x" for b in bounds)
              + " the base rate")
        share = []
        edges = [0.0] + bounds + [1.0]
        for lo, hi in zip(edges, edges[1:]):
            share.append(sum(1 for y in ys if lo <= y < hi) / len(ys))
        names = [lbl for _, _, lbl in PL.RISK_TIERS] + ["excessive"]
        detail = ", ".join(f"{nm} {sh:.0%}"
                           for nm, sh in zip(names, share))
        # The asymmetry is deliberate. Most names being LOW risk is the expected
        # shape - the tiers exist to isolate a risky tail, not to split the
        # population evenly. Most names landing in the TOP tier is the
        # misconfiguration, because it silently cuts every position to the
        # smallest size. Failing on a large bottom tier would reject a correct
        # setup.
        empty = [nm for nm, sh in zip(names, share) if sh == 0]
        if empty:
            check("tiers divide the population", FAIL, detail,
                  f"{', '.join(empty)} is unreachable - the curve no longer "
                  f"spans these boundaries, so that tier can never fire")
        elif share[-1] > 0.30:
            check("tiers divide the population", FAIL, detail,
                  f"{share[-1]:.0%} of the curve sits in the top tier, so most "
                  f"names get the smallest position size; the boundaries were "
                  f"set for a different base rate and have not followed it")
        elif share[0] > 0.70:
            check("tiers divide the population", WARN, detail,
                  "nearly everything is in the lowest tier, so the tiers are "
                  "barely doing anything - not wrong, but check it is intended")
        else:
            check("tiers divide the population", PASS, detail)

        # The share check above only catches a SEVERE mismatch. A tier named
        # "excessive" is a claim relative to the base rate, so the boundary has
        # to sit meaningfully above it or the name promises more than it delivers
        # - and a drift big enough to double how many names get the smallest
        # position can still leave the shares inside the bands above. Test the
        # multiples too, as a band rather than as exact values.
        lo_mult, hi_mult = bounds[0] / base, bounds[-1] / base
        md = f"lowest boundary {lo_mult:.2f}x base, top boundary {hi_mult:.2f}x"
        if hi_mult < 1.2:
            check("tier boundaries scaled to the base rate", FAIL, md,
                  f"the top tier fires on names barely above average risk; set "
                  f"it near {2.3 * base:.2f} so 'excessive' means excessive")
        elif hi_mult < 2.0 or lo_mult > 1.15:
            check("tier boundaries scaled to the base rate", WARN, md,
                  f"the boundaries were set against a different base rate. At "
                  f"{base:.1%} the equivalents are about "
                  f"{0.91 * base:.2f}, {1.67 * base:.2f}, {2.35 * base:.2f}")
        else:
            check("tier boundaries scaled to the base rate", PASS, md)

        veto = getattr(PL, "MAX_DRAWDOWN_PROB", None)
        check("drawdown veto state", PASS,
              "off - V51 showed refusing those names cost more than it saved"
              if veto is None else f"ON at {veto:.0%}")

    # ------------------------------------------------------------- scoring
    tickers = cfg.tickers
    if not tickers:
        try:
            from pillar2_v43_universe import DEPLOY_UNIVERSE
            tickers = list(DEPLOY_UNIVERSE)[:4]
        except Exception:
            tickers = []
    if not tickers:
        check("scored a ticker", WARN, "no tickers to try",
              "pass --tickers NVDA AMD")
        return _finish(), list(_results)

    scored, r = None, None
    for t in tickers:
        try:
            r = P2.assess(t, equity=cfg.equity, calib=calib)
        except Exception:
            check("assess() runs", FAIL, traceback.format_exc().strip()
                  .splitlines()[-1])
            return _finish(), list(_results)
        if r:
            scored = t
            break
    if r is None:
        check("assess() returns a result", FAIL,
              f"none of {tickers} produced output",
              "check the price cache has these names with enough history")
        return _finish(), list(_results)
    check("assess() runs", PASS, f"scored {scored}")

    # the pipeline looks the probability up by a key built from the constants -
    # if that key is absent every candidate silently lands in the unknown tier
    key = f"prob_drawdown_{int(P2.DRAWDOWN * 100)}pct_{P2.HORIZON_DAYS}d"
    if PL is not None and getattr(PL, "PROB_KEY", key) != key:
        check("pipeline reads the right probability key", FAIL,
              f"pipeline wants {PL.PROB_KEY}, module emits {key}")
    elif key not in r["risk"]:
        check("pipeline reads the right probability key", FAIL,
              f"{key} missing from the risk block")
    else:
        check("pipeline reads the right probability key", PASS,
              f"{key} = {r['risk'][key]:.1%}")

    if r.get("return_forecast") is None and "disclaimer" in r:
        check("refusals intact", PASS,
              "no return forecast, disclaimer present")
    else:
        check("refusals intact", FAIL,
              "the module is emitting a return forecast",
              "Pillar 2 has no validated basis for one")

    s, lv = r["sizing"], r["levels"]
    if s["position_pct_of_equity"] is not None and \
            s["position_pct_of_equity"] <= s["position_cap_pct"] + 1e-9:
        check("position cap respected", PASS,
              f"{s['position_pct_of_equity']}% of equity, cap "
              f"{s['position_cap_pct']}%, effective risk "
              f"{s['effective_risk_pct']}%")
    else:
        check("position cap respected", FAIL, json.dumps(s))

    if lv["stop"] < lv["entry"] < lv["target"]:
        check("levels ordered", PASS,
              f"stop {lv['stop']} < entry {lv['entry']} < target {lv['target']}, "
              f"R:R {lv['rr']}")
    else:
        check("levels ordered", FAIL, json.dumps(lv))

    # ------------------------------------------------------------- pipeline
    if PL is not None:
        try:
            out = PL.evaluate_trade_setup(scored, equity=cfg.equity,
                                          calib=calib, verbose=False)
            if "final_decision" not in out:
                check("pipeline returns final_decision", FAIL,
                      f"keys: {sorted(out)[:8]}")
            else:
                check("pipeline returns final_decision", PASS,
                      f"{scored} -> {out['final_decision']}"
                      + (f" ({out.get('reason', '')[:60]})"
                         if out["final_decision"] != "EXECUTE" else
                         f", tier {out['telemetry'].get('risk_tier')}"))
        except Exception:
            check("pipeline runs", FAIL,
                  traceback.format_exc().strip().splitlines()[-1],
                  "the news pillar may be missing; that is fine for a smoke "
                  "test but the gate will reject everything")

    # ------------------------------------------------------------ portfolio
    if PF is not None:
        _portfolio_checks(PF, PL, P2, calib, cfg, curve)

    return _finish(), list(_results)


def _portfolio_checks(PF, PL, P2, calib, cfg, curve):
    """The book-level limits, and whether they can actually be satisfied."""
    name_cap = getattr(P2, "MAX_POSITION_PCT", 20.0)
    per_trade = (getattr(PL, "BASE_RISK_PER_TRADE", 0.02) * 100.0
                 if PL is not None else 2.0)

    # --- limits have to be orderable, or one of them can never bind ----------
    ordered = name_cap <= PF.MAX_CLUSTER_PCT <= PF.MAX_GROSS_PCT
    check("portfolio limits are ordered", PASS if ordered else FAIL,
          f"name {name_cap:.0f}% <= cluster {PF.MAX_CLUSTER_PCT:.0f}% "
          f"<= gross {PF.MAX_GROSS_PCT:.0f}%",
          "a cluster ceiling below the per-name ceiling makes the per-name "
          "ceiling unreachable; a gross ceiling below the cluster ceiling makes "
          "the cluster ceiling decoration")

    # --- one trade must not breach the book budget by itself -----------------
    if per_trade <= PF.MAX_PORTFOLIO_RISK_PCT:
        check("one trade fits inside the book risk budget", PASS,
              f"{per_trade:.1f}% per trade against a {PF.MAX_PORTFOLIO_RISK_PCT:.1f}% "
              f"book budget - room for about "
              f"{PF.MAX_PORTFOLIO_RISK_PCT / per_trade:.1f} full-size names")
    else:
        check("one trade fits inside the book risk budget", FAIL,
              f"{per_trade:.1f}% per trade exceeds the {PF.MAX_PORTFOLIO_RISK_PCT:.1f}% "
              f"book budget", "every single trade would be scaled down; raise "
              "MAX_PORTFOLIO_RISK_PCT or lower BASE_RISK_PER_TRADE")

    # --- the book drawdown ceiling has to be inside the curve's range --------
    target, src = PF.dd_ceiling(calib)
    p_floor = min(float(p["p"]) for p in curve)
    p_top = max(float(p["p"]) for p in curve)
    if not np.isfinite(target):
        check("book drawdown ceiling readable", WARN,
              "no base rate in the calibration and no override set",
              "set MAX_PORTFOLIO_DD_PROB, or recalibrate so base_rate is written")
    elif target < p_floor:
        check("book drawdown ceiling reachable", FAIL,
              f"ceiling {target:.1%} ({src}) is below the curve's minimum "
              f"{p_floor:.1%}",
              "no amount of scaling can satisfy it, so the constraint is "
              "skipped entirely - raise it or recalibrate")
    elif target > p_top:
        check("book drawdown ceiling reachable", WARN,
              f"ceiling {target:.1%} ({src}) is above the curve's maximum "
              f"{p_top:.1%}", "the constraint can never bind; it is decoration")
    else:
        check("book drawdown ceiling reachable", PASS,
              f"{target:.1%} ({src}), inside the curve's {p_floor:.1%}-"
              f"{p_top:.1%} range")

    # --- and the limits must actually be enforced on a book that breaches ----
    # Built from the live calibration and the real price cache where possible,
    # so this exercises the deployed path rather than a fixture.
    names = (cfg.tickers or ["NVDA", "AMD", "AVGO", "MU", "MRVL", "SMCI"])[:6]
    cands, failed = PF.assess_candidates(names, calib=calib, equity=cfg.equity)
    if len(cands) < 2:
        check("portfolio limits enforced", WARN,
              f"only {len(cands)} of {len(names)} names could be assessed",
              "pass --tickers with names that have cached prices")
        return
    # force a breach: every name at the per-name ceiling
    for c in cands:
        c["position_pct_of_equity"] = name_cap
    try:
        book = PF.build_book(cands, calib=calib, equity=cfg.equity)
    except Exception:
        check("portfolio limits enforced", FAIL,
              traceback.format_exc().strip().splitlines()[-1])
        return
    p = book["portfolio"]
    bad = []
    if p["gross_pct"] > PF.MAX_GROSS_PCT + 1e-6:
        bad.append(f"gross {p['gross_pct']}% > {PF.MAX_GROSS_PCT}%")
    if p["risk_at_stops_pct"] > PF.MAX_PORTFOLIO_RISK_PCT + 1e-6:
        bad.append(f"risk {p['risk_at_stops_pct']}% > "
                   f"{PF.MAX_PORTFOLIO_RISK_PCT}%")
    for cl in book["clusters"]:
        if cl["gross_pct_after"] > PF.MAX_CLUSTER_PCT + 1e-6:
            bad.append(f"cluster {'+'.join(cl['members'])} "
                       f"{cl['gross_pct_after']}%")
    if bad:
        check("portfolio limits enforced", FAIL, "; ".join(bad),
              "build_book returned a book that breaches its own limits")
    else:
        check("portfolio limits enforced", PASS,
              f"{len(cands)} names forced to {name_cap:.0f}% each "
              f"({len(cands) * name_cap:.0f}% gross) came back at "
              f"{p['gross_pct']}% gross, {p['risk_at_stops_pct']}% risk, "
              f"bound by {p['binding_constraint']}")
        check("correlation measured, not assumed", PASS
              if not book["correlation"]["assumed_for"] else WARN,
              f"mean pairwise {book['correlation']['mean_pairwise']} over "
              f"{book['correlation']['window_bars']} bars"
              + (f"; assumed for {', '.join(book['correlation']['assumed_for'])}"
                 if book["correlation"]["assumed_for"] else ""),
              "names without enough overlapping history get the mean measured "
              "correlation, which is a guess")

    # --- the pipeline must actually call the layer ---------------------------
    if PL is not None and hasattr(PL, "apply_portfolio"):
        fake = []
        for c in cands:
            fake.append({
                "ticker": c["ticker"], "final_decision": "EXECUTE",
                "risk": {"forecast_vol_annual_pct": c["forecast_vol_annual_pct"]},
                "levels": {"entry": c["entry"], "stop_pct": c["stop_pct"]},
                "sizing": {"position_pct_of_equity": name_cap,
                           "stop_pct": c["stop_pct"],
                           "effective_risk_pct": name_cap * c["stop_pct"] / 100.0},
                "telemetry": {}})
        try:
            PL.apply_portfolio(fake, calib=calib, equity=cfg.equity,
                               verbose=False)
            kept = [f for f in fake if f["final_decision"] == "EXECUTE"]
            tot = sum(f["sizing"]["effective_risk_pct"] for f in kept)
            rewrote = all("sizing_before_portfolio" in f for f in kept)
            ok = tot <= PF.MAX_PORTFOLIO_RISK_PCT + 1e-6 and rewrote
            check("pipeline hands sizes to the portfolio layer",
                  PASS if ok else FAIL,
                  f"{len(kept)} names, {tot:.2f}% total risk at stops, "
                  f"pre-portfolio sizes preserved: {rewrote}",
                  "apply_portfolio must replace sizing with the book-scaled "
                  "numbers, or the pipeline reports one size and trades another")
        except Exception:
            check("pipeline hands sizes to the portfolio layer", FAIL,
                  traceback.format_exc().strip().splitlines()[-1])


def _finish():
    n_fail = sum(1 for _, s, _, _ in _results if s == FAIL)
    n_warn = sum(1 for _, s, _, _ in _results if s == WARN)
    print("\n" + "=" * 88)
    print(f"{len(_results)} checks | {n_fail} failed | {n_warn} warned")
    if n_fail:
        print("\nFAILED:")
        for name, s, detail, fix in _results:
            if s == FAIL:
                print(f"  {name}: {detail}")
                if fix:
                    print(f"    -> {fix}")
        print("\nA failure means a number downstream is WRONG rather than")
        print("missing. Do not deploy or quote results until these clear.")
    elif n_warn:
        print("\nNothing broken. Warnings are worth reading before deploying.")
    else:
        print("\nAll clear.")
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS,
                    help="names to score; defaults to a few from the deploy list")
    ap.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    cfg = ap.parse_args()
    code, _ = run_preflight(tickers=cfg.tickers, equity=cfg.equity)
    return code


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below.
# Exit code 0 = safe to deploy. Exit code 1 = a number downstream is WRONG,
# not merely missing, so do not trade or quote results until it clears.
#
# Passing any command-line flag still works and takes over.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TICKERS = DEFAULT_TICKERS   # names to score, e.g. ["AMD", "NVDA", "MSTR"]
    RUN_EQUITY  = DEFAULT_EQUITY    # account size the sizing checks assume
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())            # e.g. python deployment_preflight.py --equity 25000
    else:
        _code, _results = run_preflight(tickers=RUN_TICKERS, equity=RUN_EQUITY)
        sys.exit(_code)
