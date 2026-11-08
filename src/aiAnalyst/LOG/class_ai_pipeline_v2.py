#!/usr/bin/env python3
"""
class_ai_pipeline_v2.py - master synthesis with Pillar 2 as the RISK layer.

WHAT CHANGED AND WHY

The old pipeline made Pillar 2 the lead voice: it decided direction
("technical_sentiment": "Bullish"), supplied a confidence, and News and
Fundamentals could only veto it or boost its size. That architecture rested on
Pillar 2 being able to predict direction. It cannot.

Out-of-sample information coefficient was indistinguishable from zero across two
label formulations (barrier-touch and forward excess return), five horizons
(5/10/20/60 bars and 180 days), three neutralisations (raw, date-demeaned,
sector-residual) and two universes (62-name tech, 301-name broad). Nine model
versions. What survived every control was magnitude, not direction:

    forward_vol = 7.35 + 0.162 yz5 + 0.146 yz22 + 0.448 yz66    (V53/V59)

So Pillar 2 stops voting on direction and starts doing the job it can actually
do, which is three jobs:

    RISK VETO   refuse trades carrying a measured dilution or solvency flag.
                The drawdown-probability veto is OFF by default - V51 showed
                refusing those names cost more in return than it saved in
                drawdown. They are shrunk instead.
    SIZING      convert a risk budget into a position size, capped for
                concentration, using the forecast stop distance
    LEVELS      entry / stop / target from VOL_SCALED geometry, so the reward
                geometry is constant in volatility units across names

DIRECTION NOW COMES FROM ELSEWHERE. This is the load-bearing consequence and it
should be stated plainly in the write-up: with Pillar 2 demoted, the only
directional signal in the system is Pillar 1 (news catalyst) plus the
fundamental engine. Neither has been through the walk-forward testing Pillar 2
was put through. A calibrated risk layer does not create edge; it limits ruin.
If the news pillar has no directional edge either, this system is a long-only
random entry with good risk control - which is still a legitimate thesis result,
but it is a different claim from "the model picks winners".

    GATE ORDER    direction (Pillar 1 + Fundamentals)
                  -> risk veto (Pillar 2)
                  -> size (Pillar 2, scaled by conviction and risk tier)
                  -> levels (Pillar 2)

USAGE
  python class_ai_pipeline_v2.py --ticker NVDA --equity 25000
  python class_ai_pipeline_v2.py --screen NVDA AMD AVGO MU --equity 25000
"""
import argparse
import json
import os
import sys

import class_ai_pillar2_risk_v2 as P2

# Pillar 1 and the fundamental engine are optional at import time so this file
# can be exercised on its own while your teammates' modules are still moving.
try:
    from class_ai_pillar1_news import get_news_catalyst_score
except ImportError:
    get_news_catalyst_score = None


# =============================================================================
# CONFIGURATION
# =============================================================================
ACCOUNT_EQUITY = 10000.0
BASE_RISK_PER_TRADE = 0.02          # 2% of equity at risk if the stop fills
MAX_POSITION_PCT = 20.0             # concentration ceiling, one name

# --- direction gate -----------------------------------------------------------
# Pillar 2 supplies no direction, so something must. Until the news pillar is
# validated, REQUIRE_DIRECTION=False lets you run the risk layer standalone
# (useful for the thesis: it isolates the risk contribution from the entry rule).
REQUIRE_DIRECTION = True
NEWS_BULLISH_MIN = 6                # catalyst_score needed to call it bullish
FUND_HEALTH_MIN = 4                 # below this, the balance sheet vetoes

# --- risk gate ----------------------------------------------------------------
# Thresholds on the CALIBRATED probability, whose question is set by
# class_ai_pillar2_risk_v2 (currently a 20% fall within 90 days - V57 and V59
# showed the old 30%/180d setting was the worst of the usable band, and the only
# one whose Brier skill never cleared zero).
#
# These are ABSOLUTE probabilities but they were chosen as MULTIPLES of the
# calibration's base rate - roughly 0.9x, 1.7x and 2.3x - so that "low" means at
# or below average risk and "excessive" means genuinely unusual. When the
# question changes the base rate moves and these have to follow it. They did not,
# once, and the result was twice as many names silently cut to the smallest
# position size.
#
#   base rate these were set against: 13.2%  (20% fall in 90 days)
#   0.12 = 0.91x   0.22 = 1.67x   0.31 = 2.35x
#
# deployment_preflight.py checks this against the live calibration. Rerun it
# after every recalibration.
#
# THE HARD VETO IS OFF, AND THAT IS AN EVIDENCE-BASED CHANGE.
# V51 compared refusing trades above the ceiling against merely shrinking them.
# Refusing LOST: Calmar was worse by 0.020, 95% CI +0.004 to +0.035 excluding
# zero. The veto bought 0.64pp of drawdown for 0.44pp of CAGR and dropped 8% of
# trades to do it. Set MAX_DRAWDOWN_PROB to a number to turn it back on; leaving
# it None shrinks the worst names to EXCESSIVE_FLOOR of the ceiling instead,
# which is what the evidence supports.
MAX_DRAWDOWN_PROB = None
EXCESSIVE_FLOOR = 0.25              # ceiling multiplier above the top tier
RISK_TIERS = [(0.12, 1.00, "low"),
              (0.22, 0.75, "moderate"),
              (0.31, 0.50, "elevated")]  # (upper bound, ceiling multiplier, label)


# Flags that stop the trade outright. Both are EDGAR-measured with real base
# rates behind them: the top share-growth quintile saw a 30% drawdown 41.6% of
# the time within 6 months, against a 25% rate for the rest of that sample.
HARD_FLAGS = {"SHORT_RUNWAY", "HIGH_DILUTION"}
# Flags that shrink the position instead of blocking it.
SOFT_FLAGS = {"MODERATE_DILUTION", "FREQUENT_OFFERINGS", "LOW_LIQUIDITY",
              "VOL_EXPANDING"}
SOFT_FLAG_MULT = 0.80               # compounding, per soft flag
SOFT_FLAG_FLOOR = 0.50              # never shrink below half on flags alone

# --- conviction boost ---------------------------------------------------------
# Applied to the risk BUDGET, not to the finished position, so it can never push
# a position past the ceiling. For low-volatility names the ceiling is already
# binding and the boost does nothing; the pipeline says so rather than reporting
# a multiplier that moved no shares.
CATALYST_BOOST = 1.25
CATALYST_BOOST_MIN = 8              # catalyst_score needed to earn it

# The key Pillar 2 uses for the drawdown probability is built from its own
# constants, so changing DRAWDOWN or HORIZON_DAYS there does not silently break
# the lookup here.
PROB_KEY = f"prob_drawdown_{int(P2.DRAWDOWN * 100)}pct_{P2.HORIZON_DAYS}d"


# =============================================================================
# PILLAR 1.5: FUNDAMENTAL ENGINE (still a stub on your teammate's side)
# =============================================================================
def get_mock_fundamental_score(ticker):
    return {
        "ticker": ticker,
        "financial_health_score": 7,
        "solvency_status": "Healthy",
        "cash_runway_months": 24,
        "is_mock": True,
    }


# =============================================================================
# DIRECTION
# =============================================================================
def direction_view(ticker, news_days=3):
    """
    The only directional opinion in the system. Returns a dict even when the
    news pillar is unavailable, so the caller never has to None-check.
    """
    if get_news_catalyst_score is None:
        return {"available": False, "sentiment": "Unknown", "score": 5,
                "bullish": False,
                "note": "class_ai_pillar1_news not importable"}
    try:
        r = get_news_catalyst_score(ticker, days_back=news_days) or {}
    except Exception as e:
        return {"available": False, "sentiment": "Error", "score": 5,
                "bullish": False, "note": f"{type(e).__name__}: {e}"}
    sent = r.get("catalyst_sentiment", "Neutral")
    score = r.get("catalyst_score", 5)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 5
    return {"available": True, "sentiment": sent, "score": score,
            "bullish": sent == "Bullish" and score >= NEWS_BULLISH_MIN,
            "bearish_veto": sent == "Bearish" and score >= 7}


# =============================================================================
# RISK GATE
# =============================================================================
def risk_tier(p):
    """Map a calibrated drawdown probability onto a ceiling multiplier."""
    if p is None:
        # No probability means no calibrated basis for sizing. Do not default to
        # full size on missing data - that is the failure mode that turns a risk
        # model into decoration.
        return 0.50, "unknown"
    for bound, mult, label in RISK_TIERS:
        if p < bound:
            return mult, label
    return EXCESSIVE_FLOOR, "excessive"


def flag_review(flags, fund):
    """
    Split Pillar 2's flags into vetoes and size penalties.

    Pillar 2 derives SHORT_RUNWAY from EDGAR cash and operating cash flow. The
    fundamental engine measures the same thing from its own data. Counting both
    would penalise one balance sheet twice, so when the fundamental engine is
    live its runway number wins and Pillar 2's is demoted to corroboration.
    """
    fund_live = not fund.get("is_mock", True)
    vetoes, penalties, demoted = [], [], []
    mult = 1.0
    for f in flags or []:
        code = f.get("code")
        if code == "SHORT_RUNWAY" and fund_live:
            demoted.append(f)
            continue
        if code in HARD_FLAGS:
            vetoes.append(f)
        elif code in SOFT_FLAGS:
            penalties.append(f)
            mult *= SOFT_FLAG_MULT
    mult = max(mult, SOFT_FLAG_FLOOR) if penalties else 1.0
    return vetoes, penalties, demoted, mult


# =============================================================================
# SIZING
# =============================================================================
def resize(equity, entry, stop_pct, risk_budget, budget_mult, cap_mult,
           cap_pct=MAX_POSITION_PCT):
    """
    Sizing with TWO separate controls, because a single multiplier plus a flat
    notional cap produces the wrong answer.

    The problem with the flat cap: a 2% risk budget behind a 4.8% stop implies
    42% of equity, so the cap binds and the position lands at 20%, risking
    0.96%. Behind a 12.6% stop the same budget implies 15.9%, the cap does not
    bind, and the position risks the full 2%. The cap therefore INVERTS the risk
    budget - the calmest name ends up with the least risk on it and the most
    volatile name with the most. Every name under roughly 45% annual volatility
    sizes to exactly 20%, so the calibrated volatility forecast never reaches
    the sizing decision at all.

    The split:

      budget_mult  scales the RISK BUDGET. Conviction lives here. It can be
                   absorbed by the cap, and that is intended - a news score is
                   not grounds to breach a concentration limit.
      cap_mult     scales the CEILING. Risk tier and flags live here, so they
                   always bite, capped or not.

    effective_risk_pct is the number to report and to sum across open
    positions. The nominal budget stops being true the moment the cap engages.
    """
    if (not stop_pct or stop_pct <= 0 or entry <= 0
            or budget_mult <= 0 or cap_mult <= 0):
        return None
    budget = risk_budget * budget_mult
    cap = cap_pct * cap_mult
    raw = budget / (stop_pct / 100.0) * 100.0
    pos = min(raw, cap)
    eff_risk = pos * stop_pct / 100.0
    shares = int(equity * pos / 100.0 / entry)
    # Did the conviction boost actually change the position, or did the ceiling
    # eat it? Reporting a 1.25x multiplier that moved nothing is misleading.
    base_raw = risk_budget / (stop_pct / 100.0) * 100.0
    absorbed = bool(budget_mult != 1.0 and min(base_raw, cap) == pos)
    return {
        "nominal_risk_pct": round(budget * 100, 3),
        "budget_multiplier": round(budget_mult, 3),
        "cap_multiplier": round(cap_mult, 3),
        "stop_pct": round(stop_pct, 2),
        "uncapped_position_pct": round(raw, 1),
        "position_pct_of_equity": round(pos, 1),
        "position_cap_applied": raw > cap,
        "position_cap_pct": round(cap, 1),
        "conviction_absorbed_by_cap": absorbed,
        "effective_risk_pct": round(eff_risk, 3),
        "position_value": round(equity * pos / 100.0, 2),
        "shares": shares,
        "max_loss_at_stop": round(shares * entry * stop_pct / 100.0, 2),
    }


# =============================================================================
# MASTER SYNTHESIS
# =============================================================================
def _no_trade(ticker, reason, **extra):
    out = {"ticker": ticker, "final_decision": "NO TRADE", "reason": reason,
           "sizing": None, "levels": None}
    out.update(extra)
    return out


def evaluate_trade_setup(ticker, equity=ACCOUNT_EQUITY,
                         risk_budget=BASE_RISK_PER_TRADE, calib=None,
                         target_date=None, verbose=True):
    """
    Returns a dict that ALWAYS carries "final_decision". The old version used
    "decision" on its early-exit path and "final_decision" on the main path, so
    any caller reading result["final_decision"] raised KeyError on every
    rejected ticker - which is most of them.
    """
    def say(*a):
        if verbose:
            print(*a)

    say(f"\n{'=' * 60}\n  {ticker}\n{'=' * 60}")

    # --- 1. direction ---------------------------------------------------------
    say("\n[1] Direction  (Pillar 1 news + fundamentals)")
    dirn = direction_view(ticker)
    fund = get_mock_fundamental_score(ticker)
    fund_score = fund.get("financial_health_score", 5)
    say(f"    news: {dirn['sentiment']} ({dirn['score']}/10)"
        f"{'' if dirn['available'] else '  [' + dirn.get('note', '') + ']'}")
    say(f"    fundamental health: {fund_score}/10"
        f"{'  [mock]' if fund.get('is_mock') else ''}")

    telemetry = {"news_sentiment": dirn["sentiment"], "news_score": dirn["score"],
                 "news_available": dirn["available"], "fund_score": fund_score,
                 "fund_is_mock": fund.get("is_mock", True)}

    if fund_score <= FUND_HEALTH_MIN:
        return _no_trade(ticker, f"fundamental health {fund_score}/10 at or below "
                                 f"the {FUND_HEALTH_MIN} floor", telemetry=telemetry)
    if dirn.get("bearish_veto"):
        return _no_trade(ticker, f"bearish catalyst ({dirn['score']}/10)",
                         telemetry=telemetry)
    if REQUIRE_DIRECTION and not dirn["bullish"]:
        return _no_trade(ticker, f"no bullish catalyst (news {dirn['sentiment']} "
                                 f"{dirn['score']}/10)", telemetry=telemetry)

    # --- 2. risk --------------------------------------------------------------
    say("\n[2] Risk  (Pillar 2)")
    r = P2.assess(ticker, target_date=target_date, equity=equity,
                  risk_budget=risk_budget, calib=calib,
                  max_position_pct=MAX_POSITION_PCT)
    if r is None:
        # assess() returns None for a missing cache file and for a file with
        # too few bars. Say both rather than asserting the wrong one - on a
        # screen these look identical and mean different fixes.
        return _no_trade(ticker, "Pillar 2 returned nothing: no cached price "
                                 f"file, or fewer than {P2.MIN_BARS} bars",
                         telemetry=telemetry)

    p_dd = r["risk"].get(PROB_KEY)
    tier_mult, tier = risk_tier(p_dd)
    say(f"    forecast vol {r['risk']['forecast_vol_annual_pct']}% annual "
        f"(realised 60d {r['risk']['realised_vol_60d_pct']}%)")
    # Built from the module's constants, never written out. This line said
    # "P(30% drawdown in 180d)" while the model computed a 20% fall in 90 days,
    # so the output described a different question from the one being answered.
    say(f"    P({P2.DRAWDOWN:.0%} drawdown in {P2.HORIZON_DAYS}d) = "
        f"{'n/a' if p_dd is None else f'{p_dd:.1%}'}  -> {tier} risk"
        f"   [vol {r['risk'].get('vol_model', 'n/a')}]")

    telemetry.update({"prob_drawdown": p_dd, "risk_tier": tier,
                      "forecast_vol_pct": r["risk"]["forecast_vol_annual_pct"]})

    if (MAX_DRAWDOWN_PROB is not None and p_dd is not None
            and p_dd >= MAX_DRAWDOWN_PROB):
        return _no_trade(ticker, f"drawdown probability {p_dd:.1%} at or above "
                                 f"the {MAX_DRAWDOWN_PROB:.0%} ceiling",
                         telemetry=telemetry, risk=r["risk"], flags=r["flags"])

    vetoes, penalties, demoted, flag_mult = flag_review(r["flags"], fund)
    for f in (r["flags"] or []):
        mark = ("VETO" if f in vetoes else "size" if f in penalties
                else "note" if f in demoted else "info")
        say(f"    [{mark}] {f['code']}: {f['detail']}")
    if vetoes:
        return _no_trade(ticker,
                         "; ".join(f"{f['code']} - {f['detail']}" for f in vetoes),
                         telemetry=telemetry, risk=r["risk"], flags=r["flags"])

    # --- 3. size --------------------------------------------------------------
    say("\n[3] Size")
    boost = (CATALYST_BOOST if dirn["available"] and dirn["sentiment"] == "Bullish"
             and dirn["score"] >= CATALYST_BOOST_MIN else 1.0)
    cap_mult = tier_mult * flag_mult
    sizing = resize(equity, r["levels"]["entry"], r["levels"]["stop_pct"],
                    risk_budget, boost, cap_mult, MAX_POSITION_PCT)
    if sizing is None or sizing["shares"] < 1:
        return _no_trade(ticker, "position rounds to zero shares at this equity",
                         telemetry=telemetry, risk=r["risk"], flags=r["flags"])

    say(f"    budget {risk_budget * 100:.1f}% x catalyst {boost:.2f} "
        f"-> {sizing['nominal_risk_pct']}%")
    say(f"    ceiling {MAX_POSITION_PCT:.0f}% x tier {tier_mult:.2f} "
        f"x flags {flag_mult:.2f} -> {sizing['position_cap_pct']}%")
    say(f"    {sizing['shares']} shares  ${sizing['position_value']:,.0f}  "
        f"({sizing['position_pct_of_equity']}% of equity)  "
        f"risking {sizing['effective_risk_pct']}%")
    if sizing["position_cap_applied"]:
        say(f"    ceiling bound: {sizing['uncapped_position_pct']}% -> "
            f"{sizing['position_pct_of_equity']}%; effective risk "
            f"{sizing['effective_risk_pct']}% not "
            f"{sizing['nominal_risk_pct']}%")
    if sizing["conviction_absorbed_by_cap"]:
        say(f"    note: the {boost:.2f}x catalyst boost changed nothing - the "
            f"ceiling was already binding")

    return {
        "ticker": ticker,
        "as_of": r["as_of"],
        "price": r["price"],
        "final_decision": "EXECUTE",
        "reason": ("bullish catalyst, risk within tolerance"
                   if dirn["bullish"] else "risk within tolerance "
                                           "(direction gate disabled)"),
        "risk": r["risk"],
        "sizing": sizing,
        "levels": r["levels"],
        "flags": r["flags"],
        "context": r["context"],
        "telemetry": telemetry,
        "return_forecast": None,
        "disclaimer": r["disclaimer"],
    }


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--screen", nargs="*", help="rank several tickers at once")
    ap.add_argument("--equity", type=float, default=ACCOUNT_EQUITY)
    ap.add_argument("--risk", type=float, default=BASE_RISK_PER_TRADE)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-direction", action="store_true",
                    help="skip the news gate and run the risk layer alone")
    ap.add_argument("--json", action="store_true", help="output only JSON")
    cfg = ap.parse_args()

    if cfg.no_direction:
        globals()["REQUIRE_DIRECTION"] = False

    if not os.path.exists(P2.CALIB_FILE):
        sys.exit(f"{P2.CALIB_FILE} not found. Run:\n"
                 f"  python {P2.__name__}.py --calibrate")
    # Load once. assess() would otherwise re-read and re-parse this for every
    # ticker in a screen.
    calib = P2.load_calibration()

    tickers = cfg.screen or ([cfg.ticker] if cfg.ticker else [])
    if not tickers:
        sys.exit("Pass --ticker or --screen.")

    results = []
    for t in tickers:
        try:
            results.append(evaluate_trade_setup(
                t, equity=cfg.equity, risk_budget=cfg.risk, calib=calib,
                target_date=cfg.date, verbose=not cfg.json))
        except Exception as e:
            results.append(_no_trade(t, f"{type(e).__name__}: {e}"))

    if cfg.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        return

    if len(results) > 1:
        print(f"\n{'=' * 60}\n  SCREEN\n{'=' * 60}")
        ex = [r for r in results if r["final_decision"] == "EXECUTE"]
        ex.sort(key=lambda r: r["telemetry"].get("prob_drawdown") or 1.0)
        print(f"  {'ticker':<8}{'decision':<11}{'P(dd)':>8}{'tier':>11}"
              f"{'pos%':>7}{'eff risk%':>11}")
        for r in ex:
            t = r["telemetry"]
            p = t.get("prob_drawdown")
            print(f"  {r['ticker']:<8}{'EXECUTE':<11}"
                  f"{('n/a' if p is None else f'{p:.1%}'):>8}"
                  f"{t.get('risk_tier', ''):>11}"
                  f"{r['sizing']['position_pct_of_equity']:>7.1f}"
                  f"{r['sizing']['effective_risk_pct']:>11.2f}")
        for r in results:
            if r["final_decision"] != "EXECUTE":
                print(f"  {r['ticker']:<8}{'NO TRADE':<11}  {r['reason']}")
        if ex:
            tot = sum(r["sizing"]["effective_risk_pct"] for r in ex)
            cap = sum(r["sizing"]["position_pct_of_equity"] for r in ex)
            print(f"\n  {len(ex)} positions | {cap:.0f}% of equity deployed | "
                  f"{tot:.2f}% total risk at stops")
            if cap > 100:
                print("  WARNING: allocations exceed equity. The per-name cap "
                      "does not constrain the portfolio total - add a portfolio "
                      "budget before trading a screen this wide.")
    else:
        print("\n" + json.dumps(results[0], indent=2))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values, run.
#
# This is the OLD pipeline, kept only to show what the portfolio layer changed.
# It sizes each name on its own and never looks at the book, so a wide screen
# can allocate more than 100% of equity. Use class_ai_pipeline_v3.py to trade.
#
# v2 never got a run_* function - all of its behaviour lives inside main() -
# so the block below hands main() the arguments the CLI would have passed.
# Passing any command-line flag still works and takes over.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TICKERS      = ["NVDA", "AMD", "MSTR"]
    RUN_EQUITY       = ACCOUNT_EQUITY
    RUN_RISK         = BASE_RISK_PER_TRADE
    RUN_DATE         = None     # "2026-06-30" screens as of a past date
    RUN_NO_DIRECTION = True     # True skips the news gate and runs risk alone
    RUN_AS_JSON      = False
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                  # e.g. python class_ai_pipeline_v2.py --ticker NVDA

    else:
        _argv = [sys.argv[0], "--screen", *RUN_TICKERS,
                 "--equity", str(RUN_EQUITY), "--risk", str(RUN_RISK)]
        if RUN_DATE:
            _argv += ["--date", RUN_DATE]
        if RUN_NO_DIRECTION:
            _argv.append("--no-direction")
        if RUN_AS_JSON:
            _argv.append("--json")
        sys.argv = _argv
        main()
