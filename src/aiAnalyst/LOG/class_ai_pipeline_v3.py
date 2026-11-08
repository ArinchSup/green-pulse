#!/usr/bin/env python3
"""
class_ai_pipeline_v3.py - master synthesis with Pillar 2 as the RISK layer,
                          plus a PORTFOLIO layer above it.

WHAT CHANGED FROM v2
--------------------
v2 sized every name correctly and the book incorrectly. Six names at the 20%
concentration ceiling is 120% of equity; each risking 2% at its stop is 12% of
equity gone if they stop out together, and in a technology universe they largely
do. v2 printed a warning about this and traded anyway.

v3 adds class_ai_portfolio.py between sizing and output. Per-name sizing is
unchanged - every number Pillar 2 produces is still produced the same way - and
then the book as a whole is shrunk until it satisfies four limits: gross
exposure, total risk at stops, per-correlation-cluster exposure, and a book-level
drawdown probability. No name is refused for being risky - V51 compared refusing
against shrinking and refusing lost, so the layer only shrinks. A name can still
disappear, but only by scaling below a tradeable size, and it is reported as
dropped with the reason rather than silently omitted.

Two consequences worth stating in the write-up:

  * a screen now returns SMALLER positions than v2 did, and that is the fix, not
    a regression. v2's sizes were only correct for a one-position account.
  * --held matters. A screen that does not know what is already on measures an
    empty book and will authorise a second full allocation on top of the first.

WHY PILLAR 2 IS A RISK LAYER AT ALL

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
                  -> PORTFOLIO (class_ai_portfolio: correlation, budget, caps)

USAGE
  python class_ai_pipeline_v3.py --ticker NVDA --equity 25000
  python class_ai_pipeline_v3.py --screen NVDA AMD AVGO MU --equity 25000
  python class_ai_pipeline_v3.py --screen NVDA AMD --held AVGO:12 MU:8 --equity 25000
"""
import argparse
import json
import os
import sys

import class_ai_pillar2_risk_v2 as P2
import class_ai_portfolio as PF

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
                         target_date=None, verbose=True,
                         require_direction=None):
    """
    Returns a dict that ALWAYS carries "final_decision". The old version used
    "decision" on its early-exit path and "final_decision" on the main path, so
    any caller reading result["final_decision"] raised KeyError on every
    rejected ticker - which is most of them.
    """
    # A parameter rather than a global read, so a caller can isolate the risk
    # layer without mutating module state. None means "use the module default".
    if require_direction is None:
        require_direction = REQUIRE_DIRECTION

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
    if require_direction and not dirn["bullish"]:
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
# PORTFOLIO LAYER
# =============================================================================
def apply_portfolio(results, calib=None, equity=ACCOUNT_EQUITY, as_of=None,
                    held_specs=None, verbose=True):
    """
    Shrink the accepted names until the BOOK satisfies its limits.

    Mutates each EXECUTE result in place:

      sizing                    replaced with the portfolio-final numbers
      sizing_before_portfolio    the per-name intent, kept for the write-up
      portfolio_scale            what this name was multiplied by
      risk_contribution_pct      its share of book volatility

    `sizing` carries the FINAL numbers rather than the intent, and a name the
    portfolio layer zeroes is demoted to NO TRADE. Both choices point the same
    way: anything downstream that ignores the new fields still reads a tradeable
    size and never trades a dropped name. The reverse convention would have the
    pipeline hand a 20% position to a caller that had not been updated.

    Returns the book dict, or None when there is nothing to construct.
    """
    def say(*a):
        if verbose:
            print(*a)

    ex = [r for r in results if r.get("final_decision") == "EXECUTE"
          and r.get("sizing") and r.get("levels")]
    held, held_bad = PF.parse_holdings(held_specs, calib=calib, equity=equity,
                                       as_of=as_of)
    for b in held_bad:
        say(f"  holding {b['spec']}: {b['reason']}")

    if not ex and not held:
        return None
    if len(ex) < 2 and not held:
        # One name against an empty book cannot breach a portfolio limit: its
        # position is already under the per-name ceiling and its risk under the
        # per-trade budget. Running the layer would print a book report that
        # says nothing. Say why instead of silently skipping it.
        say("\n  portfolio layer skipped: one position against an empty book "
            "cannot breach a portfolio limit. Pass --held to count what is "
            "already on.")
        return None

    cands = [{"ticker": r["ticker"],
              "position_pct_of_equity": r["sizing"]["position_pct_of_equity"],
              "stop_pct": r["sizing"]["stop_pct"],
              "entry": r["levels"]["entry"],
              "forecast_vol_annual_pct": r["risk"]["forecast_vol_annual_pct"]}
             for r in ex]

    book = PF.build_book(cands, calib=calib, equity=equity, as_of=as_of,
                         held=held)
    by_ticker = {q["ticker"]: q for q in book["positions"] if q["proposed"]}
    dropped = {d["ticker"]: d["reason"] for d in book.get("dropped", [])}
    excluded = {e["ticker"]: e["reason"] for e in book.get("excluded", [])}

    for r in ex:
        t = r["ticker"]
        q = by_ticker.get(t)
        if t in excluded and q is None:
            r["final_decision"] = "NO TRADE"
            r["reason"] = f"portfolio layer: {excluded[t]}"
            continue
        if q is None:
            continue
        r["sizing_before_portfolio"] = dict(r["sizing"])
        r["portfolio_scale"] = q["scaled_by"]
        r["risk_contribution_pct"] = q["risk_contribution_pct"]
        if q["new_pct"] <= 0 or not q["shares_to_buy"]:
            r["final_decision"] = "NO TRADE"
            r["reason"] = ("portfolio layer: "
                           + dropped.get(t, "no room inside the portfolio "
                                            "limits once the rest of the book "
                                            "is counted"))
            r["sizing"] = None
            continue
        # `sizing` describes the ORDER TO PLACE, so it carries the new exposure
        # only. When this name is an addition to something already on, the
        # combined figures sit beside it rather than replacing it - a caller
        # placing an order needs the increment, and a caller reporting risk needs
        # the total, and conflating the two is how a 15% holding plus a 20% buy
        # gets recorded as a 20% position.
        new_risk = round(q["new_pct"] * q["stop_pct"] / 100.0, 3)
        s = dict(r["sizing"])
        s.update({
            "position_pct_of_equity": q["new_pct"],
            "effective_risk_pct": new_risk,
            "shares": q["shares_to_buy"],
            "position_value": round(equity * q["new_pct"] / 100.0, 2),
            "max_loss_at_stop": round(equity * new_risk / 100.0, 2),
            "portfolio_scaled": True,
            "already_held_pct": q["existing_pct"],
            "book_position_pct_total": q["position_pct"],
            "book_effective_risk_pct_total": q["effective_risk_pct"],
        })
        r["sizing"] = s
    return book


# =============================================================================
# CLI
# =============================================================================
def run_screen(tickers, equity=ACCOUNT_EQUITY, risk_budget=BASE_RISK_PER_TRADE,
               as_of=None, held=None, require_direction=None,
               use_portfolio=True, calib=None, verbose=False):
    """
    The whole pipeline for one or many tickers. THE function to call.

        out = run_screen(["NVDA", "AMD"], equity=25000)
        out["results"]     # one dict per ticker, always with final_decision
        out["portfolio"]   # the book, or None

        run_screen(["NVDA"], equity=25000, require_direction=False)
        run_screen(["NVDA", "AMD"], held=["AVGO:12"], equity=25000)
        run_screen([...], use_portfolio=False)   # what v2 did; do not trade it

    `verbose` prints the per-ticker workings. Use print_screen() afterwards for
    the summary table. Raises RuntimeError if the calibration is missing.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if not tickers:
        raise RuntimeError("run_screen needs at least one ticker.")
    if calib is None:
        if not os.path.exists(P2.CALIB_FILE):
            raise RuntimeError(f"{P2.CALIB_FILE} not found. Run: "
                               f"python {P2.__name__}.py --calibrate")
        # Load once. assess() would otherwise re-read and re-parse this for
        # every ticker in a screen.
        calib = P2.load_calibration()

    results = []
    for t in tickers:
        try:
            results.append(evaluate_trade_setup(
                t, equity=equity, risk_budget=risk_budget, calib=calib,
                target_date=as_of, verbose=verbose,
                require_direction=require_direction))
        except Exception as e:
            results.append(_no_trade(t, f"{type(e).__name__}: {e}"))

    book = None
    if use_portfolio:
        book = apply_portfolio(results, calib=calib, equity=equity, as_of=as_of,
                               held_specs=held, verbose=verbose)
    elif verbose:
        print("\n  PORTFOLIO LAYER DISABLED. These sizes are correct for a "
              "one-position account and too large for any other.")
    return {"results": results, "portfolio": book}


def print_screen(results, book=None, use_portfolio=True):
    """The summary table. Separated from run_screen so a caller can take the
    numbers without the printing, or print without re-running."""
    if len(results) == 1 and book is None:
        print("\n" + json.dumps(results[0], indent=2, default=str))
        return
    print(f"\n{'=' * 60}\n  SCREEN\n{'=' * 60}")
    ex = [r for r in results if r["final_decision"] == "EXECUTE"]
    ex.sort(key=lambda r: r["telemetry"].get("prob_drawdown") or 1.0)
    print(f"  {'ticker':<8}{'decision':<11}{'P(dd)':>8}{'tier':>11}"
          f"{'pos%':>7}{'was':>7}{'eff risk%':>11}")
    for r in ex:
        t = r["telemetry"]
        p = t.get("prob_drawdown")
        was = (r.get("sizing_before_portfolio") or {}).get(
            "position_pct_of_equity")
        print(f"  {r['ticker']:<8}{'EXECUTE':<11}"
              f"{('n/a' if p is None else f'{p:.1%}'):>8}"
              f"{t.get('risk_tier', ''):>11}"
              f"{r['sizing']['position_pct_of_equity']:>7.1f}"
              f"{('-' if was is None else f'{was:.1f}'):>7}"
              f"{r['sizing']['effective_risk_pct']:>11.2f}")
    for r in results:
        if r["final_decision"] != "EXECUTE":
            print(f"  {r['ticker']:<8}{'NO TRADE':<11}  {r['reason']}")
    tot = sum(r["sizing"]["effective_risk_pct"] for r in ex)
    cap = sum(r["sizing"]["position_pct_of_equity"] for r in ex)
    if ex:
        print(f"\n  {len(ex)} new positions | {cap:.1f}% of equity deployed "
              f"| {tot:.2f}% risk at stops")
    if book:
        PF.describe(book)
    elif not use_portfolio and cap > 100:
        print("\n  WARNING: allocations exceed equity. This is what v2 did. "
              "Drop --no-portfolio to size the book properly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--screen", nargs="*", help="rank several tickers at once")
    ap.add_argument("--held", nargs="*", metavar="TICKER:PCT",
                    help="positions already on, as a percent of equity "
                         "(NVDA:12 AVGO:8). They consume the portfolio limits "
                         "and are never resized. Omitting them measures an "
                         "empty book and oversizes everything.")
    ap.add_argument("--equity", type=float, default=ACCOUNT_EQUITY)
    ap.add_argument("--risk", type=float, default=BASE_RISK_PER_TRADE)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-direction", action="store_true",
                    help="skip the news gate and run the risk layer alone")
    ap.add_argument("--no-portfolio", action="store_true",
                    help="per-name sizing only, as v2 behaved. Use it to show "
                         "what the portfolio layer changes, not to trade.")
    ap.add_argument("--json", action="store_true", help="output only JSON")
    cfg = ap.parse_args()

    # main() parses arguments and turns exceptions into exit codes. All the
    # behaviour lives in run_screen() / print_screen(), which can be called
    # directly without going through argparse or mutating module globals.
    tickers = cfg.screen or ([cfg.ticker] if cfg.ticker else [])
    if not tickers:
        sys.exit("Pass --ticker or --screen.")
    try:
        out = run_screen(tickers, equity=cfg.equity, risk_budget=cfg.risk,
                         as_of=cfg.date, held=cfg.held,
                         require_direction=False if cfg.no_direction else None,
                         use_portfolio=not cfg.no_portfolio,
                         verbose=not cfg.json)
    except RuntimeError as e:
        sys.exit(str(e))

    if cfg.json:
        print(json.dumps(out, indent=2, default=str))
        return
    print_screen(out["results"], out["portfolio"],
                 use_portfolio=not cfg.no_portfolio)


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. This is the whole pipeline:
# risk model -> per-name sizing -> portfolio limits. Edit the values, run.
#
# Passing any command-line flag still works and takes over.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TICKERS = ["NVDA", "AMD", "MSTR"]   # one name or many; a list either way
    RUN_HELD    = []          # already on, as % of equity: ["AVGO:12", "MU:8"].
                              # Leaving this empty measures an EMPTY book and
                              # therefore oversizes everything.
    RUN_EQUITY  = ACCOUNT_EQUITY
    RUN_RISK    = BASE_RISK_PER_TRADE
    RUN_DATE    = None        # "2026-06-30" screens as of a past date

    RUN_REQUIRE_DIRECTION = False  # False skips the news gate and runs the risk
                                   # layer alone - what you want while pillars 1
                                   # and 3 are not wired in. Set it to None to
                                   # use REQUIRE_DIRECTION above instead, and
                                   # every name will come back NO TRADE until
                                   # real catalyst scores are feeding in.
    RUN_USE_PORTFOLIO     = True   # False = per-name sizing only, as v2 behaved.
                                   # Useful to show what the layer changes.
                                   # Do not trade the False version.
    RUN_VERBOSE = False       # True prints the per-ticker workings as it goes
    RUN_AS_JSON = False       # True prints the raw payload instead of the table
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                     # e.g. python class_ai_pipeline_v3.py --json

    else:
        try:
            _out = run_screen(RUN_TICKERS, equity=RUN_EQUITY,
                              risk_budget=RUN_RISK, as_of=RUN_DATE,
                              held=RUN_HELD,
                              require_direction=RUN_REQUIRE_DIRECTION,
                              use_portfolio=RUN_USE_PORTFOLIO,
                              verbose=RUN_VERBOSE and not RUN_AS_JSON)
        except RuntimeError as _e:
            sys.exit(str(_e))

        if RUN_AS_JSON:
            print(json.dumps(_out, indent=2, default=str))
        else:
            print_screen(_out["results"], _out["portfolio"],
                         use_portfolio=RUN_USE_PORTFOLIO)
