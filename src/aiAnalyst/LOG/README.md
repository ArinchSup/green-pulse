# Pillar 2 — Risk Model

**What it answers:** for one stock on one date, the probability of a fall of
**20% or more within 90 days**, plus the entry/stop/target levels and position
size that follow from it.

**What it does not answer:** direction. Nine model versions (V39–V50) tested
that and none survived out of sample. Pillar 2 forecasts *magnitude*. Direction,
if it comes from anywhere, comes from Pillar 1 (news) and the fundamental engine.

---

## Runtime flow

```
                    price_cache_v43/         edgar_cache/
                    (daily OHLCV, .pkl)      (SEC filings JSON)
                            |                      |
                            v                      v
   ┌──────────────────────────────────────────────────────────┐
   │  class_ai_pillar2_risk_v2.py          THE RISK MODEL      │
   │                                                           │
   │  yang_zhang_series()  -> yz5, yz22, yz66                  │
   │  forecast_vol()       -> annualised volatility forecast   │
   │  prob_drawdown()      -> P(20% fall in 90d), off the curve│
   │  edgar_flags()        -> dilution / runway / liquidity    │
   │  compute_levels()     -> entry, stop, target  (trade_config)
   │                                                           │
   │  assess(ticker)  ->  {risk, sizing, levels, flags, ...}   │
   └──────────────────────────────────────────────────────────┘
                            |         ^
                            |         |  pillar2_risk_calibration.json
                            |         |  (shrinkage coefficients + 12-bin curve)
                            v
   ┌──────────────────────────────────────────────────────────┐
   │  class_ai_pipeline_v3.py              THE SYNTHESIS LAYER │
   │                                                           │
   │  1. direction_view()   news + fundamentals  -> gate       │
   │  2. risk gate          hard flags veto; prob -> tier      │
   │  3. resize()           risk budget x conviction,          │
   │                        ceiling x tier x flags             │
   │  4. apply_portfolio()  hands the book to the layer below  │
   └──────────────────────────────────────────────────────────┘
                            |
                            v
   ┌──────────────────────────────────────────────────────────┐
   │  class_ai_portfolio.py                THE PORTFOLIO LAYER │
   │                                                           │
   │  correlation_matrix()  252d log returns, shrunk, PSD      │
   │  clusters()            single-linkage at rho >= 0.70      │
   │  build_book()          shrink until 4 limits hold:        │
   │      gross <= 100%  ·  risk at stops <= 6%                │
   │      cluster <= 40% ·  book P(fall) <= base rate          │
   └──────────────────────────────────────────────────────────┘
                            |
                            v
                  final positions + shares
```

Nothing is vetoed on risk level — only shrunk. V51 tested refusing against
shrinking and refusing lost.

---

## File map

| File | Role | Status |
|---|---|---|
| `class_ai_pillar2_risk_v2.py` | The risk model. Volatility forecast → calibrated drawdown probability → levels → per-name size. | **Deployed** |
| `pillar2_risk_calibration.json` | Fitted shrinkage coefficients + the 12-bin curve. Produced by `--calibrate`. | **Deployed** |
| `class_ai_portfolio.py` | Book-level limits: gross, risk at stops, correlation clusters, book drawdown. | **Deployed** |
| `class_ai_pipeline_v3.py` | Synthesis: direction gate → risk gate → size → portfolio. | **Deployed** |
| `trade_config.py` | Trade geometry. `HORIZON_CONFIGS`, `compute_levels()`, `walk_trade()`. | **Deployed** |
| `deployment_preflight.py` | 25 checks that the constants across all files still describe the same question. | **Run before use** |
| `class_ai_pipeline_v2.py` | Previous synthesis layer, no portfolio limits. | Superseded — keep for reference |
| `class_ai_pillar2_risk.py` | Original module (30%/180d, single volatility feature). | Keep — V52's published numbers depend on it |
| `class_ai_pillar1_news.py` | Teammate's news pillar. Imported optionally; absent is handled. | External |

---

## The deployed modules in detail

### `class_ai_pillar2_risk_v2.py`

The model. Everything else consumes its output.

**Main entry point**

```python
assess(ticker, target_date=None, equity=10000.0, risk_budget=0.02,
       calib=None, price_cache=PRICE_CACHE, edgar_cache=EDGAR_CACHE,
       max_position_pct=MAX_POSITION_PCT)
```

Returns a dict with `risk`, `sizing`, `levels`, `flags`, `context`, and a
`disclaimer` recording that it makes no directional claim. Returns `None` when
there is not enough history or the volatility forecast is not finite — it fails
loudly rather than returning a plausible-looking default.

**The chain inside it**

| Function | Does |
|---|---|
| `load_prices()` | Reads one ticker's pickle from `price_cache_v43/` |
| `yang_zhang_series(df, w)` | Volatility from the whole bar — overnight gap, open-to-close, Rogers-Satchell intraday |
| `vol_feature_panel()` / `vol_features_at()` | The three features `yz5`, `yz22`, `yz66` |
| `forecast_vol(feats, calib)` | `7.35 + 0.162·yz5 + 0.146·yz22 + 0.448·yz66` |
| `prob_drawdown(feats, calib)` | Interpolates the 12-bin curve; clamps outside the fitted range |
| `edgar_flags(ticker, as_of)` | Dilution, runway, offerings, from cached SEC filings |
| `build_observations()` | The training table — only used by `--calibrate` and the evaluators |
| `fit_vol_shrinkage()` / `fit_drawdown_curve()` | Fitting, used by `--calibrate` |
| `check_horizons()` | Warns if the risk window is shorter than the trade it covers |

**Constants that matter**

```python
HORIZON_DAYS  = 90        # the question's window
DRAWDOWN      = 0.20      # what counts as an event
VOL_FEATURES  = ["yz5", "yz22", "yz66"]
MIN_BARS      = 260       # refuse to score a name with less history
MAX_POSITION_PCT = 20.0   # single-name concentration ceiling
```

`_fwd_bars()` and `_min_fwd_bars()` are **derived** from `HORIZON_DAYS`, never
typed in. A hard-coded 80-bar floor was correct at 180 days and silently
rejected every observation at 30.

### `class_ai_pipeline_v3.py`

Turns a risk assessment into a decision.

```python
evaluate_trade_setup(ticker, equity, risk_budget, calib, target_date, verbose)
apply_portfolio(results, calib, equity, as_of, held_specs, verbose)
```

**The gate order.** Direction (news + fundamentals) → risk flags → probability
tier → size → portfolio. A name failing any gate returns a dict that always
carries `final_decision`, so callers never hit a `KeyError` on a rejection.

**Sizing uses two independent multipliers**, and this matters:

```python
resize(equity, entry, stop_pct, risk_budget, budget_mult, cap_mult, cap_pct)
#   budget_mult  scales the RISK BUDGET   — conviction lives here,
#                                            absorbable by the ceiling
#   cap_mult     scales the CEILING       — risk tier and flags live here,
#                                            always bites
```

A single multiplier plus a flat cap inverts the risk budget: every name under
about 45% volatility sizes to exactly the cap, so the calmest name ends up
carrying the least risk. Keep them separate.

**Tier boundaries are multiples of the base rate**, not fixed numbers:

```python
RISK_TIERS = [(0.12, 1.00, "low"),        # 0.91x the 13.2% base rate
              (0.22, 0.75, "moderate"),   # 1.67x
              (0.31, 0.50, "elevated")]   # 2.35x
EXCESSIVE_FLOOR  = 0.25                   # above the top tier
MAX_DRAWDOWN_PROB = None                  # veto OFF — V51 evidence
```

Recalibrate and the base rate moves; the tiers must move with it.

### `class_ai_portfolio.py`

Per-name sizing is correct and the book is not: six names at the 20% ceiling is
120% of equity, and in a technology universe they do not fall independently.

```python
build_book(candidates, calib, equity, as_of, price_cache, held, ...)
assess_candidates(tickers, ...)   # tickers -> candidate dicts via P2.assess
parse_holdings(["NVDA:12", "AVGO:8"], ...)   # existing positions
describe(book)                    # the printed report
```

`candidates` only needs `ticker`, `position_pct_of_equity`, `stop_pct`, `entry`,
`forecast_vol_annual_pct` — so the layer does not depend on the pipeline's
result shape.

**The four limits**

```python
MAX_GROSS_PCT          = 100.0   # no implicit leverage
MAX_PORTFOLIO_RISK_PCT = 6.0     # CONVENTION, not a measurement
CLUSTER_CORR           = 0.70    # single-linkage threshold
MAX_CLUSTER_PCT        = 2.0 * P2.MAX_POSITION_PCT
MAX_PORTFOLIO_DD_PROB  = None    # None = pin to the calibration's base rate
MIN_POSITION_PCT       = 1.0     # below this, drop rather than trade dust
```

**Holdings consume the limits and are never resized.** A screen run without
`--held` measures an empty book and will authorise a second full allocation on
top of one already on.

### `deployment_preflight.py`

Run it after every recalibration and before quoting any number.

```
python deployment_preflight.py --tickers NVDA AMD MU --equity 25000
```

25 checks, non-zero exit on failure. It exists because four configuration
drifts happened in one evening — each silent, each producing a plausible wrong
number. The constants live in four files and nothing else checks that they still
describe the same question.

---

## Data it needs

| Path | Contents | Notes |
|---|---|---|
| `price_cache_v43/<TICKER>.pkl` | Daily OHLCV DataFrame, DatetimeIndex | Needs `Open/High/Low/Close/Volume`; ≥ 260 bars |
| `edgar_cache/` | SEC filing JSON + `company_tickers.json` | Optional — flags degrade to empty if absent |
| `pillar2_risk_calibration.json` | Fitted model | Rebuild with `--calibrate` |
| `obs_20pct_90d.pkl` | Observation table, built by V61 | Filename carries the question so it cannot be reused across settings |

---

## How to run

```bash
# 1. Fit the model  (slow — builds the observation table from the price cache)
python class_ai_pillar2_risk_v2.py --calibrate

# 2. Check the parts still agree
python deployment_preflight.py --tickers NVDA AMD MU --equity 25000

# 3. Score one name
python class_ai_pipeline_v3.py --ticker NVDA --equity 25000

# 4. Screen several, with what you already hold
python class_ai_pipeline_v3.py --screen NVDA AMD AVGO MU \
       --held NVDA:12 AVGO:8 --equity 25000

# risk layer alone, no news gate (isolates Pillar 2 for the write-up)
python class_ai_pipeline_v3.py --screen NVDA AMD --no-direction

# what v2 did, for comparison — do not trade from this
python class_ai_pipeline_v3.py --screen NVDA AMD --no-portfolio
```

`--json` on any of the above emits `{"results": [...], "portfolio": {...}}`.
Note this differs from v2, which emitted a bare result.

## Calling it from code

Every file has a `run_*()` entry point taking explicit keyword arguments. The
CLI is a thin wrapper over the same function, so anything you can do from the
shell you can do from a notebook, and get the objects back instead of text.

```python
import class_ai_pillar2_risk_v2 as P2
import class_ai_portfolio as PF
import class_ai_pipeline_v3 as PL
import deployment_preflight as PRE

# fit
calib = P2.run_calibration(universe="deploy", step=20)
calib = P2.run_calibration(tickers=["NVDA", "AMD"], out=None)   # nothing written
calib = P2.run_calibration(horizon_days=30, drawdown=0.30,      # a sweep
                           out="sweep_30d_30pct.json")

# score one name
r = P2.run_assess("NVDA", equity=25000)
r["risk"]["prob_drawdown_20pct_90d"]

# the whole pipeline
out = PL.run_screen(["NVDA", "AMD", "AVGO"], equity=25000,
                    held=["MU:8"], require_direction=False)
out["results"]      # one dict per ticker, always with final_decision
out["portfolio"]    # the book
PL.print_screen(out["results"], out["portfolio"])

# the portfolio layer on its own, sweeping the one convention in it
for budget in (3.0, 4.5, 6.0, 8.0):
    book = PF.run_book(["NVDA", "AMD", "AVGO", "MU"], equity=25000,
                       max_risk_pct=budget, verbose=False)
    print(budget, book["portfolio"]["gross_pct"],
          book["portfolio"]["binding_constraint"])

# checks, as data rather than as printed text
code, results = PRE.run_preflight(["NVDA", "AMD"], equity=25000)
[r for r in results if r[1] == "FAIL"]
```

Each evaluator works the same way — one keyword per CLI flag, `None` keeps the
default:

```python
import evaluate_scorecard_v61 as V61
V61.run_scorecard_v61(step=25, n_boot=800, isotonic=True)

import evaluate_surface_v57 as V57
V57.run_surface_v57(max_tickers=60, n_boot=200)
```

Three conventions worth knowing:

- **`run_*()` raises, `main()` exits.** The run functions raise `RuntimeError`
  on a bad input so a caller can catch it; `main()` converts that to an exit
  code. Nothing calls `sys.exit()` from inside a library path.
- **`main(argv=None)`** accepts an explicit argument list, so
  `V61.main(["--step", "25"])` works without touching `sys.argv`.
- **Module globals are never mutated to pass a setting.** `require_direction`
  is a parameter of `run_screen()` and `evaluate_trade_setup()`, not a global
  someone flips.

---

## The evaluation suite

Each script answers one question and carries its own self-checks.

| Script | Question | Verdict |
|---|---|---|
| `evaluate_sizing_v51.py` | Does sizing by the probability beat not doing so? | No — random sizing matched it; the veto lost |
| `evaluate_calibration_v52.py` | Is the probability correct? | Yes — pinned to the **original** module |
| `evaluate_volforecast_v53.py` | Which volatility estimator? | Yang-Zhang, three scales |
| `evaluate_direct_v54.py` | Do extra features help? | No — volatility is sufficient |
| `evaluate_scorecurve_v55.py` | Does a richer score help? | No — calibration cost cancels the gain |
| `evaluate_quantile_v56.py` | Whole distribution? Unseen tickers? | Binary wins; it transfers |
| `evaluate_surface_v57.py` | Which threshold and horizon? | 30%/180d was the worst usable |
| `evaluate_survivorbias_v58.py` | How much does survivorship distort? | Immaterial forward, large backward |
| `evaluate_operating_point_v59.py` | Horizon, tail bins, market state? | Only the horizon change survives |
| `evaluate_portfolio_v60.py` | Does the curve work on a book? | Conservative; the clamped bound is safe |
| `evaluate_scorecard_v61.py` | Is the output any good as a classifier? | 8.3x top-to-bottom decile, AUC 0.701 |

**`evaluate_calibration_v52.py` imports `class_ai_pillar2_risk`, not `_v2`.**
That is deliberate — its published numbers belong to the old configuration. Its
`predict()` returns NaN against the current calibration. Reuse only its metric
functions (`auc`, `brier_decomp`, `skill`, `mae`, `block_boot`, `effective_n`),
which take `p` and `y` arrays and know nothing about any model.

---

## Invariants the preflight enforces

These are the things that silently stopped agreeing before:

- `calibration["horizon_days"]` and `["drawdown"]` match `P2.HORIZON_DAYS` / `P2.DRAWDOWN`
- `P2.HORIZON_DAYS` ≥ 0.9 × `trade_config.HORIZON_CONFIGS["MID"]["eval_days"]`
- `RISK_TIERS` boundaries are sensible multiples of the calibration's base rate
- `PROB_KEY` built from `P2.DRAWDOWN` / `P2.HORIZON_DAYS` resolves in `assess()` output
- `P2.MAX_POSITION_PCT` ≤ `PF.MAX_CLUSTER_PCT` ≤ `PF.MAX_GROSS_PCT`
- `BASE_RISK_PER_TRADE` ≤ `MAX_PORTFOLIO_RISK_PCT`
- the drawdown curve is monotone
- `assess()` emits no return forecast and keeps its disclaimer

Never write a number in one file that is derived from a constant in another.
Compute it, or the preflight will eventually be the only thing that notices.

---

## Open items, and where to touch

**1. Audit the flag vetoes.** `HIGH_DILUTION` and `SHORT_RUNWAY` are the only
hard vetoes left, and they have weaker support than the veto that was removed.
V46's confound test says dilution retains 25% of its spread inside volatility
buckets — its own verdict line reads *"explained away by volatility"* — and
runway was never confound-tested at all. Both were validated at 30%/180d on 62
tickers; the model now answers 20%/90d on 337.
→ Re-run V46's confound test at the current setting, runway included; then run
V51's paired test with flags on versus off. Touch `HARD_FLAGS` in
`class_ai_pipeline_v3.py` only after that.

**2. Floor the reported probability at the calm end.** V61: the lowest decile
predicted 1.7% and realised 4.0% — a factor of 2.4 — and those names trade at
full size on a 1.00× multiplier.
→ `prob_drawdown()` in the risk module, or a floor in `risk_tier()`.

**3. Report uncertainty.** The model prints `13.2%` flat while V61 established a
typical yearly error of 2.9pp and a worst of 6.4pp.
→ `assess()`'s `risk` block.

**4. Tune the 6% portfolio budget.** It is the conventional six-percent rule,
not a measurement. Needs a portfolio-level version of V51's path simulation.
→ `MAX_PORTFOLIO_RISK_PCT` in `class_ai_portfolio.py`.

**5. Emit a term structure.** One probability, at 30/60/90/180 days. V57 says
which cells are trustworthy.
→ `assess()`'s `risk` block; the calibration would need one curve per horizon.

**6. Forward paper trading for the direction pillar.** It cannot be backtested
at any horizon, because a language model's training data contains the outcomes.

---

## Two things not to change without evidence

**The drawdown-probability veto stays off.** V51 measured refusing against
shrinking: Calmar worse by 0.020, 95% CI +0.004 to +0.035, excluding zero. It
bought 0.64pp of drawdown for 0.44pp of CAGR and dropped 8% of trades. Setting
`MAX_DRAWDOWN_PROB` to a number turns it back on.

**Pillar 2 emits no return forecast.** `assess()` returns
`"return_forecast": None` and carries a disclaimer. The preflight fails if that
changes. Nine versions established there is no directional signal to report.
