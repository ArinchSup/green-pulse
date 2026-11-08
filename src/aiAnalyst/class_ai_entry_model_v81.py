"""
CLEANED IN V106: removed 10 functions, 7 constants, the old run section and 10
imports that nothing in the current project uses. The full original is in
log/originals_v106/class_ai_entry_model_v81.py.

V81 - THE IMPROVED ENTRY MODEL.

WHAT CHANGED FROM V76, AND WHY EACH CHANGE IS FORCED BY A RESULT
---------------------------------------------------------------
1. SIX FEATURES, NOT 28. V79 showed single mean-reversion features beat the
   28-feature model 2x. V80 then selected features on 2008-2016 ONLY and scored
   on 2017-2026, which played no part in the choice. The six it picked:

       bb_position, rsi_14, ret_5, px_vs_ema20, range60_position, px_vs_ema50

   all with "low is good". They are fixed here and never re-selected. Four of
   the six match V79's all-years pick, so the selection is stable.

2. MONOTONE CONSTRAINTS. Each feature's direction is fixed by the
   mean-reversion hypothesis (lower = more oversold = higher score). On the
   held-out 2017-2026 period this took the restricted model from +8.48pp to
   +14.14pp - unconstrained trees spent capacity on non-monotone shapes and
   interactions that did not generalise. This is theory imposing structure, not
   tuning: the direction was set before the comparison was run.

3. TRAIN/SERVE SKEW FIXED. V70 and V76 MEASURED an XGBRegressor on R-multiples
   (PREDICT_MODE = "expectancy") but SERVED an XGBClassifier on the win label -
   a different model whose performance was never measured. Their own output
   showed it: walk-forward scores ran negative, served scores were 0.38-0.42.
   It is also why every served candidate returned prob_win 0.4154: the isotonic
   map was fitted on regression-scale scores and applied to probabilities.
   Here one function builds every model, so the served model is exactly the
   measured one, and a parity self-test checks it.

4. THE SIGN TEST NOW TESTS THE CLAIM. Every earlier sign test compared fired
   win rate to the year's UNCONDITIONAL rate. That tests an unconditional edge,
   which this model has never claimed, and it penalises any rule that picks
   oversold names (RESULTS.md section 4). The claim is conditional - a better
   DAY within a name and month already chosen - so the sign test that matches
   it asks, per year: did fired signals beat their own ticker-month pools?
   Both are printed. The conditional one is the one that tests the claim; the
   unconditional one stays as a disclosed robustness check.

ALSO REPORTED
-------------
  - the rank composite and the best single feature, in the same run, as the
    baselines this model must be read against;
  - the regime split by market breadth at entry;
  - candidates per date, because market breadth is computed over each date's
    candidates, and V80's sanity check showed breadth of exactly 0.000 on four
    crash lows - possible only if some dates carry very few candidates.
"""


import joblib


MODEL_FILE = "entry_model_v81.joblib"
QUANTILE = 0.01
WINDOW_DAYS = 252


# =============================================================================
# TRAIN
# =============================================================================
def load_model(path=MODEL_FILE):
    return joblib.load(path)
