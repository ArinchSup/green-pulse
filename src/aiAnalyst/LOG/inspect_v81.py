"""
INSPECT V81 - where did 22,401 extra signals come from?

V81 fired 2,499 signals in the held-out 2017-2026 period (~1.7% of candidates,
normal for this rule) but 24,900 over all years. That puts ~22,400 signals in
2008-2016: roughly 17% of that period's candidates, from a rule meant to fire
on 1%. The random-score arms fire exactly 1% everywhere, so the rule itself is
sound for scores whose distribution is stable. V81's scores are not.

Nothing is re-run. The per-year table is already inside entry_model_v81.joblib
(model["measured"]["per_year"]), so this loads it and prints:

  - signals per year for each arm, with years over 3x the arm's median flagged
  - matched-control excess per year, so you can see whether the flood years
    are bad signals or just too many of them
  - the candidates-per-date figures stored at training time

Read it to decide which fix V82 needs:
  * the flood sits in a few years right after a regime change (2009, 2016...)
    -> fold-to-fold score drift. Fix: threshold on each fold model's OWN
       score percentile, so a new model's scale cannot outrun a trailing cut.
  * the flood is spread evenly over the early years
    -> ties at the top of a coarse model. Fix: tie handling in the rule.
"""

import sys
import numpy as np
import pandas as pd
import joblib


def inspect(model_file="entry_model_v81.joblib"):
    m = joblib.load(model_file)
    meas = m["measured"]
    py = pd.DataFrame(meas["per_year"])
    if py.empty:
        print("no per-year table in this model file")
        return

    sig = py.pivot(index="Year", columns="Arm", values="Signals").fillna(0)
    exc = py.pivot(index="Year", columns="Arm", values="Excess win pp")
    order = [c for c in ("v81 model (monotone, 6 feats)",
                         "rank composite (same 6)",
                         "V76 deployed (28 feats)") if c in sig.columns]
    sig, exc = sig[order].astype(int), exc[order]

    print("=" * 96)
    print("  SIGNALS PER YEAR")
    print("=" * 96)
    flag = sig.copy().astype(str)
    for c in order:
        med = float(sig[c].median())
        for y in sig.index:
            v = int(sig.loc[y, c])
            flag.loc[y, c] = f"{v:>6}" + (" <<<" if med and v > 3 * med
                                          else "    ")
    print(flag.to_string())
    print("\n  totals: " + ", ".join(f"{c.split(' (')[0]} {int(sig[c].sum()):,}"
                                     for c in order))
    print("  '<<<' = more than 3x that arm's median year.")

    print("\n" + "=" * 96)
    print("  MATCHED-CONTROL EXCESS PER YEAR (pp) - are flood years bad, or "
          "just diluted?")
    print("=" * 96)
    print(exc.to_string(float_format=lambda v: f"{v:+7.2f}"))

    v = order[0]
    med = float(sig[v].median())
    flood = sig.index[sig[v] > 3 * med].tolist()
    normal = sig.index[sig[v] <= 3 * med].tolist()
    if flood:
        def wavg(yrs):
            w = sig.loc[yrs, v].astype(float)
            return float((exc.loc[yrs, v] * w).sum() / w.sum())
        print(f"\n  V81 flood years: {flood}")
        print(f"    signals in flood years : {int(sig.loc[flood, v].sum()):,}"
              f"   weighted excess {wavg(flood):+.2f} pp")
        print(f"    signals in normal years: {int(sig.loc[normal, v].sum()):,}"
              f"   weighted excess {wavg(normal):+.2f} pp")
        print("    If flood-year excess is positive but smaller, the extra "
              "signals are real setups")
        print("    admitted by a cut that went stale - a COVERAGE bug, not a "
              "validity bug.")
    else:
        print("\n  No V81 year exceeds 3x its median - the excess is spread "
              "evenly (look at ties).")

    cpd = meas.get("candidates_per_date")
    if cpd:
        print(f"\n  candidates per date at training time: median "
              f"{cpd['median']} of ~{cpd['tickers_alive_median']} tickers "
              f"alive; {cpd['share_dates_under_20']:.0%} of dates under 20")


if __name__ == "__main__":
    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE = "entry_model_v81.joblib"
    # -------------------------------------------------------------------------
    inspect(RUN_MODEL_FILE)
