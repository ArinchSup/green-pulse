"""
CLEANED IN V106: removed 7 functions, 7 constants, the old run section and 10
imports that nothing in the current project uses. The full original is in
log/originals_v106/entry_filters_v93.py.

V93 - CAN THE TARGET HIT RATE GET BETTER?  DON'T BUY FALLING KNIVES.

WHY THE HIT RATE LOOKS ORDINARY
-------------------------------
Two things set it, and neither is a fault of the timing:

  1. The trade's shape. Target = 0.5 sigma of the holding window, stop = 0.6
     x target, so reward:risk is 1.67. For a stock that wanders randomly
     the target is hit first about 1 / (1 + 1.67) = 37.5% of the time. That
     is the break-even rate, and it is why ~37% shows up for ANY day.
  2. Where the model fires. V89 measured it: the model picks a much better
     day (+10 pp) inside stock-months that are worse than average (-9 pp),
     because it fires on stocks that are falling. Net: about level with an
     average entry.

Moving the target closer would raise the hit rate on paper and lower the
payoff by the same amount - no better trading, and the first thing a
reviewer would catch. The honest lever is (2): stop entering dips inside
downtrends. "Buy the 2-day dip only when the stock is above its 200-day
average" is the classic short-term mean-reversion rule (Connors & Alvarez,
2008) - and RSI(2), the feature this model leans on most (V89), is that
rule's own trigger.

WHAT IS TESTED (fixed before any result is seen)
------------------------------------------------
A filter only removes ENTER signals; the model and its cut do not change.

  F0  none                       V88's signals as they are (the reference)
  F1  uptrend                    the stock closes above its 200-day EMA
  F2  market not in a bear phase full-universe breadth >= 0.40
  F3  uptrend AND market         both

  CHOOSE   on the core stocks, 2017-2021: the filter with the highest
           average R per ENTER trade, if it keeps at least 100 signals there
           and beats F0. If none beats F0, no filter is adopted.
  CONFIRM  on the unseen liquid stocks, 2022-2026 - other stocks, later
           years, never used for any choice. The chosen filter is CONFIRMED
           only if all three hold:
             1. its ENTER signals hit the target more often than an average
                entry in the same month (all stocks), lower 90% bound > 0
             2. its average R per trade is at least F0's (it improves the
                trading, not just the hit rate)
             3. its timing edge over other days in the same stock and month
                stays positive (lower 90% bound > 0)

Signals come from the walk-forward copies of V88 (each year scored by a model
trained only on earlier years) - read from V86's caches, nothing retrained.

READING THE TABLES
------------------
  Target / Stop / Expired   share of ENTER trades ending each way (60 days)
  Avg R                     average result per trade in units of the stop
                            distance (+1.67 R = target, -1 R = stop)
  vs same stock-month       ENTER minus the average of ALL days in the same
                            stock and month - the timing claim
  vs average entry          ENTER minus the average of all stocks' days in
                            the same month - "better than buying anything"
"""

import warnings

import numpy as np
import pandas as pd

import sweep_options_v85 as V85

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# DATA
# =============================================================================
def outcome_indicators(U):
    lab = U["label"].to_numpy(float)
    r = U["r_multiple"].to_numpy(float)
    tgt = (lab == 1).astype(float)
    stp = ((lab != 1) & (r <= -0.999)).astype(float)
    exp = 1.0 - tgt - stp
    return {"tgt": tgt, "stp": stp, "exp": exp, "R": r}


def group_means(U, x):
    per = pd.PeriodIndex(pd.DatetimeIndex(U["date"]), freq="M")
    m_key = per.astype(str).to_numpy()
    tm_key = U["ticker"].astype(str).to_numpy() + "|" + m_key
    s = pd.Series(x)
    return (s.groupby(tm_key).transform("mean").to_numpy(),
            s.groupby(m_key).transform("mean").to_numpy())


class Stats:
    """Month-bootstrap statistics on one universe, weights shared by every
    comparison so differences are paired."""

    def __init__(self, U, n_boot):
        self.ev = V85.Evaluator(U, n_boot=n_boot)
        self.ind = outcome_indicators(U)
        self.pool, self.base = {}, {}
        for k, v in self.ind.items():
            self.pool[k], self.base[k] = group_means(U, v)

    def _sum(self, x, m, years):
        ev = self.ev
        idx, W = ev._weights(years)
        mm = m & np.isin(ev.year, years)
        k = len(ev.months)
        s = np.bincount(ev.mcode[mm], x[mm], k)[idx]
        c = np.bincount(ev.mcode[mm], minlength=k)[idx].astype(float)
        return W, s, c, mm

    def mean_ci(self, x, m, years):
        W, s, c, mm = self._sum(x, m, years)
        if c.sum() < 1:
            return np.nan, np.nan, np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (W @ s) / (W @ c)
        return s.sum() / c.sum(), np.nanpercentile(b, 5), \
            np.nanpercentile(b, 95)

    def paired(self, xa, ma, xb, mb, years):
        W, sa, ca, _ = self._sum(xa, ma, years)
        _, sb, cb, _ = self._sum(xb, mb, years)
        if ca.sum() < 1 or cb.sum() < 1:
            return np.nan, np.nan, np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (W @ sa) / (W @ ca) - (W @ sb) / (W @ cb)
        return (sa.sum() / ca.sum() - sb.sum() / cb.sum(),
                np.nanpercentile(d, 5), np.nanpercentile(d, 95))

    def row(self, m, years, f0=None):
        ind, pool, base = self.ind, self.pool, self.base
        n = int((m & np.isin(self.ev.year, years)).sum())
        if n == 0:
            return {"Signals": 0}
        mean = lambda x: self.mean_ci(x, m, years)[0]
        out = {"Signals": n,
               "Target %": mean(ind["tgt"]) * 100,
               "Stop %": mean(ind["stp"]) * 100,
               "Expired %": mean(ind["exp"]) * 100,
               "Avg R": mean(ind["R"])}
        for lab, ref in (("vs same stock-month", pool),
                         ("vs average entry", base)):
            d, lo, hi = self.mean_ci((ind["tgt"] - ref["tgt"]) * 100, m,
                                     years)
            out[f"Target {lab} pp"] = d
            out[f"Target {lab} lo"] = lo
            out[f"Target {lab} hi"] = hi
            dr, lor, hir = self.mean_ci(ind["R"] - ref["R"], m, years)
            out[f"R {lab}"] = dr
            out[f"R {lab} lo"] = lor
            out[f"R {lab} hi"] = hir
        if f0 is not None:
            d, lo, hi = self.paired(ind["R"], m, ind["R"], f0, years)
            out["R vs F0"], out["R vs F0 lo"], out["R vs F0 hi"] = d, lo, hi
        return out

    def baselines(self, m, years):
        """Outcome shares of the comparison groups for the same signals."""
        out = {}
        for lab, ref in (("all days of the same stock & month", self.pool),
                         ("average entry, same month (all stocks)",
                          self.base)):
            mm = m & np.isin(self.ev.year, years)
            out[lab] = {"Signals": int(mm.sum()),
                        "Target %": ref["tgt"][mm].mean() * 100,
                        "Stop %": ref["stp"][mm].mean() * 100,
                        "Expired %": ref["exp"][mm].mean() * 100,
                        "Avg R": ref["R"][mm].mean()}
        return out
