"""
V104 - THE PAPER TABLES: every result of the three entry models, in one run

One file that prints every table the thesis needs for the entry-timing models
(Pillar 2), one section per model, in the paper's order:

    MID    60-day trade    final model V88   (tested in V86/V87, V95 backtest)
    SHORT  10-day trade    final model V101  (tested in V100)
    LONG   250-day trade   final model V103  (tested in V102)
    COMPARISON             the three side by side

Every model gets the SAME numbered tables, so "table 6" means the same thing
in each section:

     1  Model card - the trade, the model, the rule to beat, how it was tested
     2  The data - stocks, candidate days, signals, how every day ends
     3  The pre-registered test - every arm, the verdict, the recorded verdict
    3b  How the rule to beat was chosen - each feature alone, top 1%
    3c  Overlapping trades - the same test with block-bootstrap intervals
     4  Three universes - model against the rule: unseen liquid, all unseen,
        core stocks
     5  Timing vs selection - the claim is timing; this shows the split
     6  Backtest of every ENTER signal - hit, stop, expiry, days held, R as
        labelled and with real fills (gaps) and costs; both halves; the rule
     7  ENTER days against the alternatives - other days of the same stock and
        month, the average entry that month, any day, the rule's days
     8  Market regimes - bull, neutral, bear; model against the rule
     9  By year
    10  The random backtest - V91's random draw repeated 200 times
    11  Calibration - why the app shows a signal and not a probability
        + a REPRODUCTION CHECK against the files of the runs that decided

NOTHING IS RETRAINED OR RE-DECIDED. The data and the walk-forward predictions
are read from the caches of V86 (MID), V100 (SHORT) and V102 (LONG). Tables
3, 3b, 3c and 4 use the same random draws as the run that decided the test,
so they reproduce it exactly (MID: V86 and V87; SHORT: V100; LONG: V102).
Tables 5-9 use the draws of the run that first reported the trades (MID: V95;
SHORT: V100's trade tables; LONG: V102); numbers never reported before are
computed on those same draws. The REPRODUCTION CHECK at the end of each
section compares the recomputed numbers with the files those runs wrote.

OUTPUT: printed; one CSV per table in thesis_tables_v104/; all tables in one
Markdown file, thesis_tables_v104/PAPER_TABLES.md, to paste into the thesis.

COST: a few minutes - everything is read from caches. The ENTER trades are
re-walked from the price files for the real-fill (gap) results.
"""

import gc
import os
import textwrap
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import trade_config as TC
import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import class_ai_entry_model_v88 as M88
import entry_filters_v93 as V93
import random_backtest_v95 as V95
import short_entry_model_v100 as V100
import long_entry_model_v102 as V102

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = "thesis_tables_v104"
W = 118
LIQ, ALL = "unseen, liquid", "unseen, all"
MODEL, RULE, RANDOM, V81 = "model", "rule", "random", "V81"
FEATS, MONO = V100.FEATS, V100.MONO
COST = V100.COST                          # 0.10% round trip
BEAR, BULL = V82.BREADTH_CUTS
CHOOSE = list(range(2008, 2017))          # where the rule to beat was chosen
BREAK_EVEN = 100 * TC.SL_TP_RATIO / (1 + TC.SL_TP_RATIO)
N_PICKS = (3000, 30000)                   # V95's runs
PICK_SEED = 7                             # V95's seed
MIN_N = 30                                # fewer signals: no interval


def yrs(a, b):
    return list(range(a, b + 1))


def span(y):
    return f"{y[0]}-{y[-1]}"


# =============================================================================
# THE THREE MODELS - what each one's tables are computed on
# =============================================================================
PROFILES = {
    "MID": {
        "trade": "60-day trade", "final": "V88",
        "model_file": os.path.join("class_model", "entry_model_v88.joblib"),
        "test": yrs(2017, 2026), "full": yrs(2008, 2026),
        "halves": (yrs(2008, 2016), yrs(2017, 2026)),
        # V86's draws: one random stream (seed 81), periods in V86's order
        "test_rs": {"seed": V85.SEED, "order": [
            yrs(2017, 2026), yrs(2017, 2021), yrs(2022, 2026),
            yrs(2008, 2026)]},
        # V95's draws (seed 81, its periods' order)
        "desc_rs": {"seed": V85.SEED, "order": [
            yrs(2008, 2016), yrs(2017, 2026), yrs(2008, 2026)]},
        "choose_rs": None,                # the core's test draws
        "blocks": {"seed": V87.SEED, "n_boot": 10000},     # V87's
        "random_seed": V85.SEED,          # V86's 'random #1'
        "fixed_rule": V86.OPPONENT,
        "dec": 2,                         # V86/V87 reported two decimals
        "interval": "calendar-month bootstrap (V86); blocks of 1-12 months "
                    "(V87)",
        "chosen": "V85: eleven options and their combinations scored on the "
                  "core stocks, 2017-2021; the best confirmed once on "
                  "2022-2026",
        "rule_how": "fixed in advance: the feature ranked first in V80's "
                    "selection (core stocks, 2008-2016)",
        "tested": "V86 (pre-registered): unseen liquid stocks, 2017-2026, "
                  "two models against the rule (V81 primary, V88 secondary "
                  "with a two-test bound); V87: block bootstrap",
        "dirs": {"v86": "thesis_tables_v86", "v87": "thesis_tables_v87",
                 "v95": "thesis_tables_v95"},
    },
    "SHORT": {
        "trade": "10-day trade", "final": "V101",
        "model_file": os.path.join("class_model", "entry_model_v101.joblib"),
        "test": V100.YEARS, "full": V100.YEARS,
        "halves": tuple(V100.HALVES.values()),
        # V100's draws: seed 100; the core drew 2008-2016 first (the rule)
        "test_rs": {"seed": V100.SEED, "order": [V100.YEARS],
                    "order_core": [V100.CHOOSE_YEARS, V100.YEARS]},
        # V100's trade tables (V93's statistics: seed 81, full years first)
        "desc_rs": {"seed": V85.SEED,
                    "order": [V100.YEARS] + list(V100.HALVES.values())},
        "choose_rs": None,
        "blocks": {"seed": V100.SEED, "n_boot": None},
        "random_seed": V100.SEED,
        "fixed_rule": None,
        "dec": 1,
        "interval": "calendar-month bootstrap",
        "chosen": "V88's recipe unchanged - nothing re-tuned for the 10-day "
                  "trade",
        "rule_how": "the best single feature on the core stocks, 2008-2016 "
                    "(table 3b), chosen in V100 before any unseen-stock "
                    "result",
        "tested": "V100 (pre-registered): unseen liquid stocks, 2008-2026, "
                  "power gate 500 signals",
        "dirs": {"v100": "thesis_tables_v100"},
    },
    "LONG": {
        "trade": "250-day (one-year) trade", "final": "V103",
        "model_file": os.path.join("class_model", "entry_model_v103.joblib"),
        "test": V102.YEARS, "full": V102.YEARS,
        "halves": tuple(V102.HALVES.values()),
        # V102: circular 12-month blocks (seed 102) for everything
        "test_rs": {"seed": V102.SEED, "block": V102.BLOCK},
        "desc_rs": {"seed": V102.SEED, "block": V102.BLOCK},
        "choose_rs": {"seed": V102.SEED, "block": 1},
        "ref_rs": {"seed": V102.SEED, "block": 1},     # 'month intervals'
        "blocks": {"seed": V102.SEED, "n_boot": None},
        "random_seed": V102.SEED,
        "fixed_rule": None,
        "dec": 1,
        "interval": "circular blocks of 12 calendar months (one trade long)",
        "chosen": "V88's recipe unchanged - nothing re-tuned for the "
                  "one-year trade",
        "rule_how": "the best single feature on the core stocks, 2008-2016 "
                    "(table 3b), chosen in V102 before any unseen-stock "
                    "result",
        "tested": "V102 (pre-registered): unseen liquid stocks, 2008-2025 "
                  "(a label needs a year of future prices), power gate 500 "
                  "signals, 12-month block intervals",
        "dirs": {"v102": "thesis_tables_v102"},
    },
}


# =============================================================================
# 1. RANDOM DRAWS - one set per period, shared by every number of a table
# =============================================================================
class Resampler:
    """
    Calendar months resampled n_boot times. Every number in a table is
    averaged over the SAME resampled months, so differences are paired.

      block None  V85's month bootstrap (V86, V95, V100): one random stream,
                  each period drawn the first time it is used - so `order`
                  replays the order of the run being reproduced
      block L     V87/V102's circular blocks of L consecutive months: a
                  period's draws depend only on (seed, L, first, last year)
    """

    def __init__(self, U, n_boot, seed, block=None, order=()):
        per = pd.PeriodIndex(pd.DatetimeIndex(U["date"]), freq="M")
        self.mcode, self.months = pd.factorize(per)
        self.myear = np.array([m.year for m in self.months])
        self.year = U["year"].to_numpy()
        self.n_boot, self.seed, self.block = int(n_boot), seed, block
        self.rng = np.random.default_rng(seed)
        self._W = {}
        for y in order:
            self.weights(y)

    def weights(self, years):
        key = tuple(years)
        if key not in self._W:
            idx = np.flatnonzero(np.isin(self.myear, years))
            n = len(idx)
            if self.block is None:                     # V85.Evaluator
                W = (self.rng.multinomial(n, np.full(n, 1 / n),
                                          size=self.n_boot).astype(float)
                     if n else np.zeros((self.n_boot, 0)))
            else:                                      # V87 / V102 blocks
                idx = idx[np.argsort(self.months[idx].to_timestamp().values)]
                L = int(self.block)
                rng = np.random.default_rng([self.seed, L, years[0],
                                             years[-1]])
                if n:
                    k = int(np.ceil(n / L))
                    starts = rng.integers(0, n, size=(self.n_boot, k))
                    flat = ((starts[:, :, None] + np.arange(L)) % n) \
                        .reshape(self.n_boot, -1)[:, :n]
                    flat = flat + np.arange(self.n_boot)[:, None] * n
                    W = np.bincount(flat.ravel(),
                                    minlength=self.n_boot * n) \
                        .reshape(self.n_boot, n).astype(float)
                else:
                    W = np.zeros((self.n_boot, 0))
            self._W[key] = (idx, W)
        return self._W[key]

    def mean(self, x, m, years):
        """Mean of x over the rows in m (within years): estimate, the
        bootstrap draws, and the number of rows."""
        idx, W = self.weights(years)
        x = np.asarray(x, float)
        mm = m & np.isin(self.year, years) & np.isfinite(x)
        K = len(self.months)
        s = np.bincount(self.mcode[mm], x[mm], K)[idx]
        c = np.bincount(self.mcode[mm], minlength=K)[idx].astype(float)
        n = int(c.sum())
        if n == 0:
            return np.nan, np.full(self.n_boot, np.nan), 0
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (W @ s) / (W @ c)
        return s.sum() / c.sum(), draws, n

    def diff(self, xa, ma, xb, mb, years):
        pa, da, na = self.mean(xa, ma, years)
        pb, db, nb = self.mean(xb, mb, years)
        return pa - pb, da - db, na, nb


def bounds(draws, a=5.0):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return ((np.percentile(d, a), np.percentile(d, 100 - a)) if d.size
            else (np.nan, np.nan))


def p_le0(draws):
    """Share of draws <= 0 (an undefined draw counts as above 0, as V85)."""
    return float(np.mean(np.asarray(draws, float) <= 0))


# =============================================================================
# 2. FORMATTING AND OUTPUT
# =============================================================================
def pct(v, d=1):
    return f"{v:.{d}f}" if v is not None and np.isfinite(v) else "-"


def sg(v, d=3):
    return f"{v:+.{d}f}" if v is not None and np.isfinite(v) else "-"


def ci_txt(p, draws, n=None, d=1):
    if p is None or not np.isfinite(p):
        return "-"
    if n is not None and n < MIN_N:
        return f"{p:+.{d}f} (n<{MIN_N})"
    lo, hi = bounds(draws)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{p:+.{d}f}"
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|")
                                         for v in r.values) + " |")
    return "\n".join(lines)


def print_df(df, width=W):
    """Print a table; split its columns into blocks if it is too wide."""
    def wide(d):
        return max(len(s) for s in d.to_string(index=False).splitlines())
    if df.shape[1] <= 3 or wide(df) <= width:
        print(df.to_string(index=False))
        return
    first, blocks, cur = df.columns[0], [], []
    for c in df.columns[1:]:
        if cur and wide(df[[first] + cur + [c]]) > width:
            blocks.append(cur)
            cur = []
        cur.append(c)
    blocks.append(cur)
    for i, b in enumerate(blocks):
        if i:
            print()
        print(df[[first] + b].to_string(index=False))


def print_kv(df, width=W):
    """A two-column table as 'item  value' lines, the value wrapped."""
    k = max(len(str(v)) for v in df.iloc[:, 0]) + 2
    for a, b in zip(df.iloc[:, 0], df.iloc[:, 1]):
        lines = textwrap.wrap(str(b), width=max(width - k - 2, 30)) or [""]
        print(f"  {str(a):<{k}}{lines[0]}")
        for line in lines[1:]:
            print(" " * (k + 2) + line)


def indent(text, pad="  "):
    return "\n".join(pad + line if line else line
                     for line in text.strip("\n").split("\n"))


class Report:
    def __init__(self, out_dir):
        self.out_dir, self.md, self.n = out_dir, [], 0
        os.makedirs(out_dir, exist_ok=True)

    def heading(self, title, intro=""):
        print("\n\n" + "#" * W + f"\n  {title}\n" + "#" * W)
        if intro:
            print(indent(intro))
        self.md.append(f"\n## {title}\n\n" + (intro.strip() + "\n" if intro
                                               else ""))

    def table(self, key, title, df, note="", kv=False, show=None):
        """Print, save as CSV and add to the Markdown file. `show` (a list
        of tables) replaces the printout only; the files get df."""
        self.n += 1
        df = df.copy().fillna("").astype(str)
        df.to_csv(os.path.join(self.out_dir, f"{key}.csv"), index=False)
        print("\n" + "=" * W + f"\n  {title}\n" + "=" * W)
        if note:
            print(indent(note) + "\n")
        if kv:
            print_kv(df)
        elif show is not None:
            for i, d in enumerate(show):
                if i:
                    print()
                print_df(d.fillna("").astype(str))
        else:
            print_df(df)
        self.md.append(f"### {title}\n\n" + (note.strip() + "\n\n" if note
                                             else "") + md_table(df) + "\n")

    def text(self, title, body):
        print("\n" + "-" * W + f"\n  {title}\n" + "-" * W)
        print(indent(body))
        self.md.append(f"**{title}**\n\n```\n{body.strip()}\n```\n")

    def save(self, header):
        p = os.path.join(self.out_dir, "PAPER_TABLES.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(header + "\n" + "\n".join(self.md))
        print(f"\n  wrote {self.n} tables to {self.out_dir}/ (one CSV each) "
              f"and {p}")


def _num(v):
    if v is None or not np.isfinite(v):
        return "-"
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1:
        return f"{int(round(v)):,}"
    return f"{v:+.4f}"


class Checks:
    """Recomputed numbers against the files of the runs that decided."""

    def __init__(self):
        self.rows = []

    def _add(self, what, mine, ref, src, same):
        self.rows.append({"Number": what, "This run": mine, "Recorded": ref,
                          "File": src, "Result": same})

    def num(self, what, mine, ref, src, tol=1e-6):
        mine = float(mine)
        if ref is None:
            return self._add(what, _num(mine), "-", src, "not recorded")
        try:
            ref = float(ref)
        except (TypeError, ValueError):
            return self._add(what, _num(mine), str(ref), src, "not recorded")
        if not np.isfinite(ref) and not np.isfinite(mine):
            return self._add(what, "-", "-", src, "SAME")
        ok = np.isfinite(ref) and np.isfinite(mine) and abs(mine - ref) <= tol
        self._add(what, _num(mine), _num(ref), src,
                  "SAME" if ok else "*** DIFFERENT ***")

    def txt(self, what, mine, ref, src):
        if ref is None:
            return self._add(what, str(mine), "-", src, "not recorded")
        ok = str(mine).strip() == str(ref).strip()
        self._add(what, str(mine), str(ref), src,
                  "SAME" if ok else "*** DIFFERENT ***")

    def missing(self, src):
        self._add("(the whole file)", "", "", src, "file not found")

    def brief(self, T):
        """Per file: how many numbers were compared and how many match;
        then every number that does not."""
        g = T.assign(n=1, same=(T["Result"] == "SAME").astype(int),
                     diff=T["Result"].str.contains("DIFFERENT").astype(int))
        per = (g.groupby("File", sort=False)[["n", "same", "diff"]].sum()
               .reset_index().rename(columns={"n": "Numbers compared",
                                              "same": "Same",
                                              "diff": "Different"}))
        per.loc[T.groupby("File", sort=False)["Result"].first()
                .eq("file not found").to_numpy(), "Numbers compared"] = 0
        bad = T[T["Result"] != "SAME"]
        return [per] + ([bad] if len(bad) else [])

    def summary(self):
        T = pd.DataFrame(self.rows)
        if not len(T):
            return T, "nothing recorded to compare"
        k = int((T["Result"] == "SAME").sum())
        bad = int(T["Result"].str.contains("DIFFERENT").sum())
        n = k + bad
        gone = int((T["Result"] == "file not found").sum())
        if n == 0:
            return T, (f"nothing to compare: {gone} recorded file(s) not "
                       f"found - check the folder names")
        line = (f"{k} of {n} recorded numbers reproduced exactly"
                if not bad else f"*** {bad} of {n} recorded numbers DIFFER "
                                f"- see the table")
        if gone:
            line += f"; {gone} recorded file(s) not found"
        return T, line


def read_csv(path, as_text=False):
    if not os.path.exists(path):
        return None
    if as_text:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path)


def read_text(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) \
        else ""


# =============================================================================
# 3. DATA - each model's data and walk-forward predictions, from its caches
# =============================================================================
def load_mid(core_cache, big_cache, n_seeds, chunk, model_in81, verbose):
    prov = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov["price_cache"]
    core, fresh, wf, _ = V87.prepare(model_in81, core_cache, big_cache,
                                     n_seeds, chunk, V86.LIQ_PCT,
                                     V86.DUP_CORR, verbose=verbose)
    hz = prov["horizon"]
    g = {"key": hz, "hold": int(TC.HORIZON_CONFIGS[hz]["lookahead_bars"]),
         "k": TC.VOL_K, "sl": TC.SL_TP_RATIO, "floor": TC.VOL_MIN_TP_PCT,
         "ceil": TC.VOL_MAX_TP_PCT}
    return {"core": core, "fresh": fresh, "wf": wf, "g": g,
            "core_cache": core_cache, "big_cache": big_cache,
            "core_names": V86.cache_names(core_cache), "step": prov["step"]}


def _load_new(mod, hold, core_cache, big_cache, n_seeds, chunk, model_in81,
              verbose):
    core_cache = core_cache or M81.load_model(model_in81)["provenance"][
        "price_cache"]
    g = mod.geometry(hold)
    core, fresh, wf, core_names, _ = mod.prepare(core_cache, big_cache, g,
                                                 n_seeds, chunk, verbose)
    return {"core": core, "fresh": fresh, "wf": wf, "g": g,
            "core_cache": core_cache, "big_cache": big_cache,
            "core_names": core_names, "step": mod.STEP}


def load_data(h, core_cache, big_cache, n_seeds, chunk, model_in81, model,
              verbose):
    if h == "MID":
        return load_mid(core_cache, big_cache, n_seeds, chunk, model_in81,
                        verbose)
    mod = V100 if h == "SHORT" else V102
    hold = int((model or {}).get("geometry", {}).get("hold", mod.HOLD))
    return _load_new(mod, hold, core_cache, big_cache, n_seeds, chunk,
                     model_in81, verbose)


_BREADTH = {}


def breadth(core_cache, names):
    key = (core_cache, tuple(names))
    if key not in _BREADTH:
        _BREADTH[key] = V82.full_breadth(core_cache, names)[0]
    return _BREADTH[key]


class Uni:
    """One universe: its candidate days, how each one ended, the matched
    comparisons (same stock and month; same month, all stocks) and the
    signals."""

    def __init__(self, name, U, w, cache, kind):
        self.name, self.U, self.w, self.cache, self.kind = (name, U, w, cache,
                                                             kind)
        self.year = U["year"].to_numpy()
        self.br = U["breadth"].to_numpy(float)
        ind = V93.outcome_indicators(U)
        self.o = {k: ind[k] for k in ("tgt", "stp", "exp", "R")}
        tcode = pd.factorize(U["ticker"].astype(str).to_numpy())[0]
        mcode = pd.factorize(pd.PeriodIndex(pd.DatetimeIndex(U["date"]),
                                            freq="M"))[0]
        tm = pd.factorize(tcode.astype(np.int64) * (int(mcode.max()) + 1)
                          + mcode)[0]

        def gm(x, c):                       # V93.group_means, same sums
            return pd.Series(x).groupby(c).transform("mean").to_numpy()
        self.pool = {k: gm(v, tm) for k, v in self.o.items()}
        self.base = {k: gm(v, mcode) for k, v in self.o.items()}
        t, r = self.o["tgt"], self.o["R"]
        self.x = {"tgt": t * 100,
                  "sm": (t - self.pool["tgt"]) * 100,
                  "sel": (self.pool["tgt"] - self.base["tgt"]) * 100,
                  "avg": (t - self.base["tgt"]) * 100,
                  "smR": r - self.pool["R"],
                  "selR": self.pool["R"] - self.base["R"],
                  "avgR": r - self.base["R"]}
        self.F, self.T, self.D, self.walk = {}, None, None, None

    def yr(self, years):
        return np.isin(self.year, years)


# =============================================================================
# 4. REAL FILLS - every ENTER trade re-walked from the price files
# =============================================================================
def walk_pairs(pairs, cache, g):
    """The label's levels (entry at the close, trade_config's stop and
    target at that day's ATR) walked again: target and stop %, days held, R
    as labelled (a check) and R with real fills - a day that OPENS beyond the
    stop or the target exits at that open (V100.gap_walk)."""
    out = []
    with V100.short_limits(g):
        for t, grp in pairs.groupby("ticker", sort=False):
            df = E64.P2.load_prices(t, cache)
            if df is None or not len(df):
                continue
            atr = E64._atr(df, 14).to_numpy(float)
            o, hi, lo, c = (df[k].to_numpy(float)
                            for k in ("Open", "High", "Low", "Close"))
            pos = pd.Index(df.index).get_indexer(pd.DatetimeIndex(grp["date"]))
            for d, p in zip(grp["date"].to_numpy(), pos):
                if p < 0:
                    continue
                lv = TC.compute_levels(c[p], atr[p], g["key"])
                if not lv.get("valid"):
                    continue
                lab = E64.triple_barrier(hi[p + 1:], lo[p + 1:], c[p + 1:],
                                         c[p], lv["stop"], lv["target"],
                                         g["hold"])
                if lab is None:
                    continue
                real = V100.gap_walk(o[p + 1:], hi[p + 1:], lo[p + 1:],
                                     c[p + 1:], c[p], lv["stop"],
                                     lv["target"], g["hold"])
                out.append((t, d, lv["target_pct"], lv["stop_pct"],
                            float(lab[2]), float(lab[1]), float(real)))
    res = pd.DataFrame(out, columns=["ticker", "date", "target_pct",
                                     "stop_pct", "held", "r_label", "r_real"])
    res["date"] = pd.to_datetime(res["date"])
    return res


def walk_universes(unis, g, keys=(MODEL, RULE)):
    need, rows = {}, {}
    for name, u in unis.items():
        m = np.zeros(len(u.U), bool)
        for k in keys:
            m |= u.F[k]
        rows[name] = np.flatnonzero(m)
        need.setdefault(u.cache, []).append(u.U.loc[rows[name],
                                                    ["ticker", "date"]])
    done = {}
    for cache, parts in need.items():
        pairs = pd.concat(parts, ignore_index=True).drop_duplicates()
        done[cache] = walk_pairs(pairs, cache, g)
    for name, u in unis.items():
        got = u.U.loc[rows[name], ["ticker", "date"]].merge(
            done[u.cache], on=["ticker", "date"], how="left")
        u.walk = {}
        for k in ("target_pct", "stop_pct", "held", "r_label", "r_real"):
            a = np.full(len(u.U), np.nan)
            a[rows[name]] = got[k].to_numpy(float)
            u.walk[k] = a
        u.walk["real"] = u.walk["r_real"] - COST / (u.walk["stop_pct"] / 100)
        u.walk["rows"] = rows[name]


# =============================================================================
# 5. TABLE ROWS
# =============================================================================
def trade_row(label, u, f, years):
    """Every ENTER trade in f within years: how it ended, its levels, and R
    as labelled and with real fills and costs."""
    mm = f & u.yr(years)
    n = int(mm.sum())
    if not n:
        return {"": label, "Signals": "0"}
    t, s, e = (u.o[k][mm] for k in ("tgt", "stp", "exp"))
    res = t.sum() + s.sum()
    wk = u.walk
    real = wk["real"][mm]
    return {"": label, "Signals": f"{n:,}",
            "Hit target %": pct(t.mean() * 100),
            "Stop %": pct(s.mean() * 100),
            "Expired %": pct(e.mean() * 100),
            "Resolved hit %": pct(t.sum() / res * 100 if res else np.nan),
            "Median target %": pct(np.nanmedian(wk["target_pct"][mm])),
            "Median stop %": pct(np.nanmedian(wk["stop_pct"][mm])),
            "Days held": pct(np.nanmean(wk["held"][mm])),
            "Avg R": sg(np.nanmean(u.o["R"][mm])),
            "Avg R, gaps + costs": (sg(np.nanmean(real))
                                    if np.isfinite(real).any() else "-")}


def levels(ref, mask):
    """Outcome shares of a group (ref = outcomes, or their pool/base means)
    over the rows in mask."""
    tt, ss = ref["tgt"][mask].mean() * 100, ref["stp"][mask].mean() * 100
    return {"Hit target %": pct(tt), "Stop %": pct(ss),
            "Expired %": pct(ref["exp"][mask].mean() * 100),
            "Resolved hit %": pct(tt / (tt + ss) * 100 if tt + ss else np.nan),
            "Avg R": sg(np.nanmean(ref["R"][mask]))}


# =============================================================================
# 6. ONE MODEL'S SECTION
# =============================================================================
def model_section(R, h, data, model, model_file, n_boot, n_runs, dirs,
                  n_seeds, verbose=True):
    P = PROFILES[h]
    t0 = time.time()
    g = data["g"]
    test, full, (h1, h2) = P["test"], P["full"], P["halves"]
    dec = P["dec"]
    core_names = data["core_names"]
    CORE = f"core {len(core_names)}"
    key = h.lower()
    chk = Checks()
    got = {}
    R.heading(f"{h} - {P['trade']} (final model {P['final']})",
              f"Trade: {g['hold']} trading days. Test: {P['tested']}.\n"
              f"Intervals: {P['interval']}. Pre-registered years "
              f"{span(test)}; descriptive tables {span(full)}.")

    # ---- universes and signals -------------------------------------------
    print(f"\n  [{h}] universes, signals and matched comparisons ...")
    b = breadth(data["core_cache"], core_names)
    core, fresh, wf = data.pop("core"), data.pop("fresh"), data.pop("wf")
    n_unseen = int(fresh["ticker"].nunique())
    unis = {}
    for name, frame, mask, tag, cache, kind in (
            (LIQ, fresh, fresh["liquid"].to_numpy(bool), "fresh",
             data["big_cache"], "liq"),
            (ALL, fresh, np.ones(len(fresh), bool), "fresh",
             data["big_cache"], "all"),
            (CORE, core, np.ones(len(core), bool), "core",
             data["core_cache"], "core")):
        U, w = V87.universe(frame, mask, wf, tag)
        U["breadth"] = U["date_n"].map(b).to_numpy(float)
        u = Uni(name, U, w, cache, kind)
        u.F[MODEL] = V85.model_fire(U, *w["combo"])
        if "V81" in w:
            u.F[V81] = V85.model_fire(U, *w["V81"])
        u.F[RANDOM] = V100.rule(U, np.random.default_rng(
            P["random_seed"]).random(len(U)))
        for nm in ("test_rs", "desc_rs"):
            s = P[nm]
            order = s.get("order_" + kind, s.get("order", ()))
            rs = Resampler(U, n_boot, s["seed"], s.get("block"), order)
            if nm == "test_rs":
                u.T = rs
            else:
                u.D = rs
        unis[name] = u
    del core, fresh, wf
    gc.collect()

    # ---- 3b first: the rule to beat ---------------------------------------
    uc = unis[CORE]
    cs = P["choose_rs"]
    CH = uc.T if cs is None else Resampler(uc.U, n_boot, cs["seed"],
                                           cs.get("block"))
    rows, lifts, fires = [], [], {}
    for f, m in zip(FEATS, MONO):
        nm = V100.rule_name(f, m)
        fire = V100.rule(uc.U, m * uc.U[f].to_numpy(float))
        fires[nm] = (f, m)
        p, dr, n = CH.mean(uc.x["sm"], fire, CHOOSE)
        lifts.append(p if n >= MIN_N else np.nan)
        rows.append({"Rule (top 1%)": nm, "Signals": f"{n:,}",
                     "Lift over same stock-month, pp [90%]":
                         ci_txt(p, dr, n, dec)})
    lifts = np.array(lifts, float)
    chosen = (list(fires)[int(np.nanargmax(lifts))] if np.isfinite(lifts).any()
              else None)
    rule = P["fixed_rule"] or chosen
    got["choose"] = dict(zip(fires, lifts))
    got["chosen"] = chosen
    if rule is None:
        raise SystemExit(f"{h}: no single-feature rule fired 30 times on the "
                         f"core, 2008-2016 - nothing to compare against")
    for r_ in rows:
        r_[""] = ("<- the rule to beat" if r_["Rule (top 1%)"] == rule
                  else "<- best here" if r_["Rule (top 1%)"] == chosen
                  else "")
    T3b = pd.DataFrame(rows)
    rf, rm = fires[rule]
    for u in unis.values():
        u.F[RULE] = V100.rule(u.U, rm * u.U[rf].to_numpy(float))

    # ---- real fills -------------------------------------------------------
    n_walk = sum(int((u.F[MODEL] | u.F[RULE]).sum()) for u in unis.values())
    print(f"  [{h}] re-walking {n_walk:,} ENTER trades (model and rule) from "
          f"the price files for real fills ...")
    walk_universes(unis, g)
    lab_ok, lab_n = 0, 0
    for u in unis.values():
        rr = u.walk["rows"]
        a, b_ = u.walk["r_label"][rr], u.o["R"][rr]
        k = np.isfinite(a) & np.isfinite(b_)
        lab_n += int(k.sum())
        lab_ok += int((np.abs(a[k] - b_[k]) < 1e-9).sum())

    # ================================================================ TABLE 1
    params = dict(E64.XGB_PARAMS)
    params.update(V85.HPARAMS[V86.COMBO_CFG["hp"]])
    emb = int(g["hold"] * 365.25 / 252) + 5
    ns = int((model or {}).get("provenance", {}).get("n_seeds", n_seeds))
    card = [
        ("Trade", f"buy at the signal day's close; target = {g['k']} x ATR% "
                  f"x sqrt({g['hold']}), kept within {g['floor']:.2%} to "
                  f"{g['ceil']:.2%}; stop = {g['sl']} x target"),
        ("Exit", f"at the target, the stop, or the close after {g['hold']} "
                 f"trading days; a day that touches both counts as the stop"),
        ("Reward : risk", f"{1 / g['sl']:.2f} : 1 - break-even "
                          f"{BREAK_EVEN:.1f}% of the trades that reach the "
                          f"target or the stop"),
        ("Model", "XGBoost classifier: the probability that the target is "
                  "hit before the stop"),
        ("Features (all monotone: lower is better)", ", ".join(FEATS)),
        ("Hyperparameters", ", ".join(
            f"{k}={params[k]}" for k in ("n_estimators", "max_depth",
                                         "learning_rate", "min_child_weight",
                                         "subsample", "colsample_bytree",
                                         "reg_lambda"))),
        ("Ensemble", (f"{ns} seeds ({V85.SEED}-{V85.SEED + ns - 1}), "
                      f"probabilities averaged" if ns > 1 else
                      f"1 seed ({V85.SEED})")),
        ("ENTER signal", "score STRICTLY above the 99th percentile of the "
                         "universe's own scores over the trailing 252 "
                         "trading days, cut refit monthly - about 1 day in "
                         "100"),
        ("Walk-forward", f"a new model each year, trained on the "
                         f"{len(core_names)} core stocks only, with data "
                         f"ending {emb} calendar days before the year (the "
                         f"longest trade plus 5 days); every signal in these "
                         f"tables comes from a model that never saw that "
                         f"year or that stock"),
        ("How the recipe was chosen", P["chosen"]),
        ("Rule to beat", f"{rule} - {P['rule_how']}"),
        ("Pre-registered test", P["tested"]),
        ("Unseen stocks", f"{n_unseen:,} stocks never used in training "
                          f"(near-copies of core stocks left out); 'liquid' "
                          f"= 20-day dollar volume above the core stocks' "
                          f"10th percentile, refit monthly"),
        ("Candidate days", f"every {data['step']} trading days per stock")]
    if model:
        p_ = model.get("provenance", {})
        card.append(("Final model", f"{os.path.basename(model_file)}: "
                                    f"trained on "
                                    f"{p_.get('n_tickers', '?')} stocks, "
                                    f"{p_.get('n_rows', 0):,} candidate "
                                    f"days, {p_.get('from', '?')} to "
                                    f"{p_.get('trained_through', '?')}"))
        par = model.get("parity", {})
        if "measured" in par:
            m_ = par["measured"]
            card.append(("Parity: final builder vs the tested fold models",
                         f"max |difference| {m_['max_abs_diff']:.1e} on "
                         f"{m_['rows']:,} rows ({m_['year']} fold)"))
        if "serving" in par:
            s_ = par["serving"]
            card.append(("Parity: the app's features vs training features",
                         f"max |difference| {s_['max_abs_diff']:.1e} on "
                         f"{s_['shared_rows']:,} of {s_['train_rows']:,} "
                         f"rows"))
        card.append(("Probability shown in the app",
                     "yes" if model.get("display_probability") else
                     "no - the app shows ENTER / WAIT and the measured hit "
                     "rate (table 11)"))
    else:
        card.append(("Final model", f"{os.path.basename(model_file)} not "
                                    f"found - run its "
                                    f"trainer (the tables below do not need "
                                    f"it)"))
    R.table(f"{key}_t01_model_card", f"{h} 1. MODEL CARD - {P['final']}, "
            f"{P['trade']}", pd.DataFrame(card, columns=["Item", "Value"]),
            kv=True)

    # ================================================================ TABLE 2
    rows = []
    for name in (LIQ, ALL, CORE):
        u = unis[name]
        yf = u.yr(full)
        n = int(yf.sum())
        ns_ = int((u.F[MODEL] & yf).sum())
        lv = levels(u.o, yf) if n else {}
        rows.append({"Universe": name,
                     "Stocks": f"{u.U.loc[yf, 'ticker'].nunique():,}",
                     "Candidate days": f"{n:,}", "ENTER signals": f"{ns_:,}",
                     "ENTER per 100 days": pct(ns_ / n * 100, 2) if n else "-",
                     **{f"Every day: {k}": v for k, v in lv.items()},
                     "Last candidate day": str(u.U["date"].max().date())})
    R.table(f"{key}_t02_data", f"{h} 2. THE DATA - candidate days and how "
            f"they end, {span(full)}", pd.DataFrame(rows),
            note=f"A candidate day every {data['step']} trading days per "
                 f"stock. 'Every day' = all candidate days, signal or not: "
                 f"the base rate.\nResolved hit % = target / (target + stop)"
                 f"; break-even {BREAK_EVEN:.1f}%.")

    # ================================================================ TABLE 3
    u = unis[LIQ]
    T = u.T
    yt = u.yr(test)
    arms = [(MODEL, f"{P['final']} (final model)")]
    if V81 in u.F:
        arms.append((V81, "V81 (previous model)"))
    arms += [(RULE, f"{rule} (rule to beat)"),
             (RANDOM, "random 1% (yardstick)")]
    rows, t3 = [], {}
    for k_, label in arms:
        f = u.F[k_]
        p, dr, n = T.mean(u.x["sm"], f, test)
        pr, _, _ = T.mean(u.x["smR"], f, test)
        row = {"Arm": label, "Signals": f"{n:,}",
               "Stocks": f"{u.U.loc[f & yt, 'ticker'].nunique():,}",
               "Lift over same stock-month, pp [90%]": ci_txt(p, dr, n, dec),
               "Excess R": sg(pr)}
        t3[k_] = {"p": p, "dr": dr, "n": n, "er": pr}
        if k_ != RULE:
            dp, dd, na, nb = T.diff(u.x["sm"], f, u.x["sm"], u.F[RULE], test)
            ok = min(na, nb) >= MIN_N
            row["Minus the rule, pp [90%]"] = ci_txt(dp, dd, min(na, nb), dec)
            row["P(minus the rule <= 0)"] = f"{p_le0(dd):.3f}" if ok else "-"
            t3[k_].update({"dp": dp, "dd": dd, "ok": ok})
        else:
            row["Minus the rule, pp [90%]"] = ""
            row["P(minus the rule <= 0)"] = ""
        both = int((f & u.F[MODEL] & yt).sum())
        row["Shared with the model"] = ("" if k_ == MODEL else
                                        f"{both / max(n, 1):.0%}")
        rows.append(row)
    got["t3"] = t3
    m3 = t3[MODEL]
    lo_m = bounds(m3["dr"])[0]
    timing = ("CONFIRMED" if m3["n"] >= MIN_N and np.isfinite(lo_m)
              and lo_m > 0 else "NOT CONFIRMED")
    lines = [f"1. TIMING - ENTER days vs all days of the same stock and "
             f"month: {ci_txt(m3['p'], m3['dr'], m3['n'], dec)} pp -> "
             f"{timing}"]
    if h == "MID":
        v = t3[V81]
        lo, hi = bounds(v["dd"])
        v81 = ("BEATS the rule" if lo > 0 else "WORSE than the rule"
               if hi < 0 else "AHEAD BUT NOT PROVEN" if v["dp"] > 0 else
               "DOES NOT BEAT the rule")
        lo2 = float(np.nanpercentile(m3["dd"], 2.5))
        v88 = ("BEATS the rule" if lo2 > 0 else "DOES NOT BEAT the rule")
        lines += [f"2. RULE, primary (V86): V81 minus '{rule}': "
                  f"{ci_txt(v['dp'], v['dd'], None, dec)} pp -> {v81}",
                  f"3. RULE, secondary (V86, two tests): V88 minus '{rule}': "
                  f"{ci_txt(m3['dp'], m3['dd'], None, dec)} pp; two-test "
                  f"lower bound (2.5%) {lo2:+.{dec}f} -> {v88}"]
        rule_v, rule_txt = v88, (f"{ci_txt(m3['dp'], m3['dd'], None, 1)}; "
                                 f"two-test bound {lo2:+.2f}")
    else:
        lo, hi = bounds(m3["dd"])
        rule_v = ("NOT MEASURABLE" if not m3["ok"] else
                  "BEATS the rule" if lo > 0 else "WORSE than the rule"
                  if hi < 0 else "AHEAD BUT NOT PROVEN" if m3["dp"] > 0 else
                  "DOES NOT BEAT the rule")
        rule_txt = ci_txt(m3["dp"], m3["dd"], None, 1)
        lines.append(f"2. RULE - model minus '{rule}', paired: "
                     f"{ci_txt(m3['dp'], m3['dd'], None, dec)} pp -> "
                     f"{rule_v}")
    if "ref_rs" in P:
        s = P["ref_rs"]
        u.Rf = Resampler(u.U, n_boot, s["seed"], s.get("block"))
        got["t3ref"] = {}
        ref_txt = []
        for k_, label in arms:
            p, dr, n = u.Rf.mean(u.x["sm"], u.F[k_], test)
            got["t3ref"][k_] = (p, dr, n)
            ref_txt.append(f"{label.split(' (')[0]} {ci_txt(p, dr, n, dec)}")
        lines.append("   (plain month intervals, for reference only: "
                     + "; ".join(ref_txt) + ")")
    R.table(f"{key}_t03_preregistered_test",
            f"{h} 3. THE PRE-REGISTERED TEST - {LIQ} stocks, {span(test)}",
            pd.DataFrame(rows),
            note="Lift = ENTER days' target hit rate minus all days of the "
                 "same stock and month (pp). 'Minus the rule'\nis paired: "
                 "both arms are averaged over the same resampled months. "
                 f"Intervals: {P['interval'].split(';')[0]}.\n\n"
                 + "\n".join(lines))
    body = "\n\n".join(
        f"[{os.path.join(dirs[d_], 'verdict.txt')}]\n"
        + (textwrap.dedent(read_text(os.path.join(
            dirs[d_], "verdict.txt"))).strip("\n") or "(not found)")
        for d_ in ("v86", "v87", "v100", "v102") if d_ in dirs)
    R.text(f"{h} 3. THE VERDICT AS RECORDED when the test was run", body)

    # =============================================================== TABLE 3b
    R.table(f"{key}_t03b_rule_choice",
            f"{h} 3b. HOW THE RULE TO BEAT WAS CHOSEN - each feature alone, "
            f"top 1%, {CORE}, {span(CHOOSE)}", T3b,
            note=("The rule for the 60-day trade was fixed in advance (V80/"
                  "V85: low bb_position). This table uses\nthe same "
                  "procedure as SHORT and LONG, for comparison only - it "
                  "chose nothing." if h == "MID" else
                  "The rule with the best lift here was fixed as the "
                  "opponent before any unseen-stock result was\ncomputed "
                  "(the choice uses the estimate).")
                 + "\nA rule needs a score STRICTLY above its cut, so a "
                   "feature that often ties at its cut (a close\nexactly at "
                   "the 10-day low) fires rarely or never.")

    # =============================================================== TABLE 3c
    bs = P["blocks"]
    nbb = int(bs["n_boot"] or n_boot)
    rows, got["t3c"] = [], {}
    for L in (1, 3, 6, 12):
        B = Resampler(u.U, nbb, bs["seed"], L)
        p, dr, n = B.mean(u.x["sm"], u.F[MODEL], test)
        dp, dd, na, nb = B.diff(u.x["sm"], u.F[MODEL], u.x["sm"], u.F[RULE],
                                test)
        got["t3c"][L] = (p, dr, dp, dd)
        row = {"Block (months)": str(L),
               "Timing lift, pp [90%]": ci_txt(p, dr, n, dec),
               "Model minus rule, pp [90%]": ci_txt(dp, dd, min(na, nb),
                                                    dec)}
        if h == "MID":
            row["Two-test bound (2.5%)"] = sg(float(np.nanpercentile(
                dd, 2.5)), dec)
        row["P(model - rule <= 0)"] = f"{p_le0(dd):.3f}"
        rows.append(row)
        del B
    R.table(f"{key}_t03c_block_bootstrap",
            f"{h} 3c. OVERLAPPING TRADES - the test with months resampled in "
            f"blocks, {LIQ}, {span(test)}", pd.DataFrame(rows),
            note=f"Trades that start a few weeks apart share part of their "
                 f"path; resampling blocks of consecutive months keeps\n"
                 f"that dependence. {nbb:,} draws, seed {bs['seed']}"
                 + (" (V87's run: these rows reproduce it)." if h == "MID"
                    else " (the 12-month row is V102's test)." if h == "LONG"
                    else " (new; V100's test used single months).")
                 + "\nThe 1-month row is the plain month bootstrap with "
                   "different random draws from table 3's.")

    # ================================================================ TABLE 4
    rows, got["t4"] = [], {}
    for name in (LIQ, ALL, CORE):
        uu = unis[name]
        Tt = uu.T
        pm, dm, nm_ = Tt.mean(uu.x["sm"], uu.F[MODEL], test)
        pr, dr, nr = Tt.mean(uu.x["sm"], uu.F[RULE], test)
        dp, dd, na, nb = Tt.diff(uu.x["sm"], uu.F[MODEL], uu.x["sm"],
                                 uu.F[RULE], test)
        row = {"Universe": name, "Model signals": f"{nm_:,}",
               "Model lift, pp [90%]": ci_txt(pm, dm, nm_, dec),
               "Rule signals": f"{nr:,}",
               "Rule lift, pp [90%]": ci_txt(pr, dr, nr, dec),
               "Model minus rule, pp [90%]": ci_txt(dp, dd, min(na, nb), dec),
               "P(model - rule <= 0)": f"{p_le0(dd):.3f}"
               if min(na, nb) >= MIN_N else "-"}
        got["t4"][(name, MODEL)] = (dp, dd)
        if V81 in uu.F:
            vp, vd, va, vb = Tt.diff(uu.x["sm"], uu.F[V81], uu.x["sm"],
                                     uu.F[RULE], test)
            row["V81 minus rule, pp [90%]"] = ci_txt(vp, vd, min(va, vb), dec)
            got["t4"][(name, V81)] = (vp, vd)
        rows.append(row)
    R.table(f"{key}_t04_three_universes",
            f"{h} 4. THREE UNIVERSES - model against the rule, {span(test)}",
            pd.DataFrame(rows),
            note="Unseen liquid = the pre-registered universe. All unseen "
                 "stocks include thin ones; the core stocks are\nthe ones "
                 "the models were trained on (today's large companies, so "
                 "they flatter absolute levels).")

    # ================================================================ TABLE 5
    rows = []
    for name in (LIQ, CORE):
        uu = unis[name]
        D = uu.D
        yf = uu.yr(full)
        for k_, label in ((MODEL, "model"), (RULE, rule)):
            f = uu.F[k_]
            mm = f & yf
            n = int(mm.sum())
            row = {"Universe": name, "Arm": label, "Signals": f"{n:,}",
                   "ENTER hit %": pct(uu.o["tgt"][mm].mean() * 100)
                   if n else "-",
                   "Same stock-month %": pct(uu.pool["tgt"][mm].mean() * 100)
                   if n else "-",
                   "Same month, all stocks %":
                       pct(uu.base["tgt"][mm].mean() * 100) if n else "-"}
            for lab, xk in (("Timing", "sm"), ("Selection", "sel"),
                            ("Total", "avg")):
                p, dr, n_ = D.mean(uu.x[xk], f, full)
                row[f"{lab}, pp [90%]"] = ci_txt(p, dr, n_)
                if name == LIQ and k_ == MODEL:
                    got.setdefault("tvs", {})[lab] = ci_txt(p, dr, n_)
            for lab, xk in (("Timing R", "smR"), ("Selection R", "selR"),
                            ("Total R", "avgR")):
                row[lab] = sg(D.mean(uu.x[xk], f, full)[0])
                if name == LIQ and k_ == MODEL:
                    got["tvs"][lab] = row[lab]
            rows.append(row)
    R.table(f"{key}_t05_timing_vs_selection",
            f"{h} 5. TIMING vs SELECTION - target hit rate, {span(full)}",
            pd.DataFrame(rows),
            note="ENTER minus the month's average entry (all stocks) = TOTAL"
                 "\n  = TIMING     ENTER day vs the other days of the same "
                 "stock and month (the claim)\n  + SELECTION  that stock-"
                 "month vs the month's average entry.\nA negative SELECTION "
                 "means the model fires in stock-months that do worse than "
                 "average (falling\nstocks): it times stocks chosen by "
                 "something else; it does not pick them.")

    # ================================================================ TABLE 6
    rows = []
    for name in (LIQ, CORE, ALL):
        uu = unis[name]
        periods = [full, h1, h2] if name != ALL else [full]
        for y_ in periods:
            rows.append(trade_row(f"{name}, {span(y_)}", uu, uu.F[MODEL], y_))
    rows.append(trade_row(f"{LIQ}, {span(full)}: rule '{rule}'", unis[LIQ],
                          unis[LIQ].F[RULE], full))
    T6 = pd.DataFrame(rows)
    got["t6"] = T6
    uu = unis[LIQ]
    mm = uu.F[MODEL] & uu.yr(full)
    stopped = mm & (uu.o["stp"] > 0)
    gapped = stopped & (uu.walk["r_real"] < -1 - 1e-9)
    cost_r = np.nanmean(COST / (uu.walk["stop_pct"][mm] / 100))
    R.table(f"{key}_t06_backtest_every_signal",
            f"{h} 6. BACKTEST OF EVERY ENTER SIGNAL - walk-forward, "
            f"{span(full)}", T6,
            note=f"Every ENTER signal traded once, nothing skipped: entry at "
                 f"the signal day's close, the label's stop and target.\n"
                 f"Avg R = result per trade in units of the stop distance, "
                 f"as labelled. 'Gaps + costs' = re-walked from the price\n"
                 f"files: a day that OPENS beyond the stop or target exits "
                 f"at that open, minus {COST:.2%} round-trip costs.\n"
                 f"Unseen liquid: {gapped.sum() / max(stopped.sum(), 1):.0%} "
                 f"of the stopped trades opened beyond the stop; costs take "
                 f"about {cost_r:.3f}R a trade.\nRe-walk check: the label's "
                 f"R reproduced on {lab_ok:,} of {lab_n:,} trades "
                 f"({lab_ok / max(lab_n, 1):.1%}).")

    # ================================================================ TABLE 7
    uu = unis[LIQ]
    D = uu.D
    f, fr = uu.F[MODEL], uu.F[RULE]
    yf = uu.yr(full)
    mm, mr = f & yf, fr & yf
    n = int(mm.sum())
    rows = [{"Group": "ENTER days (model)", "Rows": f"{n:,}",
             **levels(uu.o, mm), "ENTER minus this, pp [90%]": "",
             "ENTER minus this, R [90%]": ""}]
    got["t7"] = {}
    for label, ref, xk, rk, tag in (
            ("Other days, same stock and month", uu.pool, "sm", "smR", "sm"),
            ("Average entry, same month (all stocks)", uu.base, "avg",
             "avgR", "avg")):
        p, dr, k_ = D.mean(uu.x[xk], f, full)
        pr_, drr, _ = D.mean(uu.x[rk], f, full)
        got["t7"][tag] = (p, dr, k_)
        rows.append({"Group": label, "Rows": f"{n:,} (matched)",
                     **levels(ref, mm),
                     "ENTER minus this, pp [90%]": ci_txt(p, dr, k_),
                     "ENTER minus this, R [90%]": ci_txt(pr_, drr, k_, 3)})
    rows.append({"Group": "Any candidate day (all stocks)",
                 "Rows": f"{int(yf.sum()):,}", **levels(uu.o, yf),
                 "ENTER minus this, pp [90%]": sg(
                     (uu.o["tgt"][mm].mean() - uu.o["tgt"][yf].mean()) * 100,
                     1) if n else "-",
                 "ENTER minus this, R [90%]": sg(
                     np.nanmean(uu.o["R"][mm]) - np.nanmean(uu.o["R"][yf]))
                 if n else "-"})
    dp, dd, na, nb = D.diff(uu.x["tgt"], f, uu.x["tgt"], fr, full)
    dpr, ddr, _, _ = D.diff(uu.o["R"], f, uu.o["R"], fr, full)
    rows.append({"Group": f"ENTER days of the rule ({rule})",
                 "Rows": f"{int(mr.sum()):,}", **levels(uu.o, mr),
                 "ENTER minus this, pp [90%]": ci_txt(dp, dd, min(na, nb)),
                 "ENTER minus this, R [90%]": ci_txt(dpr, ddr, min(na, nb),
                                                     3)})
    R.table(f"{key}_t07_against_alternatives",
            f"{h} 7. ENTER DAYS AGAINST THE ALTERNATIVES - {LIQ}, "
            f"{span(full)}", pd.DataFrame(rows),
            note="The matched rows show what the SAME signals' alternatives "
                 "did: any other day of the same stock and\nmonth (the "
                 "timing claim), and an average entry that month in any "
                 "stock (timing + selection).\n'Any candidate day' is the "
                 "base rate, not matched (no interval). The rule's row is "
                 "paired by month.")

    # ================================================================ TABLE 8
    uu = unis[LIQ]
    D = uu.D
    yf = uu.yr(full)
    br = uu.br
    groups = [(f"bull (breadth > {BULL:.2f})", br > BULL),
              ("neutral", (br >= BEAR) & (br <= BULL)),
              (f"bear (breadth < {BEAR:.2f})", br < BEAR)]
    rows, got["t8"] = [], {}
    for label, gm in groups:
        f, fr = uu.F[MODEL] & gm, uu.F[RULE] & gm
        mm = f & yf
        n = int(mm.sum())
        p, dr, k_ = D.mean(uu.x["sm"], f, full)
        prl, _, kr = D.mean(uu.x["sm"], fr, full)
        dp, dd, na, nb = D.diff(uu.x["sm"], f, uu.x["sm"], fr, full)
        t_, s_ = uu.o["tgt"][mm].sum(), uu.o["stp"][mm].sum()
        lift = ci_txt(p, dr, k_)
        got["t8"][label] = {"n": n, "lift": lift, "p": p, "dr": dr,
                            "minus": ci_txt(dp, dd, min(na, nb)),
                            "hit": pct(t_ / n * 100) if n else "-"}
        rows.append({"Regime": label,
                     "Share of candidate days":
                         f"{(gm & yf).sum() / max(yf.sum(), 1):.0%}",
                     "Signals": f"{n:,}",
                     "Hit target %": pct(t_ / n * 100) if n else "-",
                     "Resolved hit %": pct(t_ / (t_ + s_) * 100)
                     if t_ + s_ else "-",
                     "Avg R, gaps + costs": sg(np.nanmean(
                         uu.walk["real"][mm])) if n else "-",
                     "Lift, pp [90%]": lift,
                     "Rule signals": f"{kr:,}",
                     "Rule lift, pp": sg(prl, 1),
                     "Model minus rule, pp [90%]":
                         ci_txt(dp, dd, min(na, nb))})
    miss = int((uu.F[MODEL] & yf & ~np.isfinite(br)).sum())
    R.table(f"{key}_t08_regimes",
            f"{h} 8. MARKET REGIMES AT ENTRY - {LIQ}, {span(full)}",
            pd.DataFrame(rows),
            note="Regime = share of the core stocks closing above their own "
                 "200-day EMA on the signal day. Lift = against\nthe other "
                 "days of the same stock and month.\n"
                 + (f"{miss:,} signals have no breadth value (fewer than 50 "
                    f"core stocks with 200 days of history) and are in no "
                    f"row." if miss else ""))

    # ================================================================ TABLE 9
    uu, uc = unis[LIQ], unis[CORE]
    rows = []
    for y in full:
        f, fr = uu.F[MODEL] & (uu.year == y), uu.F[RULE] & (uu.year == y)
        fc = uc.F[MODEL] & (uc.year == y)
        n, nr, nc = int(f.sum()), int(fr.sum()), int(fc.sum())
        t_, s_ = uu.o["tgt"][f].sum(), uu.o["stp"][f].sum()
        lift = uu.x["sm"][f].mean() if n else np.nan
        rlift = uu.x["sm"][fr].mean() if nr else np.nan
        rows.append({"Year": str(y), "Signals": f"{n:,}",
                     "Hit target %": pct(t_ / n * 100) if n else "-",
                     "Resolved hit %": pct(t_ / (t_ + s_) * 100)
                     if t_ + s_ else "-",
                     "Avg R": sg(np.nanmean(uu.o["R"][f])) if n else "-",
                     "Avg R, gaps + costs": sg(np.nanmean(uu.walk["real"][f]))
                     if n else "-",
                     "Same stock-month hit %": pct(uu.pool["tgt"][f].mean()
                                                   * 100) if n else "-",
                     "Lift, pp": sg(lift, 1), "Rule signals": f"{nr:,}",
                     "Rule lift, pp": sg(rlift, 1),
                     "Model minus rule, pp": sg(lift - rlift, 1),
                     "Core signals": f"{nc:,}",
                     "Core lift, pp": sg(uc.x["sm"][fc].mean(), 1)
                     if nc else "-",
                     "_n": n, "_l": lift, "_d": lift - rlift})
    T9 = pd.DataFrame(rows)
    ok = T9[T9["_n"] >= MIN_N]
    note9 = (f"Estimates only (a year is a short sample). In the "
             f"{len(ok)} years with {MIN_N}+ signals: lift above zero in "
             f"{int((ok['_l'] > 0).sum())}, model ahead of the rule in "
             f"{int((ok['_d'] > 0).sum())}.")
    R.table(f"{key}_t09_by_year", f"{h} 9. BY YEAR - {LIQ} (core lift for "
            f"comparison)", T9.drop(columns=["_n", "_l", "_d"]), note=note9)

    # =============================================================== TABLE 10
    print(f"  [{h}] random backtest: {n_runs} runs x {len(N_PICKS)} sizes x "
          f"2 universes ...")
    rows, got["t10"] = [], {}
    for name in (LIQ, CORE):
        uu = unis[name]
        for n_ in N_PICKS:
            Dr = V95.repeat_runs(uu.U, uu.F[MODEL], full, n_, n_runs,
                                 PICK_SEED)
            s = V95.spread_row(name, n_, Dr)
            got["t10"][(name, n_)] = s
            rows.append({
                "Universe": name, "Picks per run": f"{n_:,}",
                "Trades per run": f"{s['ENTER trades per run']:.0f}",
                "Hit % (avg)": pct(s["Hit % avg"]),
                "Hit %, middle 90% of runs":
                    f"{pct(s['Hit % lo'])} - {pct(s['Hit % hi'])}",
                "Resolved hit % (avg)": pct(s["Resolved % avg"]),
                "Resolved %, middle 90% of runs":
                    f"{pct(s['Resolved % lo'])} - {pct(s['Resolved % hi'])}",
                f"Runs at {V95.GOAL:.0f}%+ resolved":
                    f"{s['Resolved runs >= goal %']:.0f}%",
                "Other day, same stock-month, hit %":
                    pct(s["Other day hit %"]),
                "Every pick hit %": pct(s["Every pick hit %"]),
                "Runs where ENTER beat the other day":
                    f"{s['Runs ENTER beat other day %']:.0f}%"})
    R.table(f"{key}_t10_random_backtest",
            f"{h} 10. THE RANDOM BACKTEST - {n_runs} runs per row, "
            f"{span(full)}", pd.DataFrame(rows),
            note=f"V91's test, repeated: draw random (stock, day) picks, "
                 f"trade the ones the model calls ENTER, compare with a\n"
                 f"random OTHER day of the same stock and month. One run of "
                 f"3,000 picks is about 30 trades, so its hit rate is\n"
                 f"mostly luck; the 30,000-pick rows show where it settles. "
                 f"Seed {PICK_SEED} (V95's). Break-even, resolved trades: "
                 f"{BREAK_EVEN:.1f}%.")

    # =============================================================== TABLE 11
    rels, stats = [], []
    for name in (CORE, LIQ):
        uu = unis[name]
        rel, st = M88.calibration_check(uu.U, *uu.w["combo"], test, name)
        rels.append(rel)
        stats.append(st)
    rel = pd.concat(rels, ignore_index=True)
    got["t11"] = (rel, stats)
    T11 = pd.DataFrame({"Universe": rel["Universe"], "Bucket": rel["Bucket"],
                        "Rows": rel["Rows"].map(lambda v: f"{int(v):,}"),
                        "Mean probability": rel["Mean p"].map(
                            lambda v: pct(v, 3)),
                        "Hit rate": rel["Win rate"].map(lambda v: pct(v, 3))})
    show = (model or {}).get("display_probability")
    summ = "; ".join(f"{s['universe']}: Brier skill {s['skill']:+.4f}, fired "
                     f"signals' mean probability minus hit rate "
                     f"{s['fired_gap']:+.3f} on {s['fired_n']:,}"
                     for s in stats)
    R.table(f"{key}_t11_calibration",
            f"{h} 11. CALIBRATION - out-of-sample fold probabilities, "
            f"{span(test)}", T11,
            note=f"By quintile of the score, and the ENTER signals. "
                 f"{summ}.\nThe app may show the probability only if the "
                 f"fired signals' gap is within {M88.CALIB_TOL:.0%} in BOTH "
                 f"universes. Decision: "
                 + ("(model file not found)" if model is None else
                    "show it" if show else "do NOT show it - ENTER / WAIT "
                                           "and the measured hit rate")
                 + ".\nThe model ranks days within a stock and month; its "
                   "probability is not a hit rate across stocks.")

    # ========================================================== REPRODUCTION
    reproduce(h, chk, got, unis, rule, CORE, model, model_file, dirs)
    T, line = chk.summary()
    if len(T):
        R.table(f"{key}_t12_reproduction_check",
                f"{h} - REPRODUCTION CHECK: recomputed here vs the files of "
                f"the runs that first reported them", T,
                note=line + f" (every number is in {key}_t12_reproduction_"
                            f"check.csv)", show=chk.brief(T))
    print(f"\n  [{h}] {line}  |  section {(time.time() - t0) / 60:.1f} min")

    # ---- for the comparison ------------------------------------------------
    uu = unis[LIQ]
    yf = uu.yr(full)
    mm = uu.F[MODEL] & yf
    r6 = T6.iloc[0].to_dict()
    sm, av = got["t7"]["sm"], got["t7"]["avg"]
    summary = {
        "h": h, "final": P["final"], "trade": P["trade"], "hold": g["hold"],
        "test": test, "full": full, "rule": rule, "interval": P["interval"],
        "t3": {"n": m3["n"], "lift": ci_txt(m3["p"], m3["dr"], m3["n"]),
               "timing": timing, "minus": rule_txt, "rule_v": rule_v},
        "every": {**r6, "years": span(full),
                  "per_year": mm.sum() / max(len(full), 1),
                  "sm": ci_txt(*sm), "avg": ci_txt(*av)},
        "tvs": got.get("tvs", {}),
        "t8": got["t8"],
        "t10": got["t10"].get((LIQ, N_PICKS[-1])),
        "t11": {"stats": stats, "show": show},
        "check": line}
    del unis, got
    gc.collect()
    return summary


# =============================================================================
# 7. REPRODUCTION - each model against its own recorded files
# =============================================================================
def reproduce(h, chk, got, unis, rule, CORE, model, model_file, dirs):
    P = PROFILES[h]
    full = P["full"]

    def lab(d, f):                        # 'thesis_tables_v100/primary.csv'
        return f"{os.path.basename(os.path.normpath(d))}/{f}"
    if h == "MID":
        src = os.path.join(dirs["v86"], "side_by_side.csv")
        sb = read_csv(src)
        if sb is None:
            chk.missing(src)
        else:
            for vname, name in (("fresh, liquid (PRIMARY)", LIQ),
                                ("fresh, all", ALL), (CORE, CORE)):
                for arm, k in (("V81", V81), (V86.COMBO_NAME, MODEL)):
                    r = sb[(sb["Universe"] == vname) & (sb["Arm"] == arm)]
                    if not len(r) or (name, k) not in got["t4"]:
                        continue
                    r = r.iloc[0]
                    dp, dd = got["t4"][(name, k)]
                    lo, hi = bounds(dd)
                    tag = f"V86 {name}: {'V81' if k == V81 else 'V88'} - rule"
                    f86 = lab(dirs["v86"], "side_by_side.csv")
                    chk.num(tag, dp, r["vs bb"], f86)
                    chk.num(tag + " 90% lo", lo, r["lo"], f86)
                    chk.num(tag + " 90% hi", hi, r["hi"], f86)
                    if k == MODEL and name == LIQ:
                        chk.num(tag + " two-test bound",
                                float(np.nanpercentile(dd, 2.5)),
                                r["lo (Bonf k=2)"], f86)
        src = os.path.join(dirs["v87"], "block_bootstrap.csv")
        bb = read_csv(src)
        if bb is None:
            chk.missing(src)
        else:
            for L, (p, dr, dp, dd) in got["t3c"].items():
                r = bb[(bb["Universe"] == "fresh, liquid (PRIMARY)")
                       & (bb["Arm"] == V86.COMBO_NAME)
                       & (bb["Years"].astype(str) == span(P["test"]))
                       & (bb["Block (months)"] == L)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lo, hi = bounds(dd)
                chk.num(f"V87 {L}-month blocks: V88 - rule", dp, r["Diff"],
                        lab(dirs["v87"], "block_bootstrap.csv"))
                chk.num(f"V87 {L}-month blocks: 90% lo", lo, r["90% lo"],
                        lab(dirs["v87"], "block_bootstrap.csv"))
                chk.num(f"V87 {L}-month blocks: 90% hi", hi, r["90% hi"],
                        lab(dirs["v87"], "block_bootstrap.csv"))
        src = os.path.join(dirs["v95"], "every_signal.csv")
        es = read_csv(src)
        if es is None:
            chk.missing(src)
        else:
            for name in (LIQ, CORE):
                r = es[(es["Universe"] == name)
                       & (es["Years"].astype(str) == span(full))]
                if not len(r):
                    continue
                r = r.iloc[0]
                uu = unis[name]
                D = uu.D
                f = uu.F[MODEL]
                mm = f & uu.yr(full)
                t_, s_ = uu.o["tgt"][mm].sum(), uu.o["stp"][mm].sum()
                chk.num(f"V95 {name}: signals", float(mm.sum()), r["Signals"],
                        lab(dirs["v95"], "every_signal.csv"))
                chk.num(f"V95 {name}: hit %", uu.o["tgt"][mm].mean() * 100,
                        r["Hit %"], lab(dirs["v95"], "every_signal.csv"))
                chk.num(f"V95 {name}: resolved hit %", t_ / (t_ + s_) * 100,
                        r["Resolved hit %"],
                        lab(dirs["v95"], "every_signal.csv"))
                chk.num(f"V95 {name}: avg R", np.nanmean(uu.o["R"][mm]),
                        r["Avg R"], lab(dirs["v95"], "every_signal.csv"))
                for xk, tag in (("sm", "vs same stock-month"),
                                ("avg", "vs average entry")):
                    p, dr, _ = D.mean(uu.x[xk], f, full)
                    lo, hi = bounds(dr)
                    chk.num(f"V95 {name}: {tag}", p, r[f"{tag} pp"],
                            lab(dirs["v95"], "every_signal.csv"))
                    chk.num(f"V95 {name}: {tag} lo", lo, r[f"{tag} lo"],
                            lab(dirs["v95"], "every_signal.csv"))
                    chk.num(f"V95 {name}: {tag} hi", hi, r[f"{tag} hi"],
                            lab(dirs["v95"], "every_signal.csv"))
        src = os.path.join(dirs["v95"], "repeated_runs.csv")
        rr = read_csv(src)
        if rr is None:
            chk.missing(src)
        else:
            for (name, n_), s in got["t10"].items():
                r = rr[(rr["Universe"] == name) & (rr["Picks per run"] == n_)]
                if not len(r):
                    continue
                r = r.iloc[0]
                for col in ("ENTER trades per run", "Hit % avg",
                            "Resolved % avg"):
                    chk.num(f"V95 {name}, {n_:,} picks: {col}", s[col],
                            r[col], lab(dirs["v95"], "repeated_runs.csv"))
    else:
        d_ = dirs["v100" if h == "SHORT" else "v102"]
        tag_v = "V100" if h == "SHORT" else "V102"
        src = os.path.join(d_, "primary.csv")
        pr = read_csv(src, as_text=True)
        t3 = got["t3"]
        if pr is None:
            chk.missing(src)
        else:
            for k, arm in ((MODEL, "model"), (RULE, rule),
                           (RANDOM, "random 1% (yardstick)")):
                r = pr[pr["Arm"] == arm]
                if not len(r):
                    chk.txt(f"{tag_v} primary: arm '{arm}'", "present",
                            "absent", lab(d_, "primary.csv"))
                    continue
                r = r.iloc[0]
                a = t3[k]
                chk.txt(f"{tag_v} primary: {arm} signals", f"{a['n']:,}",
                        r["Signals"], lab(d_, "primary.csv"))
                if h == "SHORT":
                    chk.txt(f"{tag_v} primary: {arm} lift",
                            ci_txt(a["p"], a["dr"], None, 1),
                            r["Lift over same stock-month, pp [90%]"],
                            lab(d_, "primary.csv"))
                    chk.txt(f"{tag_v} primary: {arm} excess R", sg(a["er"]),
                            r["Excess R"], lab(d_, "primary.csv"))
                else:
                    chk.txt(f"{tag_v} primary: {arm} lift (12-month blocks)",
                            ci_txt(a["p"], a["dr"], None, 1),
                            r["Lift, pp [90%, 12-month blocks]"],
                            lab(d_, "primary.csv"))
                    p, dr, _ = got["t3ref"][k]
                    chk.txt(f"{tag_v} primary: {arm} lift (month intervals)",
                            ci_txt(p, dr, None, 1),
                            r["same, month intervals (reference)"],
                            lab(d_, "primary.csv"))
        vt = read_text(os.path.join(d_, "verdict.txt"))
        if vt:
            line = [x for x in vt.split("\n") if "2. RULE" in x]
            if line:
                want = line[0].split(": ")[-1].split(" pp")[0].strip()
                chk.txt(f"{tag_v} verdict: model minus rule",
                        ci_txt(t3[MODEL]["dp"], t3[MODEL]["dd"], None, 1),
                        want, lab(d_, "verdict.txt"))
        src = os.path.join(d_, "opponent_choice.csv")
        oc = read_csv(src)
        if oc is None:
            chk.missing(src)
        else:
            ok = oc["Lift pp"].notna()
            want = (oc.loc[oc.loc[ok, "Lift pp"].idxmax(), "Rule"] if ok.any()
                    else None)
            chk.txt(f"{tag_v} rule to beat", got["chosen"], want,
                    lab(d_, "opponent_choice.csv"))
            for _, r in oc.iterrows():
                if r["Rule"] in got["choose"] and pd.notna(r["Lift pp"]):
                    chk.num(f"{tag_v} rule choice: {r['Rule']}",
                            got["choose"][r["Rule"]], r["Lift pp"],
                            lab(d_, "opponent_choice.csv"))
        src = os.path.join(d_, "trades.csv")
        tr = read_csv(src, as_text=True)
        if tr is None:
            chk.missing(src)
        else:
            first = tr.columns[0]
            mine = got["t6"].set_index("")
            for name in (LIQ, CORE):
                key_ = f"{name}, {span(full)}"
                r = tr[tr[first] == key_]
                if not len(r) or key_ not in mine.index:
                    continue
                r, m_ = r.iloc[0], mine.loc[key_]
                for col in ("Signals", "Hit target %", "Stop %", "Expired %",
                            "Resolved hit %", "Avg R", "Avg R, gaps + costs"):
                    chk.txt(f"{tag_v} trades, {name}: {col}", m_[col], r[col],
                            lab(d_, "trades.csv"))
                dh = float(m_["Days held"])
                chk.num(f"{tag_v} trades, {name}: days held", dh,
                        float(r["Days held"]), lab(d_, "trades.csv"),
                        tol=0.051 if h == "SHORT" else 0.51)
                uu = unis[name]
                for xk, col in (("sm", "vs same stock-month, pp [90%]"),
                                ("avg", "vs average entry, pp [90%]")):
                    p, dr, n_ = uu.D.mean(uu.x[xk], uu.F[MODEL], full)
                    chk.txt(f"{tag_v} trades, {name}: {col.split(',')[0]}",
                            ci_txt(p, dr, None, 1), r[col],
                            lab(d_, "trades.csv"))
        src = os.path.join(d_, "regimes.csv")
        rg = read_csv(src, as_text=True)
        if rg is None:
            chk.missing(src)
        elif "Lift, pp [90%]" in rg.columns:
            for _, r in rg.iterrows():
                v = got["t8"].get(r["Group"])
                if v and v["n"] >= MIN_N and r["Lift, pp [90%]"]:
                    chk.txt(f"{tag_v} regimes: {r['Group']}", v["lift"],
                            r["Lift, pp [90%]"], lab(d_, "regimes.csv"))
    # calibration stored in the final model file
    if model and model.get("calibration", {}).get("table"):
        rel, _ = got["t11"]
        st = pd.DataFrame(model["calibration"]["table"])
        a = rel[rel["Bucket"] == "FIRED (top 1%)"].reset_index(drop=True)
        b = st[st["Bucket"] == "FIRED (top 1%)"].reset_index(drop=True)
        for i in range(min(len(a), len(b))):
            nm = a.loc[i, "Universe"]
            chk.num(f"model file calibration, {nm}: fired rows",
                    float(a.loc[i, "Rows"]), b.loc[i, "Rows"],
                    os.path.basename(model_file))
            chk.num(f"model file calibration, {nm}: fired hit rate",
                    float(a.loc[i, "Win rate"]), b.loc[i, "Win rate"],
                    os.path.basename(model_file))


# =============================================================================
# 8. THE THREE MODELS SIDE BY SIDE
# =============================================================================
def comparison(R, S):
    R.heading("COMPARISON - the three entry models side by side",
              "Unseen liquid stocks throughout. MID's pre-registered years "
              "are 2017-2026; its descriptive rows use 2008-2026\nlike "
              "SHORT's. LONG ends in 2025 (a label needs a year of future "
              "prices).")
    rows = []
    for s in S:
        t = s["t3"]
        rows.append({"Model": f"{s['h']} ({s['final']})",
                     "Trade": f"{s['hold']} days",
                     "Test years": span(s["test"]),
                     "ENTER signals": f"{t['n']:,}",
                     "Timing lift, pp [90%]": t["lift"],
                     "Timing": t["timing"], "Rule to beat": s["rule"],
                     "Model minus rule, pp [90%]": t["minus"],
                     "Against the rule": t["rule_v"]})
    R.table("cmp_c1_preregistered_tests", "C1. THE THREE PRE-REGISTERED "
            "TESTS", pd.DataFrame(rows),
            note="Timing lift = ENTER days minus the other days of the same "
                 "stock and month (target hit rate, pp).\nIntervals: month "
                 "bootstrap (MID, SHORT); 12-month blocks (LONG). MID's rule "
                 "test is the second of two\n(two-test bound shown).")

    rows = []
    for s in S:
        e = s["every"]
        rows.append({"Model": f"{s['h']} ({s['final']})",
                     "Years": e["years"], "Signals": e["Signals"],
                     "Per year": f"{e['per_year']:.0f}",
                     "Median target %": e.get("Median target %", "-"),
                     "Median stop %": e.get("Median stop %", "-"),
                     "Days held": e.get("Days held", "-"),
                     "Hit target %": e.get("Hit target %", "-"),
                     "Resolved hit %": e.get("Resolved hit %", "-"),
                     "Avg R": e.get("Avg R", "-"),
                     "Avg R, gaps + costs": e.get("Avg R, gaps + costs", "-"),
                     "vs same stock-month, pp [90%]": e["sm"],
                     "vs average entry, pp [90%]": e["avg"]})
    R.table("cmp_c2_backtest", "C2. BACKTEST OF EVERY ENTER SIGNAL - unseen "
            "liquid stocks", pd.DataFrame(rows),
            note=f"Break-even: {BREAK_EVEN:.1f}% resolved (target 1.67 x the "
                 f"stop in every model). R = result per trade in stop "
                 f"distances.")

    rows = []
    for s in S:
        v = s["tvs"]
        rows.append({"Model": f"{s['h']} ({s['final']})",
                     "Years": span(s["full"]),
                     "Timing, pp [90%]": v.get("Timing", "-"),
                     "Selection, pp [90%]": v.get("Selection", "-"),
                     "Total, pp [90%]": v.get("Total", "-"),
                     "Timing R": v.get("Timing R", "-"),
                     "Selection R": v.get("Selection R", "-"),
                     "Total R": v.get("Total R", "-")})
    R.table("cmp_c3_timing_vs_selection", "C3. TIMING vs SELECTION - unseen "
            "liquid stocks, the model's signals", pd.DataFrame(rows),
            note="Total (vs the month's average entry) = timing (vs the same "
                 "stock's other days) + selection (that stock-month).")

    rows = []
    for s in S:
        for reg, v in s["t8"].items():
            rows.append({"Model": f"{s['h']} ({s['final']})",
                         "Regime": reg, "Signals": f"{v['n']:,}",
                         "Hit target %": v["hit"],
                         "Lift, pp [90%]": v["lift"],
                         "Model minus rule, pp [90%]": v["minus"]})
    R.table("cmp_c4_regimes", "C4. MARKET REGIMES - unseen liquid stocks, "
            "descriptive years", pd.DataFrame(rows),
            note="Bull: more than 60% of the core stocks above their 200-day "
                 "EMA; bear: fewer than 40%.")

    rows = []
    for s in S:
        r = s["t10"]
        if r is None:
            continue
        rows.append({"Model": f"{s['h']} ({s['final']})",
                     "Picks per run": f"{N_PICKS[-1]:,}",
                     "Trades per run": f"{r['ENTER trades per run']:.0f}",
                     "Hit % (avg)": pct(r["Hit % avg"]),
                     "Hit %, middle 90%": f"{pct(r['Hit % lo'])} - "
                                          f"{pct(r['Hit % hi'])}",
                     "Resolved % (avg)": pct(r["Resolved % avg"]),
                     "Resolved %, middle 90%": f"{pct(r['Resolved % lo'])} - "
                                               f"{pct(r['Resolved % hi'])}",
                     f"Runs at {V95.GOAL:.0f}%+ resolved":
                         f"{r['Resolved runs >= goal %']:.0f}%",
                     "Other day hit %": pct(r["Other day hit %"]),
                     "Runs ENTER beat the other day":
                         f"{r['Runs ENTER beat other day %']:.0f}%"})
    R.table("cmp_c5_random_backtest", "C5. THE RANDOM BACKTEST - unseen "
            f"liquid stocks, {N_PICKS[-1]:,} picks per run",
            pd.DataFrame(rows))

    rows = []
    for s in S:
        st = s["t11"]["stats"]
        un = next((x for x in st if x["universe"] == LIQ), {})
        co = next((x for x in st if x["universe"].startswith("core")), {})
        rows.append({"Model": f"{s['h']} ({s['final']})",
                     "Years": span(s["test"]),
                     "Brier skill, unseen": sg(un.get("skill", np.nan), 4),
                     "Fired: mean p - hit rate, unseen":
                         sg(un.get("fired_gap", np.nan)),
                     "Brier skill, core": sg(co.get("skill", np.nan), 4),
                     "Fired: mean p - hit rate, core":
                         sg(co.get("fired_gap", np.nan)),
                     "Probability shown": ("-" if s["t11"]["show"] is None
                                           else "yes" if s["t11"]["show"]
                                           else "no")})
    R.table("cmp_c6_calibration", "C6. CALIBRATION - why the app shows a "
            "signal, not a probability", pd.DataFrame(rows),
            note=f"Shown only if |fired mean p - hit rate| <= "
                 f"{M88.CALIB_TOL:.0%} in both universes.")

    R.text("REPRODUCTION, all sections", "\n".join(
        f"{s['h']:<6} {s['check']}" for s in S))


# =============================================================================
# RUNNER
# =============================================================================
def _per(v, h):
    return v.get(h) if isinstance(v, dict) else v


def run_v104(horizons=("MID", "SHORT", "LONG"), core_cache=None,
             big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
             n_runs=200, out_dir=OUT_DIR, model_in81="entry_model_v81.joblib",
             paths=None, verbose=False):
    """core_cache, big_cache, n_seeds, chunk and n_boot may also be dicts
    keyed by horizon. paths: {horizon: {"model_file": ..., "v86": ...}}
    overrides the default file and folder names."""
    t0 = time.time()
    R = Report(out_dir)
    print("=" * W)
    print("V104 - THE PAPER TABLES: the three entry models (" +
          ", ".join(horizons) + ")")
    print("=" * W)
    S = []
    for h in horizons:
        if h not in PROFILES:
            raise SystemExit(f"unknown horizon {h!r}: use MID, SHORT, LONG")
        P = PROFILES[h]
        dirs = {**P["dirs"], **(paths or {}).get(h, {})}
        mfile = (paths or {}).get(h, {}).get("model_file", P["model_file"])
        model = joblib.load(mfile) if os.path.exists(mfile) else None
        print(f"\n  [{h}] reading its data and walk-forward predictions "
              f"(from the caches; built once if missing) ...")
        data = load_data(h, _per(core_cache, h), _per(big_cache, h),
                         _per(n_seeds, h), _per(chunk, h), model_in81, model,
                         verbose)
        S.append(model_section(R, h, data, model, mfile, _per(n_boot, h),
                               n_runs, dirs, _per(n_seeds, h), verbose))
        del data, model
        gc.collect()
    if len(S) > 1:
        comparison(R, S)
    R.save("# V104 - paper tables for the entry-timing models (Pillar 2)\n\n"
           f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by "
           f"report_results_v104.py. Sections: " + ", ".join(horizons)
           + (", comparison" if len(S) > 1 else "") + ".\n\n"
           + "\n".join(f"- {s['h']}: {s['check']}" for s in S) + "\n")
    print(f"  total {(time.time() - t0) / 60:.1f} min")
    return S


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_HORIZONS   = ["MID", "SHORT", "LONG"]   # sections, in this order
    RUN_BIG_CACHE  = "price_cache_v68"
    RUN_CORE_CACHE = None               # None = the cache V81 was trained on
    RUN_N_SEEDS    = 5                  # as V86 / V100 / V102 (cache keys)
    RUN_CHUNK      = 250                # as V86 / V100 / V102 (cache keys)
    RUN_N_BOOT     = 4000               # as V86 / V95 / V100 / V102, so the
                                        # intervals reproduce
    RUN_N_RUNS     = 200                # random backtest runs (as V95)
    RUN_OUT_DIR    = "thesis_tables_v104"
    # -------------------------------------------------------------------------

    run_v104(horizons=RUN_HORIZONS, core_cache=RUN_CORE_CACHE,
             big_cache=RUN_BIG_CACHE, n_seeds=RUN_N_SEEDS, chunk=RUN_CHUNK,
             n_boot=RUN_N_BOOT, n_runs=RUN_N_RUNS, out_dir=RUN_OUT_DIR)
