"""
V89 - EVERY RESULTS TABLE FOR THE FINAL MODEL (V88), IN ONE RUN.

Prints each table, saves it as CSV in thesis_tables_v89/, and writes all of
them into one Markdown file (ALL_TABLES.md) ready to paste into the thesis.
Nothing about the model changes: the data and fold models are read from V86's
caches, and the core numbers are recomputed the same way V86 did, so they must
match V86's tables exactly (checked in [0]).

THE TABLES
----------
   1  Model card - what V88 is, how it was chosen, its parity checks
   2  Headline - the final model in the three universes
   3  The pre-registered out-of-universe test - every arm
   4  Three universes side by side - model against the rule fixed in advance
   5  Selection period vs later years - 2017-21 (chosen) vs 2022-26
   6  Robustness to overlapping trades (V87)
   7  TIMING vs SELECTION - new. A signal's edge over a random candidate in the
      same month splits exactly into two parts:
          timing     fired day vs other days in the SAME stock and month
          selection  the stock-months it fires in vs the month's average
      The claim is about timing. This table shows what the selection part
      does, which matters for how the model may be used.
   8  By market regime - against random days, and against the rule
   9  Per year
  10  How it was chosen (V85, core) and the seed check
  11  Out-of-universe ladder: V81 -> +classify -> +short horizon -> +shallow
  12  Feature ablation on unseen stocks: drop each feature (and each group),
      refit, measure the change. Ablation, not permutation, measures what the
      model depends on (a correction this project already made once).
  13  Calibration of the probabilities (from V88)

COST: tables 1-10 and 13 take a few minutes. Tables 11-12 refit 14 model
variants over 2017-2026 (5 seeds each): about 25-30 minutes the first time,
cached afterwards. Set RUN_ABLATION = False to skip them.
"""

import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81
import measure_v82 as V82
import measure_v83 as V83
import sweep_options_v85 as V85
import out_of_universe_v86 as V86
import block_bootstrap_v87 as V87
import class_ai_entry_model_v88 as M88

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = "thesis_tables_v89"
CACHE_DIR = V86.CACHE_DIR
VERSION = "v89.1"
SEED = V85.SEED
HELD, VAL, TEST = V86.HELD_YEARS, V85.VAL_YEARS, V85.TEST_YEARS
OPP = V86.OPPONENT
FINAL = V86.COMBO_NAME
PRIMARY = "fresh, liquid (PRIMARY)"
NAMES = {FINAL: "Final model (V88)", "V81": "V81 (previous)",
         "blend V81 + bb_position": "blend V81 + bb_position",
         OPP: "low bb_position (rule fixed in advance)"}
BEAR, BULL = V82.BREADTH_CUTS


# =============================================================================
# FORMATTING AND OUTPUT
# =============================================================================
def _f(v, spec="{:+.2f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return spec.format(float(v))
    return str(v)


def ci(lo, hi):
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:+.2f}, {hi:+.2f}]"


PLAIN = {"Year", "Block (months)", "Features", "Rows", "Signals",
         "Stocks", "Fired signals", "Months"}


def display(df, specs=None):
    specs = specs or {}
    out = df.copy()
    for c in out.columns:
        if c in ("Year", "Block (months)", "Features"):
            out[c] = [str(int(v)) if pd.notna(v) else "" for v in out[c]]
        else:
            out[c] = [_f(v, specs.get(c, "{:+.2f}")) for v in out[c]]
    return out


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|")
                                         for v in r.values) + " |")
    return "\n".join(lines)


class Report:
    def __init__(self, out_dir):
        self.out_dir, self.md, self.n = out_dir, [], 0
        os.makedirs(out_dir, exist_ok=True)

    def table(self, key, title, df, note="", specs=None):
        self.n += 1
        df.to_csv(os.path.join(self.out_dir, f"{key}.csv"), index=False)
        d = display(df, specs)
        print("\n" + "=" * 104)
        print(f"  {title}")
        print("=" * 104)
        if note:
            for line in note.strip().split("\n"):
                print(f"  {line}")
            print()
        print(d.to_string(index=False))
        self.md.append(f"### {title}\n\n" + (note.strip() + "\n\n" if note
                                             else "") + md_table(d) + "\n")

    def text(self, title, body):
        print("\n" + "=" * 104 + f"\n  {title}\n" + "=" * 104)
        print(body)
        self.md.append(f"### {title}\n\n```\n{body}\n```\n")

    def save(self, header):
        p = os.path.join(self.out_dir, "ALL_TABLES.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(header + "\n\n" + "\n".join(self.md))
        print(f"\n  wrote {self.n} tables to {self.out_dir}/ and {p}")


# =============================================================================
# DATA
# =============================================================================
def make_universe(frame, mask, wf, tag):
    pos = np.full(len(frame), -1, int)
    pos[mask] = np.arange(int(mask.sum()))
    U = frame[mask].reset_index(drop=True).copy()
    U["row_id"] = np.arange(len(U))
    return U, {k: V86.restrict(*v[tag], pos) for k, v in wf.items()}, pos


def evaluator(U, n_boot, all_years):
    """V86's Evaluator with its weights drawn in V86's order, so every
    interval reproduces V86's tables exactly."""
    ev = V85.Evaluator(U, n_boot=n_boot)
    for yrs in (HELD, VAL, TEST, all_years):
        ev._weights(yrs)
    return ev


# =============================================================================
# 7. TIMING vs SELECTION
# =============================================================================
def month_means(U, col):
    per = pd.PeriodIndex(pd.DatetimeIndex(U["date"]), freq="M")
    return pd.Series(U[col].to_numpy(float)).groupby(
        per.astype(str).to_numpy()).transform("mean").to_numpy()


def decompose(U, ev, fire, years, base_w, base_r):
    """
    fired - month base = (fired - own stock-month pool)   timing
                       + (own pool - month base)          selection
    Every part is a per-signal excess, so each gets a month-bootstrap
    interval with the same resampled months.
    """
    lab = U["label"].to_numpy(float)
    r = U["r_multiple"].to_numpy(float)
    pool_w, pool_r = lab - ev.ex_w, r - ev.ex_r
    idx, W = ev._weights(years)
    m = fire & np.isin(ev.year, years)
    k = len(ev.months)
    cn = np.bincount(ev.mcode[m], minlength=k)[idx].astype(float)
    if cn.sum() < 30:
        return None
    out = {"Signals": int(m.sum()),
           "Fired win %": lab[m].mean() * 100,
           "Stock-month pool %": pool_w[m].mean() * 100,
           "Same-month base %": base_w[m].mean() * 100,
           "Fired R": r[m].mean(), "Pool R": pool_r[m].mean(),
           "Base R": base_r[m].mean()}
    for name, x, scale in (("Timing pp", lab - pool_w, 100),
                           ("Selection pp", pool_w - base_w, 100),
                           ("Total pp", lab - base_w, 100),
                           ("Timing R", r - pool_r, 1),
                           ("Selection R", pool_r - base_r, 1),
                           ("Total R", r - base_r, 1)):
        s = np.bincount(ev.mcode[m], x[m], k)[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (W @ s) / (W @ cn) * scale
        out[name] = s.sum() / cn.sum() * scale
        out[name + " lo"] = np.nanpercentile(b, 5)
        out[name + " hi"] = np.nanpercentile(b, 95)
    return out


# =============================================================================
# 11-12. ABLATION - generic walk-forward over a feature list (clf objective)
# =============================================================================
def build_clf(mono, hp, seed):
    """Exactly V85.build for the classify objective."""
    from xgboost import XGBClassifier
    params = dict(E64.XGB_PARAMS)
    params.update(V85.HPARAMS[hp])
    params["monotone_constraints"] = tuple(int(x) for x in mono)
    return XGBClassifier(random_state=seed, **params)


def wf_custom(feats, mono, hp, core, fresh, years, n_seeds, horizon, sig,
              label, resume=True):
    key = V86._hash(["v89abl", feats, mono, hp, n_seeds, years, SEED, sig,
                     VERSION])
    path = os.path.join(CACHE_DIR, f"abl_{key}.pkl")
    if resume and os.path.exists(path):
        return pd.read_pickle(path), "cached"
    t0 = time.time()
    n_bars = E64.HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    win = V82._win()
    dc, dfr = pd.DatetimeIndex(core["date"]), pd.DatetimeIndex(fresh["date"])
    acc = {"core": ([], {}), "fresh": ([], {})}
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        tr = core[dc < start - embargo]
        te = core[core["year"] == y]
        ref = core[(dc >= start - win) & (dc < start)]
        if len(tr) < 2000 or len(te) < 200:
            continue
        fte = fresh[fresh["year"] == y]
        fref = fresh[(dfr >= start - win) & (dfr < start)]
        sets = [te, ref, fte, fref]
        P = [[] for _ in sets]
        for s in range(n_seeds):
            m = build_clf(mono, hp, SEED + s)
            m.fit(tr[feats], tr["label"], verbose=False)
            for i, X in enumerate(sets):
                P[i].append(m.predict_proba(X[feats])[:, 1] if len(X)
                            else np.array([], float))
        P = [np.mean(p, axis=0) for p in P]
        for tag, a, b, pa, pb in (("core", te, ref, P[0], P[1]),
                                  ("fresh", fte, fref, P[2], P[3])):
            if not len(a):
                continue
            acc[tag][0].append(pd.DataFrame(
                {"row_id": a["row_id"].to_numpy(), "p": pa}))
            acc[tag][1][y] = pd.DataFrame({
                "row_id": np.r_[b["row_id"].to_numpy(), a["row_id"].to_numpy()],
                "date": np.r_[b["date"].to_numpy(), a["date"].to_numpy()],
                "p": np.r_[pb, pa]})
    out = {k: (pd.concat(v[0], ignore_index=True), v[1])
           for k, v in acc.items()}
    pd.to_pickle(out, path)
    return out, f"{time.time() - t0:4.0f}s"


# =============================================================================
# RUNNER
# =============================================================================
def run_v89(model_file="entry_model_v88.joblib",
            model_in81="entry_model_v81.joblib", core_cache=None,
            big_cache="price_cache_v68", n_seeds=5, chunk=250, n_boot=4000,
            run_ablation=True, resume=True, out_dir=OUT_DIR,
            v85_dir="thesis_tables_v85", v86_dir="thesis_tables_v86",
            v87_dir="thesis_tables_v87", verbose=True):
    t0 = time.time()
    R = Report(out_dir)
    print("=" * 104)
    print("V89 - RESULTS TABLES FOR THE FINAL ENTRY MODEL (V88)")
    print("=" * 104)

    # ---- data ------------------------------------------------------------------
    print("\n  reading V86's data and fold models (cached) ...")
    prov81 = M81.load_model(model_in81)["provenance"]
    core_cache = core_cache or prov81["price_cache"]
    core, fresh, wf, n_core = V87.prepare(model_in81, core_cache, big_cache,
                                          n_seeds, chunk, V86.LIQ_PCT,
                                          V86.DUP_CORR, verbose=verbose)
    b, _ = V82.full_breadth(core_cache, V86.cache_names(core_cache))
    core["breadth"] = core["date_n"].map(b).to_numpy(float)
    fresh["breadth"] = fresh["date_n"].map(b).to_numpy(float)
    all_years = sorted(core["year"].unique())[int(prov81["min_train_years"]):]
    CORE = f"core {n_core}"
    unis = {PRIMARY: make_universe(fresh, fresh["liquid"].to_numpy(bool), wf,
                                   "fresh"),
            "fresh, all": make_universe(fresh, np.ones(len(fresh), bool), wf,
                                        "fresh"),
            CORE: make_universe(core, np.ones(len(core), bool), wf, "core")}
    print("  scoring every arm on each universe ...")
    fires, evs, bases = {}, {}, {}
    for name, (U, w, _) in unis.items():
        fires[name] = V86.arms_for(U, w, 3)
        evs[name] = evaluator(U, n_boot, all_years)
        bases[name] = (month_means(U, "label"), month_means(U, "r_multiple"))

    # ---- [0] integrity -----------------------------------------------------------
    print("\n  [0] INTEGRITY - recomputed numbers must equal V86's tables")
    ok, ref_p = True, os.path.join(v86_dir, "side_by_side.csv")
    if os.path.exists(ref_p):
        ref = pd.read_csv(ref_p)
        for name in unis:
            for arm in ("V81", FINAL):
                r = ref[(ref["Universe"] == name) & (ref["Arm"] == arm)]
                p = evs[name].paired(fires[name][arm], fires[name][OPP], HELD,
                                     k_tests=2)
                if not len(r) or p is None:
                    continue
                r = r.iloc[0]
                same = all(abs(a - float(b_)) < 1e-6 for a, b_ in
                           ((p["diff"], r["vs bb"]), (p["lo"], r["lo"]),
                            (p["hi"], r["hi"])))
                ok &= same
                print(f"      {name:<24} {arm:<26} {p['diff']:+.4f} "
                      f"[{p['lo']:+.4f}, {p['hi']:+.4f}]  "
                      f"{'REPRODUCED' if same else '*** DIFFERENT ***'}")
    else:
        print(f"      {ref_p} not found - cannot compare")
        ok = False
    if not ok:
        print("      the numbers below differ from V86's - check RUN_N_BOOT "
              "(must equal V86's), RUN_CHUNK and RUN_N_SEEDS")

    # ---- 1. model card --------------------------------------------------------
    model = M88.load_model(model_file) if os.path.exists(model_file) else None
    params = dict(E64.XGB_PARAMS)
    params.update(V85.HPARAMS["shallow"])
    rows = [("Model", "XGBoost classifier: P(target hit before stop within "
                      "60 trading days)"),
            ("Features (all monotone 'lower is better')",
             ", ".join(M88.FEATURES)),
            ("Hyperparameters", ", ".join(
                f"{k}={params[k]}" for k in ("n_estimators", "max_depth",
                                             "learning_rate",
                                             "min_child_weight", "subsample",
                                             "colsample_bytree",
                                             "reg_lambda"))),
            ("Ensemble", f"{n_seeds} seeds ({SEED}-{SEED + n_seeds - 1}), "
                         f"probabilities averaged"),
            ("Signal rule", "score STRICTLY above the 99th percentile of the "
                            "universe's trailing 252 trading days, refit "
                            "monthly"),
            ("Six base features chosen on", "2008-2016 (V80)"),
            ("Objective, short-horizon features, shallow trees chosen on",
             "2017-2021, core stocks (V85)"),
            ("Confirmed on", "2,024 unseen stocks, pre-registered (V86); "
                             "block bootstrap (V87)")]
    if model is not None:
        p, par = model["provenance"], model.get("parity", {})
        rows += [("Trained on", f"{p['n_tickers']} stocks, {p['n_rows']:,} "
                                f"candidates, {p['from']} to "
                                f"{p['trained_through']}")]
        if "measured" in par:
            rows.append(("Parity: served builder vs measured fold models",
                         f"max |diff| {par['measured']['max_abs_diff']:.1e} "
                         f"on {par['measured']['rows']:,} rows "
                         f"({par['measured']['year']} fold)"))
        if "serving" in par:
            rows.append(("Parity: serving features vs training features",
                         f"max |diff| {par['serving']['max_abs_diff']:.1e} "
                         f"on {par['serving']['shared_rows']:,} of "
                         f"{par['serving']['train_rows']:,} rows"))
        rows.append(("Probability shown to users",
                     "yes" if model.get("display_probability") else
                     "no - binary signal and measured hit rate only"))
    R.table("t01_model_card", "1. MODEL CARD - V88",
            pd.DataFrame(rows, columns=["Item", "Value"]))

    # ---- 2. headline --------------------------------------------------------------
    rows = []
    for name, (U, _, _) in unis.items():
        s = evs[name].summary(fires[name][FINAL], HELD)
        n, top = V86.concentration(U, fires[name][FINAL], HELD)
        rows.append({"Universe": name, "Signals": s["signals"], "Stocks": n,
                     "Top-10 stocks' share": top, "Win-rate lift pp":
                     s["lift"], "90% CI": ci(s["lo"], s["hi"]),
                     "Expectancy lift R": s["expR"],
                     "Years beating own pool": s["cond"],
                     "Worst year pp": s["worst"]})
    R.table("t02_headline", f"2. HEADLINE - the final model, {HELD[0]}-"
            f"{HELD[-1]}, against random days in the same stock and month",
            pd.DataFrame(rows),
            note="The unseen-stock numbers (rows 1-2) are the honest estimate; "
                 "the curated core overstates the\nabsolute size because its "
                 "stocks are today's large companies.",
            specs={"Top-10 stocks' share": "{:.0%}"})

    # ---- 3. pre-registered test, all arms ----------------------------------------
    U, _, _ = unis[PRIMARY]
    ev, f = evs[PRIMARY], fires[PRIMARY]
    order = [FINAL, "V81", "blend V81 + bb_position", OPP, "low rsi_14",
             "low range60_position", "composite v2", "random #1",
             "random #2", "random #3"]
    rows = []
    for a in order:
        s = ev.summary(f[a], HELD)
        n, top = V86.concentration(U, f[a], HELD)
        row = {"Arm": NAMES.get(a, a), "Signals": s["signals"], "Stocks": n,
               "Win-rate lift pp": s["lift"], "90% CI": ci(s["lo"], s["hi"]),
               "Expectancy lift R": s["expR"], "Years beating own pool":
               s["cond"], "Worst year pp": s["worst"]}
        if a != OPP:
            p = ev.paired(f[a], f[OPP], HELD, k_tests=2)
            row.update({"vs rule pp": p["diff"],
                        "vs rule 90% CI": ci(p["lo"], p["hi"]),
                        "P(diff <= 0)": p["p_le0"]})
        rows.append(row)
    R.table("t03_preregistered_test",
            f"3. THE PRE-REGISTERED OUT-OF-UNIVERSE TEST - {PRIMARY}, "
            f"{HELD[0]}-{HELD[-1]}", pd.DataFrame(rows),
            note="Primary (V81 vs rule): ahead but not proven. Secondary "
                 "(final model vs rule): two-shot lower bound\nabove zero - "
                 "see table 4. 'vs rule' is paired by calendar month.",
            specs={"P(diff <= 0)": "{:.3f}"})

    # ---- 4. three universes --------------------------------------------------------
    rows = []
    for name in unis:
        ev, f = evs[name], fires[name]
        pf = ev.paired(f[FINAL], f[OPP], HELD, k_tests=2)
        pv = ev.paired(f["V81"], f[OPP], HELD, k_tests=2)
        sf, sv, so = (ev.summary(f[a], HELD) for a in (FINAL, "V81", OPP))
        rows.append({"Universe": name, "Final model lift": sf["lift"],
                     "V81 lift": sv["lift"], "Rule lift": so["lift"],
                     "Final - rule": pf["diff"], "90% CI": ci(pf["lo"],
                                                              pf["hi"]),
                     "Two-shot lower bound": pf["lo_bonf"],
                     "CI width": pf["hi"] - pf["lo"],
                     "V81 - rule": pv["diff"],
                     "V81 90% CI": ci(pv["lo"], pv["hi"])})
    R.table("t04_three_universes", f"4. AGAINST THE RULE FIXED IN ADVANCE "
            f"({OPP}), THREE UNIVERSES, {HELD[0]}-{HELD[-1]}",
            pd.DataFrame(rows),
            note="Same sign everywhere. Many more stocks per month narrow the "
                 "interval on the unseen stocks.")

    # ---- 5. selection period vs later -----------------------------------------
    rows = []
    for name in unis:
        ev, f = evs[name], fires[name]
        for tag, yrs in ((f"{VAL[0]}-{VAL[-1]} (selection years)", VAL),
                         (f"{TEST[0]}-{TEST[-1]} (after selection)", TEST),
                         (f"{HELD[0]}-{HELD[-1]}", HELD),
                         (f"{all_years[0]}-{all_years[-1]}", all_years)):
            p = ev.paired(f[FINAL], f[OPP], yrs)
            s = ev.summary(f[FINAL], yrs)
            if p and s:
                rows.append({"Universe": name, "Years": tag,
                             "Signals": s["signals"], "Lift pp": s["lift"],
                             "Final - rule": p["diff"],
                             "90% CI": ci(p["lo"], p["hi"])})
    R.table("t05_periods", "5. FINAL MODEL AGAINST THE RULE, BY PERIOD",
            pd.DataFrame(rows),
            note="The options were chosen on core 2017-21. Unseen stocks in "
                 "2022-26 are out of universe AND\nafter selection - the "
                 "cleanest slice.")

    # ---- 6. robustness (V87) -----------------------------------------------------
    bp = os.path.join(v87_dir, "block_bootstrap.csv")
    if os.path.exists(bp):
        bl = pd.read_csv(bp)
        bl = bl[bl["Universe"] == PRIMARY].copy()
        bl["Arm"] = bl["Arm"].map(lambda a: NAMES.get(a, a))
        bl["90% CI"] = [ci(a, b_) for a, b_ in zip(bl["90% lo"],
                                                    bl["90% hi"])]
        R.table("t06_block_bootstrap", "6. ROBUSTNESS TO OVERLAPPING TRADES "
                f"(V87) - {PRIMARY}, paired against the rule",
                bl[["Arm", "Years", "Block (months)", "Signals", "Diff",
                    "90% CI", "Bonf lo", "P(diff<=0)"]].rename(
                    columns={"Diff": "Lead pp",
                             "Bonf lo": "Two-shot lower bound"}),
                note="Pre-registered: the two-shot bound must stay above zero "
                     "with 3- AND 6-month blocks (it does: about +0.3).\nThe "
                     "1-month row is V86's method with different random draws.",
                specs={"P(diff<=0)": "{:.3f}"})
    ap = os.path.join(v87_dir, "autocorrelation.csv")
    if os.path.exists(ap):
        ac = pd.read_csv(ap)
        ac["Arm"] = ac["Arm"].map(lambda a: NAMES.get(a, a))
        R.table("t06b_autocorrelation", "6b. MONTH-TO-MONTH AUTOCORRELATION "
                "OF THE LEAD over the rule (V87)", ac,
                note="Near zero at every lag (notability line ~0.19): months "
                     "are close to independent, which is why\nthe blocks "
                     "barely move the interval.", specs={c: "{:+.3f}"
                                                         for c in ac.columns})

    # ---- 7. timing vs selection -------------------------------------------------
    rows_w, rows_r = [], []
    for name, (U, _, _) in unis.items():
        ev, f = evs[name], fires[name]
        bw, br = bases[name]
        for a in (FINAL, "V81", OPP, "low range60_position", "random #1"):
            d = decompose(U, ev, f[a], HELD, bw, br)
            if d is None:
                continue
            base = {"Universe": name, "Arm": NAMES.get(a, a),
                    "Signals": d["Signals"]}
            rows_w.append({**base,
                           "Fired win %": d["Fired win %"],
                           "Same stock-month %": d["Stock-month pool %"],
                           "Same month, all stocks %": d["Same-month base %"],
                           "Timing pp": d["Timing pp"],
                           "Timing CI": ci(d["Timing pp lo"],
                                           d["Timing pp hi"]),
                           "Selection pp": d["Selection pp"],
                           "Selection CI": ci(d["Selection pp lo"],
                                              d["Selection pp hi"]),
                           "Total pp": d["Total pp"],
                           "Total CI": ci(d["Total pp lo"], d["Total pp hi"])})
            rows_r.append({**base, "Fired R": d["Fired R"],
                           "Same stock-month R": d["Pool R"],
                           "Same month, all stocks R": d["Base R"],
                           "Timing R": d["Timing R"],
                           "Timing CI": ci(d["Timing R lo"], d["Timing R hi"]),
                           "Selection R": d["Selection R"],
                           "Selection CI": ci(d["Selection R lo"],
                                              d["Selection R hi"]),
                           "Total R": d["Total R"],
                           "Total CI": ci(d["Total R lo"], d["Total R hi"])})
    note = ("Fired - (same month, all stocks) = TIMING (fired day vs other "
            "days in the same stock and month)\n"
            "                                + SELECTION (the stock-months it "
            "fires in vs that month's average).\n"
            "The claim is TIMING. A negative SELECTION means the model fires in "
            "stock-months that are worse than\naverage - falling stocks - so it "
            "must be used to time stocks chosen by something else, not to "
            "pick them.")
    pct = {c: "{:.1f}" for c in ("Fired win %", "Same stock-month %",
                                 "Same month, all stocks %")}
    R.table("t07_timing_vs_selection_win",
            f"7. TIMING vs SELECTION - win rate, {HELD[0]}-{HELD[-1]}",
            pd.DataFrame(rows_w), note=note, specs=pct)
    R.table("t07b_timing_vs_selection_R",
            f"7b. TIMING vs SELECTION - expectancy in R, {HELD[0]}-{HELD[-1]}",
            pd.DataFrame(rows_r),
            note="The same split in R per trade (target and stop scale with "
                 "each stock's volatility).",
            specs={c: "{:+.3f}" for c in ("Fired R", "Same stock-month R",
                                          "Same month, all stocks R",
                                          "Timing R", "Selection R",
                                          "Total R")})

    # ---- 8. regimes ---------------------------------------------------------------
    rows = []
    for name in (PRIMARY, CORE):
        U, _, _ = unis[name]
        ev, f = evs[name], fires[name]
        br = U["breadth"].to_numpy(float)
        for a in (FINAL, "V81", OPP):
            for reg, mk in ((f"bear (< {BEAR:.2f})", br < BEAR),
                            (f"neutral", (br >= BEAR) & (br <= BULL)),
                            (f"bull (> {BULL:.2f})", br > BULL)):
                s = ev.summary(f[a] & mk, all_years)
                if s:
                    rows.append({"Universe": name, "Arm": NAMES.get(a, a),
                                 "Regime": reg, "Signals": s["signals"],
                                 "Lift vs random days pp": s["lift"],
                                 "90% CI": ci(s["lo"], s["hi"])})
    R.table("t08_regimes_vs_random", f"8. BY MARKET REGIME AT ENTRY, "
            f"{all_years[0]}-{all_years[-1]} - against random days in the "
            f"same stock and month", pd.DataFrame(rows),
            note="Regime = share of all core stocks above their own 200-day "
                 "EMA on the signal date.")
    rp = os.path.join(v86_dir, "regimes.csv")
    if os.path.exists(rp):
        rg = pd.read_csv(rp)
        rg["Arm"] = rg["Arm"].map(lambda a: NAMES.get(a, a))
        rg["90% CI"] = [ci(a, b_) for a, b_ in zip(rg["lo"], rg["hi"])]
        R.table("t08b_regimes_vs_rule", "8b. BULL AND NORMAL MARKETS - "
                "against the rule (V86)",
                rg[["Universe", "Regime", "Years", "Arm", "Signals",
                    "Arm lift", "bb lift", "vs bb", "90% CI"]].rename(
                    columns={"bb lift": "Rule lift", "vs bb": "Arm - rule"}),
                note="Positive on the filtered unseen stocks, flat on all "
                     "unseen stocks, slightly negative on the core:\nnot a "
                     "claim either way.")

    # ---- 9. per year ----------------------------------------------------------------
    for name, key, num in ((PRIMARY, "t09_per_year_fresh", "9"),
                           (CORE, "t09b_per_year_core", "9b")):
        U, _, _ = unis[name]
        ev, f = evs[name], fires[name]
        yr = U["year"].to_numpy()
        rows = []
        for y in all_years:
            r = {"Year": int(y)}
            for a, lab in ((FINAL, "Final"), ("V81", "V81"), (OPP, "Rule")):
                m = f[a] & (yr == y)
                r[f"{lab} n"] = int(m.sum())
                r[f"{lab} lift"] = ev.ex_w[m].mean() * 100 if m.any() \
                    else np.nan
            r["Final - rule"] = r["Final lift"] - r["Rule lift"]
            rows.append(r)
        t = pd.DataFrame(rows)
        h = t[t["Year"].isin(HELD)].dropna(subset=["Final - rule"])
        R.table(key, f"{num}. PER YEAR - {name} (point estimates, pp)", t,
                note=f"Final model ahead of the rule in "
                     f"{int((h['Final - rule'] > 0).sum())}/{len(h)} years "
                     f"of {HELD[0]}-{HELD[-1]}; lift > 0 in "
                     f"{int((h['Final lift'] > 0).sum())}/{len(h)}.")

    # ---- 10. how it was chosen (V85) -------------------------------------------
    sp = os.path.join(v85_dir, "all_arms.csv")
    if os.path.exists(sp):
        s85 = pd.read_csv(sp)
        keep = [c for c in ["Arm", "Kind", "val lift", "val vs V81",
                            "val vs bb", "test vs bb", "held lift",
                            "held vs bb", "max signals/yr"]
                if c in s85.columns]
        s85 = s85[keep].rename(columns={
            "val lift": "2017-21 lift", "val vs V81": "2017-21 vs V81",
            "val vs bb": "2017-21 vs rule", "test vs bb": "2022-26 vs rule",
            "held lift": "2017-26 lift", "held vs bb": "2017-26 vs rule",
            "max signals/yr": "Max signals/yr"})
        R.table("t10_selection_v85", "10. HOW THE FINAL MODEL WAS CHOSEN "
                "(V85, core stocks)", s85,
                note="Chosen = the model arm with the best 2017-21 lead over "
                     "the rule: 'combo top-3: 6 + 7 + 11a'.\nIts 2022-26 "
                     "number is the confirmation; other arms' 2022-26 numbers "
                     "are not claims.")
    sc = os.path.join(v85_dir, "seed_check.csv")
    if os.path.exists(sc):
        R.table("t10b_seed_check", "10b. SEED CHECK (V85, core) - the chosen "
                "model refit with other seeds", pd.read_csv(sc),
                note="Lead over the rule, pp.")

    # ---- 11-12. ladder and ablation ------------------------------------------------
    if run_ablation:
        print("\n  [11-12] LADDER AND ABLATION - refitting variants over "
              f"{HELD[0]}-{HELD[-1]} (cached per variant) ...")
        sig = V86._hash([len(core), len(fresh), str(core["date"].max()),
                         str(fresh["date"].max()),
                         sorted(fresh["ticker"].unique().tolist())])
        horizon = prov81["horizon"]
        feats, mono = list(M88.FEATURES), list(M88.DIRECTIONS)

        # integrity: this builder must reproduce the final model's folds
        chk, how = wf_custom(feats, mono, "shallow", core, fresh, [HELD[0]],
                             n_seeds, horizon, sig, "check", resume)
        for tag in ("core", "fresh"):
            a = chk[tag][0].set_index("row_id")["p"]
            b_ = wf["combo"][tag][0].set_index("row_id")["p"] \
                .reindex(a.index)
            dmax = float(np.nanmax(np.abs(a.to_numpy() - b_.to_numpy())))
            print(f"      builder check, {HELD[0]} fold, {tag}: max |diff| vs "
                  f"V86's final-model predictions {dmax:.1e}  "
                  f"{'IDENTICAL' if dmax < 1e-6 else '*** MISMATCH ***'}")

        short = [f_ for f_, _ in V85.SHORT]
        base6 = list(V85.BASE_FEATS)
        variants = [("+ classify (6 features)", base6, [-1] * 6, "default"),
                    ("+ short horizon (10, default trees)", feats, mono,
                     "default")]
        variants += [(f"drop {x}", [g for g in feats if g != x],
                      [m_ for g, m_ in zip(feats, mono) if g != x], "shallow")
                     for x in feats]
        variants += [("drop all 4 short-horizon", base6, [-1] * 6, "shallow"),
                     ("short-horizon only (4)", short, [-1] * 4, "shallow")]

        def fire_both(wfc):
            out = {}
            for name in (PRIMARY, CORE):
                U, _, pos = unis[name]
                tag = "core" if name == CORE else "fresh"
                te, refs = V86.restrict(*wfc[tag], pos)
                out[name] = V85.model_fire(U, te, refs)
            return out

        vf = {}
        t1 = time.time()
        for i, (lab, fs, mn, hp) in enumerate(variants, 1):
            wfc, how = wf_custom(fs, mn, hp, core, fresh, HELD, n_seeds,
                                 horizon, sig, lab, resume)
            vf[lab] = fire_both(wfc)
            left = (time.time() - t1) / i * (len(variants) - i)
            print(f"      [{i:>2}/{len(variants)}] {lab:<38} {how:>7}"
                  + (f"   ~{left / 60:.0f} min left" if left > 60 else ""))
            del wfc

        # 11. ladder
        steps = [("V81 (regression, 6 features)", {n: fires[n]["V81"]
                                                   for n in (PRIMARY, CORE)}),
                 ("+ classify objective", vf["+ classify (6 features)"]),
                 ("+ short-horizon features",
                  vf["+ short horizon (10, default trees)"]),
                 ("+ shallow trees = final model",
                  {n: fires[n][FINAL] for n in (PRIMARY, CORE)})]
        rows = []
        for name in (PRIMARY, CORE):
            ev, f = evs[name], fires[name]
            prev = None
            for lab, fm in steps:
                s = ev.summary(fm[name], HELD)
                po = ev.paired(fm[name], f[OPP], HELD)
                row = {"Universe": name, "Step": lab, "Signals": s["signals"],
                       "Lift pp": s["lift"], "vs rule": po["diff"],
                       "vs rule CI": ci(po["lo"], po["hi"])}
                if prev is not None:
                    pp = ev.paired(fm[name], prev, HELD)
                    row.update({"Step gain pp": pp["diff"],
                                "Step gain CI": ci(pp["lo"], pp["hi"])})
                rows.append(row)
                prev = fm[name]
        R.table("t11_ladder", f"11. THE LADDER FROM V81 TO THE FINAL MODEL, "
                f"{HELD[0]}-{HELD[-1]}", pd.DataFrame(rows),
                note="Each step adds one V85 option. 'Step gain' is paired "
                     "against the step before. On the core\n2017-21 is where "
                     "the steps were chosen; the unseen stocks are the fair "
                     "read.")

        # 12. ablation
        rows = []
        for lab, fs, mn, hp in variants[2:]:
            row = {"Variant": lab, "Features": len(fs)}
            for name, short_n in ((PRIMARY, "unseen"), (CORE, "core")):
                ev, f = evs[name], fires[name]
                s = ev.summary(vf[lab][name], HELD)
                p = ev.paired(f[FINAL], vf[lab][name], HELD)
                row.update({f"{short_n} lift": s["lift"] if s else np.nan,
                            f"{short_n}: final minus variant": p["diff"]
                            if p else np.nan,
                            f"{short_n} CI": ci(p["lo"], p["hi"]) if p
                            else ""})
            rows.append(row)
        ab = pd.DataFrame(rows).sort_values("unseen: final minus variant",
                                            ascending=False)
        R.table("t12_ablation", f"12. FEATURE ABLATION - drop and refit, "
                f"{HELD[0]}-{HELD[-1]} (importance = what the model loses)",
                ab,
                note="'final minus variant' > 0: removing it HURTS (the model "
                     "depends on it). An interval that\nincludes zero: the "
                     "other features cover for it. Measured on unseen stocks "
                     "first.")

    # ---- 13. calibration --------------------------------------------------------
    if model is not None and model.get("calibration", {}).get("table"):
        cal = pd.DataFrame(model["calibration"]["table"])
        st = pd.DataFrame(model["calibration"]["stats"])
        R.table("t13_calibration", "13. CALIBRATION OF THE PROBABILITIES "
                f"(V88, fold models, {HELD[0]}-{HELD[-1]})", cal,
                note="Mean predicted probability against the actual win rate, "
                     "by quintile of the score.",
                specs={"Mean p": "{:.3f}", "Win rate": "{:.3f}"})
        st = st.rename(columns={"universe": "Universe", "skill":
                                "Brier skill", "fired_gap":
                                "Fired: mean p - hit rate", "fired_n":
                                "Fired signals"})
        R.table("t13b_calibration_summary", "13b. CALIBRATION SUMMARY",
                st[["Universe", "Brier skill", "Fired: mean p - hit rate",
                    "Fired signals"]],
                note=f"Display rule: |mean p - hit rate| <= "
                     f"{M88.CALIB_TOL:.0%} on fired signals in BOTH universes. "
                     f"Decision: "
                     + ("show the probability" if model.get(
                         "display_probability") else
                        "do NOT show a probability - binary signal only") + ".",
                specs={"Brier skill": "{:+.4f}",
                       "Fired: mean p - hit rate": "{:+.3f}"})

    R.save(f"# V89 - results tables for the final entry model (V88)\n\n"
           f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Integrity vs "
           f"V86: {'REPRODUCED' if ok else 'NOT CHECKED / DIFFERENT'}.")
    print(f"  total {(time.time() - t0) / 60:.1f} min")


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v88.joblib"
    RUN_MODEL_IN81  = "entry_model_v81.joblib"  # provenance (cache, horizon)
    RUN_CORE_CACHE  = None              # None = the cache V81 was trained on
    RUN_BIG_CACHE   = "price_cache_v68"
    RUN_N_SEEDS     = 5                 # same as V86
    RUN_CHUNK       = 250               # same as V86 (cache key)
    RUN_N_BOOT      = 4000              # same as V86, so its intervals reproduce
    RUN_ABLATION    = True              # tables 11-12: ~25-30 min first time
    RUN_RESUME      = True              # reuse cached ablation variants
    RUN_OUT_DIR     = "thesis_tables_v89"
    # -------------------------------------------------------------------------

    run_v89(model_file=RUN_MODEL_FILE, model_in81=RUN_MODEL_IN81,
            core_cache=RUN_CORE_CACHE, big_cache=RUN_BIG_CACHE,
            n_seeds=RUN_N_SEEDS, chunk=RUN_CHUNK, n_boot=RUN_N_BOOT,
            run_ablation=RUN_ABLATION, resume=RUN_RESUME,
            out_dir=RUN_OUT_DIR)
