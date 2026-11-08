"""
pillar2_v43_universe.py - who the model LEARNS from vs who it TRADES.

The V40/V42 training universe was the 62 SHAY tickers: names picked in 2026 because
they had already worked, roughly 25 of which did not trade before 2021. That capped
training at 2022-09 to 2025-06 — one uninterrupted recovery — and it is almost
certainly why V42 learned to buy anything below its 200-day EMA (rho(conf, gap_ema200)
= -0.376). In that window the dip always came back. The model has never seen a month
where it didn't.

V43 separates the two roles:

  TRAINING_UNIVERSE - a few hundred liquid US names with history back to ~2005, so
    the training set spans 2008, 2011, 2015-16, 2018, 2020 and 2022. The model learns
    "what does a setup like this usually do", not "what does NVDA do".

  DEPLOY_UNIVERSE - the SHAY list, unchanged. Train broad, trade narrow.

!! SURVIVORSHIP WARNING. This list is written from today's listed companies, so names
   that went bankrupt or were acquired are missing, and the training set inherits a
   mild "everyone here survived" bias. That is far better than 62 hand-picked winners
   but it is NOT point-in-time. If your institution has CRSP, Sharadar or Norgate
   delisted-inclusive data, replace load_training_universe() with a point-in-time
   constituent query and re-run. Treat absolute return numbers from this universe as
   optimistic; the model-vs-random COMPARISON is unaffected, since both arms draw
   from the same list.

Tickers whose history does not cover the requested start date are dropped
automatically by the generator, so a wrong guess here degrades gracefully.
"""

BENCHMARK = "SPY"          # broad market for relative strength and regime
BENCHMARK_ALT = "QQQ"      # kept for comparison with V41/V42 reports

# ---------------------------------------------------------------------------
# TRAINING UNIVERSE — curated for long, continuous history and real liquidity.
# Grouped only for readability; the model never sees these group labels.
# ---------------------------------------------------------------------------
_TECH = """
AAPL MSFT INTC CSCO ORCL IBM TXN QCOM ADBE CRM NVDA AMD MU AMAT LRCX KLAC ADI MCHP
SWKS AVGO HPQ STX WDC NTAP JNPR AKAM CTSH INFY ACN IT GLW TER ON MRVL ADSK ANSS
CDNS SNPS VRSN MSI ZBRA TDY TRMB INTU ADP PAYX FFIV CIEN
""".split()

_INTERNET = """
GOOGL AMZN EBAY NFLX BKNG EXPE IAC
""".split()

_COMM = """
T VZ TMUS CMCSA DIS CHTR OMC IPG
""".split()

_FINANCE = """
JPM BAC WFC C GS MS USB PNC TFC SCHW BLK AXP COF DFS BK STT TROW AMP MET PRU AFL
ALL TRV PGR CB AIG V MA ICE CME NDAQ SPGI MCO FITB KEY RF HBAN CFG MTB NTRS
""".split()

_HEALTH = """
JNJ PFE MRK ABT LLY BMY AMGN GILD BIIB REGN VRTX UNH CI HUM CVS MCK CAH BDX SYK
BSX MDT ZBH EW ISRG TMO DHR A RMD IDXX ALGN WAT MTD HOLX DGX LH BAX
""".split()

_ENERGY = """
XOM CVX COP EOG OXY SLB HAL VLO MPC PSX KMI WMB OKE DVN HES APA CTRA
""".split()

_INDUSTRIAL = """
BA CAT DE HON GE MMM UPS FDX LMT NOC GD RTX EMR ETN ITW PH ROK CMI PCAR NSC UNP
CSX ODFL JBHT CHRW EXPD WM RSG URI FAST GWW SWK DOV AME ROP TT JCI TXT LII SNA
""".split()

_CONSUMER = """
PG KO PEP PM MO COST WMT TGT HD LOW MCD SBUX NKE TJX ROST DG DLTR YUM CMG DPZ KR
SYY GIS K HSY CL KMB CHD CLX EL STZ TAP MNST ORLY AZO AAP BBY ULTA LULU DECK CROX
WHR NWL HAS MAT
""".split()

_MATERIALS = """
APD SHW ECL NEM FCX NUE STLD VMC MLM DD PPG ALB IP PKG SEE CF MOS
""".split()

_UTILITIES = """
NEE DUK SO D AEP EXC XEL ED WEC ES PEG SRE AES NI CMS DTE PPL FE
""".split()

_REITS = """
AMT PLD CCI EQIX SPG PSA O VTR WELL DLR ESS AVB EQR MAA UDR HST KIM REG
""".split()

_TRANSPORT_AUTO = """
F GM TSLA LUV DAL UAL ALK
""".split()

TRAINING_UNIVERSE = sorted(set(
    _TECH + _INTERNET + _COMM + _FINANCE + _HEALTH + _ENERGY + _INDUSTRIAL
    + _CONSUMER + _MATERIALS + _UTILITIES + _REITS + _TRANSPORT_AUTO
))

# ---------------------------------------------------------------------------
# DEPLOY UNIVERSE — unchanged from V40/V42, so backtests stay comparable.
# ---------------------------------------------------------------------------
DEPLOY_UNIVERSE = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI", "CRDO",
]


def load_training_universe(include_deploy=True):
    """
    Tickers to BUILD the dataset from.

    include_deploy=True adds the deploy names too. They contribute only their own
    short histories, so they cannot widen the date range, but they let the model see
    the kind of high-volatility name it will actually be asked about. Set False for a
    stricter separation between what it learns from and what it trades.
    """
    names = set(TRAINING_UNIVERSE)
    if include_deploy:
        names |= set(DEPLOY_UNIVERSE)
    return sorted(names)


def universe_summary():
    return (f"training {len(TRAINING_UNIVERSE)} + deploy {len(DEPLOY_UNIVERSE)} "
            f"= {len(load_training_universe())} unique tickers")


if __name__ == "__main__":
    print(universe_summary())
    print(f"\ntraining-only names not in deploy: "
          f"{len(set(TRAINING_UNIVERSE) - set(DEPLOY_UNIVERSE))}")
    print(f"deploy names also in training:     "
          f"{sorted(set(TRAINING_UNIVERSE) & set(DEPLOY_UNIVERSE))}")
