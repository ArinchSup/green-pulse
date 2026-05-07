// src/variable.ts — market data + helpers
import type {
  Market, Holding, Transaction, NewsItem, Alert,
  RangeKey, SectorKey, PricePoint
} from "./types";

const seedRandom = (seed: number) => {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
};

const genSeries = (base: number, count: number, vol: number, seed: number): PricePoint[] => {
  const rnd = seedRandom(seed);
  let p = base;
  const out: PricePoint[] = [];
  for (let i = 0; i < count; i++) {
    const move = (rnd() - 0.485) * base * vol;
    p = Math.max(base * 0.6, p + move);
    out.push({ i, t: i, value: parseFloat(p.toFixed(2)) });
  }
  return out;
};

export const RANGES: Record<RangeKey, { count: number; vol: number; label: string }> = {
  "1D": { count: 78,  vol: 0.004, label: "Today" },
  "1W": { count: 84,  vol: 0.008, label: "1 Week" },
  "1M": { count: 90,  vol: 0.013, label: "1 Month" },
  "3M": { count: 120, vol: 0.018, label: "3 Months" },
  "1Y": { count: 180, vol: 0.025, label: "1 Year" },
  "5Y": { count: 240, vol: 0.04,  label: "5 Years" },
};

export const RANGE_KEYS: RangeKey[] = ["1D","1W","1M","3M","1Y","5Y"];

const buildMarket = (
  id: string, label: string, ticker: string,
  base: number, dayChange: number, seed: number, sector: SectorKey
): Market => {
  const data = {} as Record<RangeKey, PricePoint[]>;
  (Object.entries(RANGES) as [RangeKey, typeof RANGES[RangeKey]][]).forEach(([k, cfg], idx) => {
    data[k] = genSeries(base, cfg.count, cfg.vol, seed + idx * 7);
  });
  const cur = base * (1 + dayChange / 100);
  data["1D"][data["1D"].length - 1].value = parseFloat(cur.toFixed(2));
  return {
    id, label, ticker, sector, base,
    price: parseFloat(cur.toFixed(2)),
    change: dayChange,
    up: dayChange >= 0,
    data,
  };
};

export const MARKETS: Market[] = [
  buildMarket("nvda",   "NVIDIA",       "NVDA",  892.55,    3.41, 83,  "Equity"),
  buildMarket("aapl",   "Apple",        "AAPL",  189.84,   -0.62, 89,  "Equity"),
  buildMarket("tsla",   "Tesla",        "TSLA",  248.12,   -2.11, 97,  "Equity"),
  buildMarket("msft",   "Microsoft",    "MSFT",  421.30,    0.84, 101, "Equity"),
];

export const HOLDINGS: Holding[] = [
  { ticker: "NVDA", shares: 42,   avgCost: 612.40   },
  { ticker: "AAPL", shares: 120,  avgCost: 174.20   },
  { ticker: "MSFT", shares: 38,   avgCost: 380.10   },
];

export const TRANSACTIONS: Transaction[] = [
  { id: 1, date: "May 04", type: "BUY",  ticker: "NVDA", shares: 8,    price: 884.10,   status: "filled" },
  { id: 2, date: "May 04", type: "SELL", ticker: "TSLA", shares: 15,   price: 252.40,   status: "filled" },
  { id: 4, date: "May 03", type: "BUY",  ticker: "MSFT", shares: 6,    price: 418.92,   status: "filled" },
  { id: 5, date: "May 02", type: "SELL", ticker: "AAPL", shares: 20,   price: 191.20,   status: "filled" },
];

export const NEWS: NewsItem[] = [
  { time: "08:42", source: "Reuters",   tag: "MACRO",    headline: "Fed minutes signal patience on cuts as inflation cools toward target", impact: "neutral" },
  { time: "08:31", source: "Bloomberg", tag: "TECH",     headline: "NVIDIA partners with three sovereign AI labs for next-gen H200 deployment", impact: "up" },
  { time: "07:36", source: "CNBC",      tag: "EARNINGS", headline: "Apple revenue beats; services growth offsets iPhone softness in China", impact: "up" },
];

export const ALERTS: Alert[] = [
  { id: 1, ticker: "NVDA", condition: ">", target: 900,    active: true  },
  { id: 3, ticker: "TSLA", condition: ">", target: 260,    active: false },
];

export const findMarket = (ticker: string, list: Market[] = MARKETS): Market | undefined =>
  list.find(m => m.ticker === ticker);

export const NavItems = [
  { id: "overview",  label: "Overview"  },
  { id: "portfolio", label: "Portfolio" },
  { id: "watchlist", label: "Watchlist" },
  { id: "settings",  label: "Settings"  },
];
