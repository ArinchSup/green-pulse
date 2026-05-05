// src/types.ts — shared types

export type SectorKey = "Index" | "Currency" | "Commodity" | "Crypto" | "Equity";
export type RangeKey = "1D" | "1W" | "1M" | "3M" | "1Y" | "5Y";
export type Impact = "up" | "down" | "neutral";

export interface PricePoint {
  i: number;
  t: number;
  value: number;
}

export interface Market {
  id: string;
  label: string;
  ticker: string;
  sector: SectorKey;
  base: number;
  price: number;
  change: number;
  up: boolean;
  data: Record<RangeKey, PricePoint[]>;
}

export interface Holding {
  ticker: string;
  shares: number;
  avgCost: number;
}

export interface Transaction {
  id: number;
  date: string;
  type: "BUY" | "SELL";
  ticker: string;
  shares: number;
  price: number;
  status: string;
}

export interface NewsItem {
  time: string;
  source: string;
  tag: string;
  headline: string;
  impact: Impact;
}

export interface Alert {
  id: number;
  ticker: string;
  condition: ">" | "<";
  target: number;
  active: boolean;
}

export type Tone = "up" | "down" | "neutral" | "warn";
