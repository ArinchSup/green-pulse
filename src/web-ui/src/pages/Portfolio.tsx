// src/pages/Portfolio.tsx
import { useMemo } from "react";
import type { Market, Holding, Transaction } from "../types";
import { LineChart, Donut, Pill } from "../primitives";
import { fmtPrice, fmtPct, fmtMoney } from "../format";
import { findMarket } from "../variable";

const Kpi = ({ label, value, sub, tone, big }: {
  label: string; value: string | number; sub?: string; tone?: "up" | "down"; big?: boolean;
}) => (
  <div className={`kpi ${big ? "big" : ""}`}>
    <div className="kpi-label">{label}</div>
    <div className={`kpi-value ${tone || ""}`}>{value}</div>
    {sub && <div className={`kpi-sub ${tone || ""}`}>{sub}</div>}
  </div>
);

export const Portfolio = ({ markets, holdings, transactions, onTrade }: {
  markets: Market[]; holdings: Holding[]; transactions: Transaction[]; onTrade: (ticker: string) => void;
}) => {
  const enriched = holdings.map(h => {
    const m = findMarket(h.ticker, markets)!;
    const value = h.shares * m.price;
    const cost = h.shares * h.avgCost;
    const pl = value - cost;
    const plPct = (pl / cost) * 100;
    return { ...h, m, value, cost, pl, plPct };
  });
  const total = enriched.reduce((a, b) => a + b.value, 0);
  const totalCost = enriched.reduce((a, b) => a + b.cost, 0);
  const totalPL = total - totalCost;
  const totalPLPct = (totalPL / totalCost) * 100;
  const dayPL = enriched.reduce((a, b) => a + (b.value * b.m.change / 100), 0);

  const colors = ["#00d46a","#39ff8e","#7be8a3","#44aaff","#bb88ff","#ffcc44","#ff4466"];
  const slices = enriched.map((h, i) => ({ label: h.ticker, value: h.value, color: colors[i % colors.length] })).sort((a, b) => b.value - a.value);

  const portfolioSeries = useMemo(() => {
    let p = total * 0.92;
    return Array.from({ length: 30 }, (_, i) => {
      p = p + (Math.random() - 0.42) * total * 0.012;
      if (i === 29) p = total;
      return { i, t: i, value: parseFloat(p.toFixed(2)) };
    });
  }, [total]);

  return (
    <div className="page">
      <div className="kpi-row">
        <Kpi label="Total Value"  value={fmtMoney(total)}    big />
        <Kpi label="Today P&L"    value={fmtMoney(dayPL)}    tone={dayPL >= 0 ? "up" : "down"}  sub={fmtPct(dayPL/total*100)} />
        <Kpi label="Total P&L"    value={fmtMoney(totalPL)}  tone={totalPL >= 0 ? "up" : "down"} sub={fmtPct(totalPLPct)} />
        <Kpi label="Cost Basis"   value={fmtMoney(totalCost)} />
        <Kpi label="Positions"    value={enriched.length} />
        <Kpi label="Buying Power" value={fmtMoney(24580.42)} />
      </div>
      <div className="grid-portfolio">
        <div className="card">
          <div className="card-title">
            <span>Portfolio · 30D</span>
            <Pill tone={totalPL >= 0 ? "up" : "down"}>{fmtPct(totalPLPct)}</Pill>
          </div>
          <LineChart data={portfolioSeries} up={totalPL >= 0} height={220} />
        </div>
        <div className="card">
          <div className="card-title">Allocation</div>
          <div className="allocation">
            <Donut slices={slices} size={180} thickness={22} />
            <div className="alloc-legend">
              {slices.map(s => (
                <div key={s.label} className="alloc-row">
                  <span className="alloc-dot" style={{ background: s.color }}></span>
                  <span className="alloc-label">{s.label}</span>
                  <span className="alloc-pct">{((s.value/total)*100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="card">
        <div className="card-title">Holdings</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Symbol</th><th>Shares</th><th>Avg Cost</th><th>Price</th>
              <th>Day</th><th>Market Value</th><th>Total P&L</th><th>Allocation</th><th></th>
            </tr>
          </thead>
          <tbody>
            {enriched.map(h => (
              <tr key={h.ticker}>
                <td>
                  <div className="sym-cell">
                    <div className="sym-tk">{h.ticker}</div>
                    <div className="sym-name">{h.m.label}</div>
                  </div>
                </td>
                <td className="num">{h.shares}</td>
                <td className="num">${fmtPrice(h.avgCost)}</td>
                <td className="num">${fmtPrice(h.m.price)}</td>
                <td className={h.m.up ? "num up" : "num down"}>{fmtPct(h.m.change)}</td>
                <td className="num">{fmtMoney(h.value)}</td>
                <td className={h.pl >= 0 ? "num up" : "num down"}>
                  {fmtMoney(h.pl)} <span className="dim">({fmtPct(h.plPct)})</span>
                </td>
                <td><div className="alloc-bar"><div style={{ width: `${(h.value/total)*100}%` }} /></div></td>
                <td><button className="mini-btn" onClick={() => onTrade(h.ticker)}>Trade</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <div className="card-title">Recent Activity</div>
        <table className="data-table">
          <thead><tr><th>Date</th><th>Type</th><th>Symbol</th><th>Shares</th><th>Price</th><th>Total</th><th>Status</th></tr></thead>
          <tbody>
            {transactions.map(t => (
              <tr key={t.id}>
                <td className="dim">{t.date}</td>
                <td><Pill tone={t.type === "BUY" ? "up" : "down"}>{t.type}</Pill></td>
                <td className="bold">{t.ticker}</td>
                <td className="num">{t.shares}</td>
                <td className="num">${fmtPrice(t.price)}</td>
                <td className="num">{fmtMoney(t.shares * t.price)}</td>
                <td><Pill tone="neutral">{t.status}</Pill></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
