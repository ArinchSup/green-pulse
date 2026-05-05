// src/pages/Watchlist.tsx
import { useMemo, useState } from "react";
import type { Market } from "../types";
import { Sparkline } from "../primitives";
import { fmtPrice, fmtPct } from "../format";

export const Watchlist = ({ markets, onSelect, onTrade, watched, toggleWatch }: {
  markets: Market[];
  onSelect: (id: string) => void;
  onTrade: (ticker: string) => void;
  watched: Set<string>;
  toggleWatch: (id: string) => void;
}) => {
  type SortCol = "ticker" | "sector" | "price" | "change";
  const [sortBy, setSortBy] = useState<SortCol>("change");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<string>("All");

  const sectors = ["All", ...new Set(markets.map(m => m.sector))];

  const sorted = useMemo(() => {
    let list = markets.filter(m =>
      (filter === "All" || m.sector === filter) &&
      (q === "" || m.ticker.toLowerCase().includes(q.toLowerCase()) || m.label.toLowerCase().includes(q.toLowerCase()))
    );
    list = [...list].sort((a, b) => {
      const av = a[sortBy] as any, bv = b[sortBy] as any;
      const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
      return dir === "asc" ? cmp : -cmp;
    });
    return list;
  }, [markets, sortBy, dir, q, filter]);

  const click = (col: SortCol) => {
    if (sortBy === col) setDir(dir === "asc" ? "desc" : "asc");
    else { setSortBy(col); setDir("desc"); }
  };

  const Header = ({ col, children, align = "left" }: { col: SortCol; children: React.ReactNode; align?: "left" | "right" }) => (
    <th onClick={() => click(col)} style={{ textAlign: align, cursor: "pointer" }}>
      {children}
      <span className="sort-arrow">{sortBy === col ? (dir === "asc" ? "▲" : "▼") : "↕"}</span>
    </th>
  );

  return (
    <div className="page">
      <div className="watchlist-toolbar">
        <input className="search-input" placeholder="Search ticker or name…" value={q} onChange={e => setQ(e.target.value)} />
        <div className="filter-chips">
          {sectors.map(s => (
            <button key={s} className={`chip ${filter === s ? "active" : ""}`} onClick={() => setFilter(s)}>{s}</button>
          ))}
        </div>
      </div>
      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th></th>
              <Header col="ticker">Symbol</Header>
              <Header col="sector">Sector</Header>
              <Header col="price" align="right">Price</Header>
              <Header col="change" align="right">Day Δ</Header>
              <th style={{ textAlign: "center" }}>Trend (1W)</th>
              <th style={{ textAlign: "right" }}></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(m => (
              <tr key={m.id}>
                <td>
                  <button className="watch-star" onClick={() => toggleWatch(m.id)} aria-label="Toggle watch">
                    {watched.has(m.id) ? "★" : "☆"}
                  </button>
                </td>
                <td onClick={() => onSelect(m.id)} style={{ cursor: "pointer" }}>
                  <div className="sym-cell">
                    <div className="sym-tk">{m.ticker}</div>
                    <div className="sym-name">{m.label}</div>
                  </div>
                </td>
                <td className="dim">{m.sector}</td>
                <td className="num">${fmtPrice(m.price)}</td>
                <td className={m.up ? "num up" : "num down"}>{m.up ? "▲" : "▼"} {fmtPct(m.change)}</td>
                <td style={{ display: "flex", justifyContent: "center" }}>
                  <Sparkline data={m.data["1W"]} up={m.up} width={120} height={28} />
                </td>
                <td style={{ textAlign: "right" }}>
                  <button className="mini-btn" onClick={() => onTrade(m.ticker)}>Trade</button>
                </td>
              </tr>
            ))}
            {sorted.length === 0 && (<tr><td colSpan={7} className="empty">No matches.</td></tr>)}
          </tbody>
        </table>
      </div>
    </div>
  );
};
